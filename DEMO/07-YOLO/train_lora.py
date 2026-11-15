#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO11n + LoRA 在 TT100K 上的微调入口（单卡 / 双卡 DDP 均可）。

核心做法：继承 Ultralytics 的 `DetectionTrainer`，重写 `get_model()` 工厂方法，
在“预训练权重已加载、模型即将交给 trainer”的那一刻注入 LoRA。

为什么必须挂在 `get_model` 上（而不是先 `YOLO("yolo11n.pt")` 注入再 `model.train()`）：
`BaseTrainer.setup_model()` 内部会根据 ckpt 的 yaml **重新 new 一个 DetectionModel** 再
`load(weights)`，之前注入的 LoRA 结构会被丢弃（key 对不上，静默丢失）。挂在工厂方法上才是唯一
稳定的注入点。

双卡说明：
- `batch` 是**所有卡加起来的总 batch**，Ultralytics 内部会 `batch // world_size` 分到每张卡；
- `nbs`（nominal batch size）按“每卡 batch”给，累加步数 = round(nbs / 每卡batch)；
- LoRA 超参通过环境变量传给 DDP 子进程（子进程由 cloudpickle + 临时脚本拉起，只继承 args，
  不继承你自己挂在 trainer 实例上的属性）。

常用命令::

    # 单卡
    python train_lora.py --data /data/tt100k_yolo/data.yaml --device 0 --batch 32

    # 双 V100 16G
    python train_lora.py --data /data/tt100k_yolo/data.yaml --device 0,1 --batch 64 --nbs 32

    # 只调 det head，r=8，显存和参数量都更省
    python train_lora.py --data .../data.yaml --lora-targets head --lora-r 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lora import (  # noqa: E402
    fold_ckpt,
    inject_lora,
    lora_parameter_stats,
    lora_state_dict,
    load_lora_state_dict,
)

from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.utils import LOGGER, RANK  # noqa: E402

try:  # 8.3+ 从这里导出；老版本在 ultralytics.cfg
    from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT
except ImportError:  # pragma: no cover
    from ultralytics.cfg import DEFAULT_CFG, DEFAULT_CFG_DICT

LORA_ENV = "YOLO11_LORA_CFG"
DEFAULT_LORA = {
    "enabled": True,
    "r": 8,
    "alpha": 16.0,
    "targets": "all",
    "exclude": [".dfl"],
    "freeze_bn": True,
    "freeze_norm": True,
    "init_from": None,
}


def lora_cfg_from_env() -> dict:
    cfg = dict(DEFAULT_LORA)
    raw = os.environ.get(LORA_ENV)
    if raw:
        try:
            cfg.update(json.loads(raw))
        except json.JSONDecodeError:
            LOGGER.warning(f"环境变量 {LORA_ENV} 不是合法 JSON，使用默认 LoRA 配置")
    return cfg


class LoRATrainer(DetectionTrainer):
    """注入 LoRA 并只训练 LoRA 分支的 DetectionTrainer。"""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.lora_cfg = lora_cfg_from_env()
        self.lora_stats = {}
        self._lora_injected = False

    # ---- 关键钩子：模型工厂，预训练权重已 load 完毕 ----
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if not self._lora_injected and weights is not None:
            # 记录“基座权重里没有、或形状对不上”的模块（TT100K 45 类时就是 3 个分类卷积）。
            # 这些模块是随机初始化的，必须随 LoRA 一起存下来才能精确重建模型。
            try:
                base_sd = weights.state_dict()
                model_sd = model.state_dict()
                self._non_base_modules = {
                    k.rsplit(".", 1)[0]
                    for k, v in model_sd.items()
                    if k not in base_sd or tuple(base_sd[k].shape) != tuple(v.shape)
                }
            except Exception as e:
                LOGGER.warning(f"[LoRA] 统计非基座模块失败（不影响训练）: {e}")
                self._non_base_modules = None
        if not self.lora_cfg.get("enabled", True) or self._lora_injected:
            return model

        self.lora_stats = inject_lora(
            model,
            targets=self.lora_cfg["targets"],
            r=int(self.lora_cfg["r"]),
            alpha=float(self.lora_cfg["alpha"]),
            exclude=tuple(self.lora_cfg.get("exclude") or ()),
            freeze_norm=bool(self.lora_cfg.get("freeze_norm", True)),
            freeze_others=True,
            verbose=RANK in {-1, 0},
        )
        self._lora_injected = True

        init_from = self.lora_cfg.get("init_from")
        if init_from:
            state = torch.load(init_from, map_location="cpu", weights_only=False)
            sd = state["lora"] if isinstance(state, dict) and "lora" in state else state
            load_lora_state_dict(model, sd)
            if RANK in {-1, 0}:
                LOGGER.info(f"[LoRA] 已热启动 LoRA 权重: {init_from}")
        return model

    # ---- 冻结 BN 统计量：LoRA 微调时不要让 BN 的 running stats 被少量数据带偏 ----
    def _model_train(self):
        super()._model_train()
        if self._lora_injected and self.lora_cfg.get("freeze_bn", True):
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


# --------------------------------------------------------------------- 参数
def build_parser():
    p = argparse.ArgumentParser(
        description="YOLO11n LoRA 微调 TT100K", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 数据 / 模型
    p.add_argument("--model", default="yolo11n.pt", help="COCO 预训练权重")
    p.add_argument("--data", required=True, help="tt100k2yolo.py 生成的 data.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640, help="TT100K 小目标多，1280 精度更好但更慢")
    p.add_argument("--batch", type=int, default=64, help="**所有卡的总 batch**")
    p.add_argument("--nbs", type=int, default=32, help="nominal batch，按每卡 batch 给；控制梯度累加")
    p.add_argument("--device", default="0,1", help="'0' 单卡，'0,1' 双卡 DDP，'' 自动")
    p.add_argument("--workers", type=int, default=3, help="每个 rank 的 dataloader 进程；8 核建议 2~3")
    p.add_argument("--cache", choices=["off", "ram", "disk"], default="off",
                   help="16G 内存不要用 ram；disk 会写 ~7GB npy 缓存换速度")
    p.add_argument("--fraction", type=float, default=1.0, help="只用训练集的一部分（0~1），冒烟测试用")
    p.add_argument("--project", default="runs/tt100k")
    p.add_argument("--name", default="lora_r8")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--save-period", type=int, default=5, help="每 N 个 epoch 存一次 epochN.pt")
    # 优化
    p.add_argument("--optimizer", default="AdamW", choices=["AdamW", "Adam", "SGD", "auto"])
    p.add_argument("--lr0", type=float, default=1e-3, help="LoRA 常用 5e-4 ~ 2e-3")
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--cos-lr", choices=["on", "off"], default="on")
    p.add_argument("--amp", choices=["on", "off"], default="on", help="V100 支持 fp16 张量核，建议开")
    # 增强（交通标志域的先验）
    p.add_argument("--mosaic", type=float, default=0.5, help="小目标多，mosaic 过强会切碎标志，0.5 起步")
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--fliplr", type=float, default=0.0, help="镜像会让文字/数字反写，交通标志必须 0")
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.7)
    p.add_argument("--hsv-v", type=float, default=0.4)
    # 流程
    p.add_argument("--val", choices=["on", "off"], default="on")
    p.add_argument("--plots", choices=["on", "off"], default="on")
    p.add_argument("--multi-scale", type=float, default=0.0)
    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-targets", default="all",
                   help="all / 1x1 / 3x3 / head，可用 + 组合，如 head+1x1")
    p.add_argument("--lora-exclude", default=".dfl", help="逗号分隔的排除子串")
    p.add_argument("--no-freeze-bn", action="store_true", help="允许更新 BN running stats")
    p.add_argument("--tune-norm", action="store_true",
                   help="放开 BN 的 gamma/beta 一起训练（不再是纯 LoRA，域差异大时通常略有提升）")
    p.add_argument("--resume-lora", default=None,
                   help="从 lora_last.pt 热启动（Ultralytics 原生 resume 不适用于 LoRA，用这个续训）")
    return p


def to_overrides(a) -> dict:
    ov = {
        "model": a.model,
        "data": a.data,
        "epochs": a.epochs,
        "imgsz": a.imgsz,
        "batch": a.batch,
        "nbs": a.nbs,
        "device": a.device,
        "workers": a.workers,
        "cache": False if a.cache == "off" else ("disk" if a.cache == "disk" else True),
        "fraction": a.fraction,
        "project": a.project,
        "name": a.name,
        "exist_ok": a.exist_ok,
        "seed": a.seed,
        "patience": a.patience,
        "save_period": a.save_period,
        "optimizer": a.optimizer,
        "lr0": a.lr0,
        "lrf": a.lrf,
        "momentum": a.momentum,
        "weight_decay": a.weight_decay,
        "warmup_epochs": a.warmup_epochs,
        "cos_lr": a.cos_lr == "on",
        "amp": a.amp == "on",
        "mosaic": a.mosaic,
        "close_mosaic": a.close_mosaic,
        "scale": a.scale,
        "translate": a.translate,
        "degrees": a.degrees,
        "fliplr": a.fliplr,
        "flipud": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": a.hsv_h,
        "hsv_s": a.hsv_s,
        "hsv_v": a.hsv_v,
        "multi_scale": a.multi_scale,
        "rect": False,
        "val": a.val == "on",
        "plots": a.plots == "on",
        "pretrained": True,
        "verbose": True,
    }
    # 不同 ultralytics 版本参数集合有差异，过滤掉不认识的 key，避免 check_dict_alignment 报错
    return {k: v for k, v in ov.items() if k in DEFAULT_CFG_DICT}


def describe_plan(a):
    world = len([d for d in str(a.device).split(",") if d.strip()]) if a.device else 1
    per_gpu = max(a.batch // max(world, 1), 1)
    accum = max(round(a.nbs / per_gpu), 1)
    print("=" * 66)
    print(f"模型      : {a.model}   LoRA r={a.lora_r} alpha={a.lora_alpha} targets={a.lora_targets}")
    print(f"数据      : {a.data}")
    print(f"设备      : {a.device}  (world_size={world})")
    print(f"batch     : 总 {a.batch} → 每卡 {per_gpu}，梯度累加 {accum} → 等效 batch {per_gpu * world * accum}")
    print(f"imgsz     : {a.imgsz}   AMP: {a.amp}   workers/rank: {a.workers} (合计 {a.workers * world})")
    print(f"优化器    : {a.optimizer} lr0={a.lr0} wd={a.weight_decay} epochs={a.epochs}")
    print("=" * 66)


def save_lora_callback(trainer):
    """每次存 ckpt 时顺手把 LoRA 权重单独存一份（小、可直接热启动）。

    除了 LoRA 张量，还额外存一份检测头（`model.<last>.`）的基座权重：
    TT100K 只有 45 类，分类头的形状与 COCO 80 类不同，这部分是训练时随机初始化的，
    不存下来就无法从「lora_last.pt + 基座」精确重建模型。
    """
    if RANK not in {-1, 0} or not getattr(trainer, "_lora_injected", False):
        return
    model = trainer.ema.ema if getattr(trainer, "ema", None) is not None else trainer.model
    from ultralytics.utils.torch_utils import unwrap_model

    model = unwrap_model(model)
    sd = lora_state_dict(model)

    inner = getattr(model, "model", None)
    head_prefix = f"model.{len(inner) - 1}." if inner is not None and hasattr(inner, "__len__") else None
    non_base = getattr(trainer, "_non_base_modules", None)
    head_sd = {}
    for k, v in model.state_dict().items():
        if "lora_A" in k or "lora_B" in k:
            continue
        if non_base is not None:
            # 只留基座里没有/形状不匹配的模块（通常就是分类头最后那层卷积）
            if k.rsplit(".", 1)[0] not in non_base:
                continue
        elif not (head_prefix and k.startswith(head_prefix)):
            continue
        head_sd[k] = v.detach().cpu().clone()

    payload = {
        "lora": sd,
        "head": head_sd,
        "cfg": getattr(trainer, "lora_cfg", {}),
        "epoch": getattr(trainer, "epoch", -1),
        "base_model": getattr(trainer.args, "model", ""),
        "data": getattr(trainer.args, "data", ""),
        "names": getattr(trainer, "data", {}).get("names", {}),
        "nc": getattr(trainer, "data", {}).get("nc", None),
    }
    out = Path(trainer.wdir) / "lora_last.pt"
    torch.save(payload, out)
    LOGGER.info(f"[LoRA] 已保存 LoRA 权重 {out} ({len(sd)} 个 LoRA 张量, {len(head_sd)} 个非基座张量)")


def verify_callback(trainer):
    """on_train_start：确认真的只有 LoRA 参数在优化器里。"""
    if RANK not in {-1, 0}:
        return
    from ultralytics.utils.torch_utils import unwrap_model

    model = unwrap_model(trainer.model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    groups = [(len(g["params"]), g.get("param_group", "?")) for g in trainer.optimizer.param_groups]
    LOGGER.info(f"[LoRA] 可训练参数张量数: {len(trainable)}，优化器分组: {groups}")
    if not trainable:
        raise RuntimeError("没有任何可训练参数，LoRA 注入可能失败")
    bad = [n for n in trainable if "lora_A" not in n and "lora_B" not in n]
    if bad:
        LOGGER.warning(f"[LoRA] 有 {len(bad)} 个非 LoRA 参数仍在训练，例如 {bad[:3]}")
    total_opt = sum(n for n, _ in groups)
    LOGGER.info(f"[LoRA] 优化器内参数张量总数: {total_opt}")


def fold_artifacts(trainer, include_last: bool = False):
    """把刚存下的 ckpt 折叠成标准 YOLO 权重。

    Ultralytics 在 `final_eval()` 里会通过 `AutoBackend` 加载 best.pt 并调用 `model.fuse()`，
    而 `fuse_conv_and_bn()` 会给 `conv.weight` 赋 `nn.Parameter`——我们的包装层权重是 buffer，
    会触发 ``register_parameter: attribute 'weight' already exists``。
    折叠后 best.pt 就是原生卷积，既躲开这个坑，也能直接 `YOLO("best.pt")` 使用。
    `last.pt` 默认保留包装结构（NaN 恢复需要它和在线模型同结构），训练结束时再一并折叠。
    """
    if RANK not in {-1, 0} or not getattr(trainer, "_lora_injected", False):
        return
    wdir = Path(trainer.wdir)
    for p in sorted(wdir.glob("*.pt")):
        if p.name == "last.pt" and not include_last:
            continue
        if p.name.startswith("lora_") or "lora_last" in p.name:
            continue
        try:
            if fold_ckpt(p):
                LOGGER.info(f"[LoRA] 已把 {p.name} 折叠为标准 YOLO 权重")
        except Exception as e:  # 折叠失败不应该中断训练
            LOGGER.warning(f"[LoRA] 折叠 {p.name} 失败: {e}")


def save_and_fold(trainer):
    save_lora_callback(trainer)
    fold_artifacts(trainer, include_last=False)


def finish(trainer):
    save_lora_callback(trainer)
    fold_artifacts(trainer, include_last=True)


def main(argv=None):
    a = build_parser().parse_args(argv)

    cfg = dict(DEFAULT_LORA)
    cfg.update(
        {
            "enabled": True,
            "r": a.lora_r,
            "alpha": a.lora_alpha,
            "targets": a.lora_targets,
            "exclude": [s for s in str(a.lora_exclude).split(",") if s],
            "freeze_bn": not a.no_freeze_bn,
            "freeze_norm": not a.tune_norm,
            "freeze_others": True,
            "init_from": a.resume_lora,
        }
    )
    # 必须写到环境变量里：双卡 DDP 的子进程只继承 args，不继承 trainer 实例属性
    os.environ[LORA_ENV] = json.dumps(cfg)

    describe_plan(a)
    overrides = to_overrides(a)

    trainer = LoRATrainer(overrides=overrides)
    trainer.add_callback("on_train_start", verify_callback)
    trainer.add_callback("on_model_save", save_and_fold)
    trainer.add_callback("on_train_end", finish)
    trainer.train()

    if RANK in {-1, 0}:
        print("\n" + "=" * 66)
        print(f"训练结束。权重目录: {trainer.save_dir / 'weights'}")
        print(f"  best.pt     : 已折叠为标准 YOLO 权重（LoRA 烘进基座），可直接 YOLO(best.pt) 加载/预测/导出")
        print(f"  last.pt     : 同上，最后一个 epoch")
        print(f"  lora_last.pt: 纯 LoRA 权重 + 检测头基座（几十 KB ~ 几 MB），用于 --resume-lora 热启动")
        print(f"  评估: python eval_and_predict.py --weights {trainer.save_dir / 'weights' / 'best.pt'} "
              f"--data {a.data}")
        print("=" * 66)
    return trainer


if __name__ == "__main__":
    main()
