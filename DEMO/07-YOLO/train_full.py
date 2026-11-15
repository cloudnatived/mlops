#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO11n 在 TT100K 上的**全量微调**（fine-tune 全部参数）—— 作为 LoRA 的对照基线。

这台机器的约束（8 vCPU / 16 GB RAM / 2×V100-16G）决定了几个默认值：

* `workers=3`（每 rank）—— 双卡时合计 6 个 dataloader 进程，8 核刚好不打架；
  每个 worker 解码 2048×2048 大图时约 200~400 MB 内存，16G 内存下别超过 4。
* `cache=False` —— 16G 内存装不下训练集（`cache=ram` 必 OOM）。磁盘够的话可用 `--cache disk`，
  用 ~7 GB 的 npy 缓存换掉重复的 JPEG 解码，第一次构建比较慢。
* `imgsz=640` 起步；TT100K 全是小目标，1024/1280 精度明显更好，按 preset 切换。
* `fliplr=0` —— 交通标志镜像后数字/文字反写，是错误标注。

常用命令::

    # 双卡 640（默认）
    python train_full.py --data /data/tt100k_yolo/data.yaml --device 0,1

    # 双卡 1024，batch 相应减小
    python train_full.py --data /data/tt100k_yolo/data.yaml --device 0,1 --preset 1024

    # 单卡、只用 10% 数据、5 epoch 跑通流程
    python train_full.py --data .../data.yaml --device 0 --fraction 0.1 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO  # noqa: E402

try:
    from ultralytics.utils import DEFAULT_CFG_DICT
except ImportError:  # pragma: no cover
    from ultralytics.cfg import DEFAULT_CFG_DICT

# preset: imgsz / 总 batch / 每 rank workers
PRESETS = {
    "640": {"imgsz": 640, "batch": 64, "workers": 3},
    "1024": {"imgsz": 1024, "batch": 16, "workers": 2},
    "1280": {"imgsz": 1280, "batch": 8, "workers": 2},
}


def build_parser():
    p = argparse.ArgumentParser(
        description="YOLO11n 全量微调 TT100K", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--data", required=True)
    p.add_argument("--preset", choices=list(PRESETS), default="640",
                   help="640=快，1024/1280=小目标更准但更吃显存")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, help="总 batch（跨卡），不给则用 preset")
    p.add_argument("--nbs", type=int, default=32, help="nominal batch，按每卡 batch 给")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="0,1")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--cache", choices=["off", "ram", "disk"], default="off")
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--project", default="runs/tt100k")
    p.add_argument("--name", default="full_ft")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--save-period", type=int, default=5)
    p.add_argument("--optimizer", default="SGD", choices=["SGD", "AdamW", "Adam", "auto"])
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--cos-lr", choices=["on", "off"], default="on")
    p.add_argument("--amp", choices=["on", "off"], default="on")
    p.add_argument("--freeze", type=int, default=0, help="冻结前 N 层（0=全部训练）")
    p.add_argument("--mosaic", type=float, default=0.5)
    p.add_argument("--close-mosaic", type=int, default=15)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--fliplr", type=float, default=0.0, help="交通标志不要镜像")
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.7)
    p.add_argument("--hsv-v", type=float, default=0.4)
    p.add_argument("--val", choices=["on", "off"], default="on")
    p.add_argument("--plots", choices=["on", "off"], default="on")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    preset = PRESETS[a.preset]
    imgsz = a.imgsz or preset["imgsz"]
    batch = a.batch or preset["batch"]
    workers = a.workers if a.workers is not None else preset["workers"]

    world = len([d for d in str(a.device).split(",") if d.strip()]) if a.device else 1
    per_gpu = max(batch // max(world, 1), 1)
    accum = max(round(a.nbs / per_gpu), 1)

    ov = {
        "data": a.data,
        "epochs": a.epochs,
        "imgsz": imgsz,
        "batch": batch,
        "nbs": a.nbs,
        "device": a.device,
        "workers": workers,
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
        "freeze": a.freeze,
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
        "rect": False,
        "val": a.val == "on",
        "plots": a.plots == "on",
        "pretrained": True,
        "verbose": True,
    }
    ov = {k: v for k, v in ov.items() if k in DEFAULT_CFG_DICT}

    print("=" * 66)
    print(f"全量微调 {a.model}   preset={a.preset}")
    print(f"imgsz={imgsz} 总batch={batch} → 每卡 {per_gpu}，累加 {accum} → 等效 {per_gpu * world * accum}")
    print(f"device={a.device} workers/rank={workers} (合计 {workers * world}) cache={a.cache}")
    print(f"{a.optimizer} lr0={a.lr0} epochs={a.epochs} AMP={a.amp}")
    print("=" * 66)

    model = YOLO(a.model)
    results = model.train(**ov)

    save_dir = Path(model.trainer.save_dir)
    print("\n" + "=" * 66)
    print(f"训练结束: {save_dir}")
    print(f"  best.pt: {save_dir / 'weights' / 'best.pt'}")
    try:
        metrics = getattr(model.trainer, "metrics", {}) or {}
        keys = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"]
        print("  最终指标: " + json.dumps({k: round(float(metrics[k]), 4) for k in keys if k in metrics}))
    except Exception:
        pass
    print("=" * 66)
    return results


if __name__ == "__main__":
    main()
