#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA（低秩适配）实现：用于 Ultralytics YOLO11 检测模型的参数高效微调。

设计要点（都是踩过坑之后的选择，改动前请先读这一段）：

1. 基座卷积权重存成 **buffer**（`base_weight`），不是 `nn.Parameter`。
   原因：Ultralytics 的 `BaseTrainer._setup_train()` 里有这段逻辑——

       for k, v in self.model.named_parameters():
           if any(x in k for x in freeze_layer_names):
               v.requires_grad = False
           elif not v.requires_grad and v.dtype.is_floating_point:
               v.requires_grad = True      # <== 会把我们冻结的权重重新打开！

   也就是说，靠 `requires_grad_(False)` 冻结基座是**无效**的，训练一开始就会被解冻。
   放进 buffer 之后，`named_parameters()` 里根本不会出现它们，Ultralytics 无从解冻，
   同时它们仍然会出现在 `state_dict()` 里（`persistent=True`），所以权重能正常保存，
   也会跟随 `.to(device) / .half()` 一起搬移。

2. 低秩分支写成 `W_delta = B @ A` 再 reshape 成卷积核，与主卷积核相加后一次 `F.conv2d`。
   不做 im2col/unfold：640×640 输入下 unfold 张量能到 900MB/图，必 OOM。

3. B 初始化为 0，A 用 kaiming_uniform。因此训练开始瞬间，模型输出与预训练模型**逐位一致**
   （`check_env.py --selftest` 会验证这一点），不会破坏 COCO 预训练权重。

4. `weight` / `bias` 暴露成 property，兼容 Ultralytics 的 `Conv.fuse_convs()` /
   `fuse_conv_and_bn()`，导出或 `model.fuse()` 时不会 AttributeError。

对外接口
--------
inject_lora(model, targets, r, alpha, exclude)   注入 LoRA，返回统计信息
lora_parameter_stats(model)                      统计可训练/总参数量
lora_state_dict(model)                           只导出 LoRA 权重
load_lora_state_dict(model, sd)                  加载 LoRA 权重（热启动）
merge_lora(model)                                把 LoRA 增量合并进基座权重（原地）
fold_to_plain(model)                             合并 + 把包装层换回标准 nn.Conv2d（导出标准 ckpt）
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """在 `nn.Conv2d` 上叠加低秩增量： ``y = conv(x, W0) + scale * conv(x, B@A)``。

    Args:
        conv: 被包装的原始卷积（权重会被迁移到 buffer，原模块不再被使用）。
        r: 低秩维度；``r=0`` 或 ``groups>1`` 时只做“冻结包装”，不挂 LoRA 分支
           （YOLO11 里有 7 个 depthwise/分组卷积，无法用整体核的低秩分解表示，
           但同样必须冻结，否则它们会变成全参数训练）。
        alpha: 缩放因子，实际缩放为 ``alpha / r``（默认 alpha=2r，即 scale=2）。
    """

    def __init__(self, conv: nn.Conv2d, r: int = 8, alpha: float = 16.0):
        super().__init__()
        if conv.padding_mode != "zeros":
            raise ValueError(f"LoRAConv2d 只支持 padding_mode='zeros'，收到 {conv.padding_mode}")

        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(self.r, 1)

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size: Tuple[int, int] = tuple(conv.kernel_size)
        self.stride: Tuple[int, int] = tuple(conv.stride)
        self.padding: Tuple[int, int] = tuple(conv.padding)
        self.dilation: Tuple[int, int] = tuple(conv.dilation)
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode

        # groups>1（depthwise 等）无法用整体卷积核的低秩分解表示 -> 只冻结，不加 LoRA
        self.use_lora = self.groups == 1 and self.r > 0

        # ---- 冻结的基座权重：buffer，而非 Parameter ----
        self.register_buffer("base_weight", conv.weight.detach().clone(), persistent=True)
        if conv.bias is not None:
            self.register_buffer("base_bias", conv.bias.detach().clone(), persistent=True)
        else:
            self.register_buffer("base_bias", None, persistent=True)

        # ---- 可训练的低秩分支 ----
        if self.use_lora:
            k = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            self.lora_A = nn.Linear(k, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, self.out_channels, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B.weight)

        self.merged = False

    # ------------------------------------------------------------------ 前向
    def delta_weight(self) -> Optional[torch.Tensor]:
        """返回低秩增量卷积核；不挂 LoRA 时返回 None。"""
        if not self.use_lora:
            return None
        w = self.lora_B.weight @ self.lora_A.weight  # [out, in*kh*kw]
        return w.view(self.out_channels, self.in_channels, *self.kernel_size) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.base_bias
        if self.merged or not self.use_lora:
            return F.conv2d(x, self.base_weight, bias, self.stride, self.padding, self.dilation, self.groups)
        w = self.base_weight + self.delta_weight().to(self.base_weight.dtype)
        return F.conv2d(x, w, bias, self.stride, self.padding, self.dilation, self.groups)

    # ------------------------------------- 兼容 Ultralytics fuse / 导出 / 统计
    @property
    def weight(self) -> torch.Tensor:
        """等效卷积核（含 LoRA 增量），供 `fuse_conv_and_bn` 等工具使用。"""
        if self.merged or not self.use_lora:
            return self.base_weight
        return self.base_weight + self.delta_weight().to(self.base_weight.dtype)

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        """`fuse_conv_and_bn()` 会执行 ``conv.weight = fused_w``。

        此时传入的权重**已经包含 LoRA 增量**（因为它读的是上面的 getter），所以这里要
        把 `merged` 置位，避免 forward 里再叠加一次 delta。
        """
        self.base_weight = value.detach() if isinstance(value, torch.Tensor) else value
        self.merged = True

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_bias

    @bias.setter
    def bias(self, value: Optional[torch.Tensor]) -> None:
        self.base_bias = value.detach() if isinstance(value, torch.Tensor) else value

    def extra_repr(self) -> str:  # pragma: no cover
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, groups={self.groups}, "
            f"lora={self.r if self.use_lora else 'off'}, alpha={self.alpha}, merged={self.merged}"
        )


# --------------------------------------------------------------------------- 注入
def _match(name: str, conv: nn.Conv2d, specs: List[str], head_prefix: Optional[str]) -> bool:
    for s in specs:
        s = s.strip().lower()
        if s == "all":
            return True
        if s == "1x1" and tuple(conv.kernel_size) == (1, 1):
            return True
        if s == "3x3" and tuple(conv.kernel_size) == (3, 3):
            return True
        if s == "head" and head_prefix and name.startswith(head_prefix):
            return True
        if s.startswith("name:") and s[5:] in name.lower():
            return True
    return False


@torch.no_grad()
def freeze_norm_affine(model: nn.Module) -> int:
    """把 BatchNorm 的 γ/β 从 `nn.Parameter` 改成 buffer，使其不参与训练。

    为什么需要：Ultralytics 只认 `freeze` 参数，凡是 `nn.Parameter` 默认都是可训练的。
    要想“只有 LoRA 分支在训练”，BN 的 affine 参数也必须变成 buffer。
    BN 的 forward 接受普通 Tensor，所以改成 buffer 不影响前向；state_dict 的 key 名也不变。

    关掉它（``freeze_norm=False``）相当于额外放开 BN 的 γ/β，通常是几百到几千个参数，
    在域差异较大的数据集上往往能多涨一点精度——但从“参数高效”的定义上就不再是纯 LoRA 了。

    Returns:
        被冻结的参数张量个数。
    """
    norms = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    if hasattr(nn, "SyncBatchNorm"):
        norms = norms + (nn.SyncBatchNorm,)
    n = 0
    for m in model.modules():
        if not isinstance(m, norms):
            continue
        for attr in ("weight", "bias"):
            p = getattr(m, attr, None)
            if isinstance(p, nn.Parameter):
                t = p.detach().clone()
                delattr(m, attr)
                m.register_buffer(attr, t, persistent=True)
                n += 1
    return n


def inject_lora(
    model: nn.Module,
    targets: str = "all",
    r: int = 8,
    alpha: float = 16.0,
    exclude: Iterable[str] = (".dfl",),
    freeze_norm: bool = True,
    freeze_others: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """把模型中的 `nn.Conv2d` 替换为 `LoRAConv2d`。

    Args:
        targets: ``all`` / ``1x1`` / ``3x3`` / ``head``，可用 ``+`` 组合，
                 例如 ``head+1x1``；也可用 ``name:xxx`` 按模块名子串匹配。
        exclude: 名字中包含这些子串的卷积不挂 LoRA（默认排除 ``.dfl``）。
        freeze_norm: 是否把 BN 的 γ/β 一并冻结（保持“只有 LoRA 在训练”）。
        freeze_others: 第二遍扫描，把**没被选中的卷积**也包成“只冻结、不加 LoRA”的层。
            必须保持 True，否则 ``targets=head`` 这类配置会让其余卷积保持全参数训练，
            “LoRA 微调”就名不副实了（实测 head 目标下会有 82% 参数可训练）。

    Returns:
        统计字典：``{"n_wrapped", "n_group_only", "n_frozen_only", "trainable", "total"}``
    """
    specs = [s for s in str(targets).replace(",", "+").split("+") if s.strip()]
    exclude = tuple(exclude or ())

    head_prefix = None
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "__len__") and len(inner) > 0:
        head_prefix = f"model.{len(inner) - 1}."

    n_wrapped = 0
    n_group_only = 0
    n_frozen_only = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Conv2d) or isinstance(child, LoRAConv2d):
                continue
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            matched = (not any(e in full for e in exclude)) and _match(full, child, specs, head_prefix)
            if matched:
                if child.groups != 1:
                    n_group_only += 1
                setattr(parent, child_name, LoRAConv2d(child, r=r, alpha=alpha))
                n_wrapped += 1
            elif freeze_others:
                setattr(parent, child_name, LoRAConv2d(child, r=0, alpha=alpha))
                n_frozen_only += 1

    n_norm = freeze_norm_affine(model) if freeze_norm else 0

    stats = {
        "n_wrapped": n_wrapped,
        "n_group_only": n_group_only,
        "n_frozen_only": n_frozen_only,
        "n_norm_frozen": n_norm,
        "r": r,
        "alpha": alpha,
        "targets": targets,
    }
    stats.update(lora_parameter_stats(model))
    if verbose:
        ratio = stats["trainable"] / max(stats["total"], 1) * 100
        print(
            f"[LoRA] 挂 LoRA 的卷积 {n_wrapped} 个 (targets={targets}, r={r}, alpha={alpha})，"
            f"其中 groups>1 只能冻结 {n_group_only} 个；"
            f"其余冻结卷积 {n_frozen_only} 个，冻结 BN 仿射 {n_norm} 个；"
            f"可训练参数 {stats['trainable']:,} / 总参数 {stats['total']:,} —— 占 {ratio:.2f} percent"
        )
    return stats


def lora_parameter_stats(model: nn.Module) -> Dict[str, int]:
    """返回可训练参数量 / 总参数量（含 buffer）。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters()) + sum(
        b.numel() for b in model.buffers() if isinstance(b, torch.Tensor)
    )
    return {"trainable": int(trainable), "total": int(total)}


def lora_param_names(model: nn.Module) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


# --------------------------------------------------------------------- 权重读写
def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """只提取 LoRA 分支权重（体积很小，几十 KB ~ 几 MB）。"""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor], strict: bool = True) -> None:
    """把 LoRA 权重加载进已注入 LoRA 的模型（用于断点热启动）。"""
    own = model.state_dict()
    missing = [k for k in state if k not in own]
    if missing:
        msg = f"LoRA 权重与当前模型不匹配，缺少 {len(missing)} 个 key，例如 {missing[:3]}"
        if strict:
            raise KeyError(msg)
        print(f"[LoRA][warn] {msg}")
    res = model.load_state_dict(state, strict=False)
    unexpected = [k for k in res.unexpected_keys if "lora" in k]
    if unexpected:
        raise KeyError(f"加载 LoRA 权重出现未知 key：{unexpected[:3]}")


# ------------------------------------------------------------------------ 合并
@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """把 LoRA 增量原地合并进基座权重，合并后前向只剩一次卷积（推理零额外开销）。"""
    n = 0
    for m in model.modules():
        if isinstance(m, LoRAConv2d) and not m.merged:
            delta = m.delta_weight()
            if delta is not None:
                m.base_weight.add_(delta.to(m.base_weight.dtype))
            m.merged = True
            n += 1
    if n:
        print(f"[LoRA] 已合并 {n} 个 LoRA 分支到基座权重")
    return model


@torch.no_grad()
def fold_to_plain(model: nn.Module) -> nn.Module:
    """合并 LoRA 并把 `LoRAConv2d` 还原成标准 `nn.Conv2d`。

    用于导出**标准 YOLO 权重**：导出的 ckpt 里 state_dict 的 key 与原生 YOLO11 完全一致
    （`....weight` / `....bias`），可以脱离本仓库直接用 `YOLO("xxx.pt")` 加载、验证、导出 ONNX。
    """
    merge_lora(model)
    n = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, LoRAConv2d):
                continue
            conv = nn.Conv2d(
                child.in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.base_bias is not None,
                padding_mode=child.padding_mode,
            )
            conv.weight.data.copy_(child.base_weight)
            if child.base_bias is not None:
                conv.bias.data.copy_(child.base_bias)
            conv.to(device=child.base_weight.device, dtype=child.base_weight.dtype)
            conv.weight.requires_grad_(True)
            setattr(parent, child_name, conv)
            n += 1
    print(f"[LoRA] 已还原 {n} 个标准卷积层")
    return model


def fold_ckpt(path, out_path=None, half: bool = True) -> bool:
    """把 Ultralytics 保存的 ckpt 折叠成标准 YOLO 权重（LoRA 合并进基座、包装层还原）。

    为什么要在训练过程中做这件事：`AutoBackend` 加载 ckpt 时会调用 `model.fuse()`，
    而 `fuse_conv_and_bn()` 执行的是 ``conv.weight = nn.Parameter(...)`` —— 我们的包装层
    权重是 buffer（并且 `weight` 是属性），会撞上 `register_parameter` 的
    ``attribute 'weight' already exists``。把产物直接折叠成原生卷积，既躲开这个坑，
    也让 `best.pt` 能脱离本仓库直接 `YOLO(...)` 使用。

    Args:
        path: 待折叠的 ckpt（原地覆盖或写到 out_path）。
        half: 是否以 fp16 保存（与 Ultralytics 的产物一致，V100 推理直接用）。

    Returns:
        是否发生了折叠（没有 LoRA 包装层时返回 False）。
    """
    src = Path(path)
    dst = Path(out_path) if out_path else src
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return False
    model = ckpt.get("ema") or ckpt.get("model")
    if model is None or not any(isinstance(m, LoRAConv2d) for m in model.modules()):
        return False

    model.float()
    fold_to_plain(model)
    if hasattr(model, "criterion"):
        try:
            model.criterion = None
        except Exception:
            pass
    if half:
        model.half()
    if ckpt.get("model") is not None:
        ckpt["model"] = model
    ckpt["ema"] = model
    if "optimizer" in ckpt:
        ckpt.pop("optimizer", None)  # 参数结构已变，optimizer 状态不再可用
    if "scaler" in ckpt:
        ckpt.pop("scaler", None)
    torch.save(ckpt, dst)
    return True
