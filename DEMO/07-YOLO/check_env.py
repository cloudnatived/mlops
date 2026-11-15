#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""上线前自检：环境、GPU、数据集、LoRA 实现正确性。

    python check_env.py --data /data/tt100k_yolo/data.yaml --selftest

`--selftest` 做三件关键验证（不需要数据集、不需要 GPU）：
1. 注入 LoRA 后，模型输出与注入前**逐位一致**（B 初始为 0 的数学必然），说明没有破坏预训练权重；
2. `merge_lora()` 之后输出仍一致，说明增量合并公式正确；
3. `fold_to_plain()` 之后 state_dict 的 key 与原版 YOLO11 完全相同，说明导出的 ckpt 可脱离本仓库使用。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from lora import fold_to_plain, inject_lora, lora_parameter_stats, merge_lora  # noqa: E402


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_env():
    hr("1. 环境")
    print(f"python       : {sys.version.split()[0]}")
    try:
        import ultralytics
        print(f"ultralytics  : {ultralytics.__version__}")
    except Exception as e:
        print(f"ultralytics  : 未安装 ({e})")
    print(f"torch        : {torch.__version__}  (cuda {torch.version.cuda})")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device count : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU{i}: {p.name}  {p.total_memory / 2**30:.1f} GB  "
                  f"capability {p.major}.{p.minor}  multiprocessor {p.multi_processor_count}")
        arch = torch.cuda.get_arch_list()
        cc = torch.cuda.get_device_capability(0)
        print(f"arch list    : {arch}")
        if f"sm_{cc[0]}{cc[1]}" not in arch:
            print(f"  [!] 当前 torch 未编译 sm_{cc[0]}{cc[1]}，V100 上会 "
                  f"'no kernel image is available'。请装 cu118/cu121 的官方 wheel。")
        elif cc == (7, 0):
            print("  [ok] 含 sm_70，V100 可用（fp16 张量核需 amp=True）")
    else:
        print("  [!] 没有可见的 GPU，只能做 CPU 自检")

    import psutil  # ultralytics 依赖里带了 psutil

    vm = psutil.virtual_memory()
    print(f"RAM          : {vm.total / 2**30:.1f} GB (可用 {vm.available / 2**30:.1f} GB)")
    print(f"cpu 逻辑核   : {psutil.cpu_count(logical=True)}")
    if vm.total / 2**30 < 20:
        print("  [!] 内存 <20GB：务必 cache=False、workers<=3，否则 dataloader 会把内存打满")
    print(f"disk (cwd)   : {shutil.disk_usage('.').free / 2**30:.1f} GB free")


def check_dataset(yaml_path: str):
    hr("2. 数据集")
    import yaml

    path = Path(yaml_path).expanduser().resolve()
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(d.get("path") or path.parent)
    nc = d.get("nc") or len(d.get("names", {}))
    print(f"yaml     : {path}")
    print(f"root     : {root}")
    print(f"nc       : {nc}")
    names = d.get("names", {})
    if isinstance(names, dict):
        print(f"names    : {len(names)} 类, 前 8 个: {list(names.values())[:8]}")
        if len(names) != nc:
            print(f"  [!] nc={nc} 与 names 数量 {len(names)} 不一致")
    ok = True
    for split in ("train", "val", "test"):
        if split not in d:
            continue
        img_dir = root / d[split]
        if not img_dir.exists():
            print(f"{split:<9}: [缺失] {img_dir}")
            if split == "train":
                ok = False
            continue
        imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        lab_dir = root / "labels" / split
        labs = list(lab_dir.rglob("*.txt")) if lab_dir.exists() else []
        empty = 0
        for p in labs[:200000]:
            if p.stat().st_size == 0:
                empty += 1
        print(f"{split:<9}: {len(imgs):>6} 图 / {len(labs):>6} 标注文件 / {empty} 个空标注")
        if len(imgs) == 0:
            print(f"  [!] {split} 没有图片")
            ok = False
        if len(imgs) != len(labs):
            print(f"  [!] 图与标注数量不一致，可能是软链接失效或漏写 label")

    # 抽检标注数值
    lab_dir = root / "labels" / "train"
    samples = list(lab_dir.rglob("*.txt"))[:200] if lab_dir.exists() else []
    bad = []
    boxes = 0
    for p in samples:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            boxes += 1
            if len(parts) != 5:
                bad.append((p.name, "字段数不为5"))
                continue
            c = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
            if c < 0 or c >= nc:
                bad.append((p.name, f"cls={c} 越界"))
            if any(v < 0 or v > 1 for v in vals):
                bad.append((p.name, f"归一化越界 {vals}"))
            if vals[2] <= 0 or vals[3] <= 0:
                bad.append((p.name, f"宽高<=0 {vals}"))
    print(f"抽检     : {len(samples)} 个标注文件, {boxes} 个框, 问题 {len(bad)}")
    for b in bad[:5]:
        print(f"  [!] {b}")
    return ok


def _flat_output(model, x):
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, (tuple, list)):
        return [t.clone() for t in out if isinstance(t, torch.Tensor)]
    return [out.clone()]


def selftest(model_name: str = "yolo11n.pt", imgsz: int = 640):
    hr("3. LoRA 实现自检")
    from ultralytics import YOLO

    yolo = YOLO(model_name)
    model = yolo.model.float().eval()
    x = torch.randn(1, 3, imgsz, imgsz)

    base_out = _flat_output(model, x)
    stats = inject_lora(model, targets="all", r=8, alpha=16.0, verbose=True)

    print(f"LoRA 参数量: {stats['trainable']:,} / {stats['total']:,} "
          f"({stats['trainable'] / max(stats['total'], 1) * 100:.2f}%)")
    bad_params = [n for n, p in model.named_parameters() if "base_weight" in n or "base_bias" in n]
    assert not bad_params, f"基座权重仍是 Parameter（会被 Ultralytics 解冻）: {bad_params[:3]}"
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    non_lora = [n for n in trainable if "lora_A" not in n and "lora_B" not in n]
    print(f"可训练张量 {len(trainable)} 个，其中非 LoRA {len(non_lora)} 个"
          + (f"（例如 {non_lora[:3]}）" if non_lora else ""))
    assert not non_lora, "仍有非 LoRA 参数可训练，freeze_norm 逻辑失效"

    out_lora = _flat_output(model, x)
    d1 = max((a - b).abs().max().item() for a, b in zip(base_out, out_lora))
    print(f"[1] 注入 LoRA 后输出差异 (应为 ~0): {d1:.3e}")
    assert d1 < 1e-4, "B=0 初始化被破坏，注入后输出不应变化"

    # 随机化 B，验证增量确实生效，再合并
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "lora_B"):
                m.lora_B.weight.normal_(0, 0.02)
    out_rand = _flat_output(model, x)
    merge_lora(model)
    out_merged = _flat_output(model, x)
    d2 = max((a - b).abs().max().item() for a, b in zip(out_rand, out_merged))
    print(f"[2] merge 前后输出差异 (应为 ~0): {d2:.3e}")
    assert d2 < 1e-4, "merge 公式与 forward 不一致"

    fold_to_plain(model)
    out_plain = _flat_output(model, x)
    d3 = max((a - b).abs().max().item() for a, b in zip(out_merged, out_plain))
    print(f"[3] fold 成标准卷积后输出差异 (应为 ~0): {d3:.3e}")
    keys = list(model.state_dict().keys())
    bad = [k for k in keys if "lora_" in k or "base_weight" in k]
    print(f"[4] 标准 key 检查: {len(keys)} 个 key, 异常 {len(bad)}")
    assert not bad, bad[:3]

    stats2 = lora_parameter_stats(model)
    print(f"最终模型参数: {stats2['total']:,}")
    print("\n[PASS] LoRA 注入 / 前向 / 合并 / 导出全部通过")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="环境与数据集自检", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=None, help="data.yaml")
    p.add_argument("--selftest", action="store_true", help="LoRA 数学正确性自检（需联网下 yolo11n.pt）")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="env_report.json")
    a = p.parse_args(argv)

    check_env()
    ok = True
    if a.data:
        ok = check_dataset(a.data) and ok
    if a.selftest:
        ok = selftest(a.model, a.imgsz) and ok

    report = {"ok": bool(ok), "torch": torch.__version__, "cuda": torch.cuda.is_available()}
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'全部通过' if ok else '存在问题，请先修复上面标 [!] 的项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
