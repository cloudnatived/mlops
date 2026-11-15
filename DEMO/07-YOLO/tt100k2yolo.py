#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TT100K（清华-腾讯 100K 交通标志数据集） → YOLO 检测格式转换。

官方数据解压后的典型布局（data.zip 解开后多一层 data/）::

    TT100K/
    └── data/
        ├── annotations.json      # {"imgs": {id: {...}}, "types": [221 类]}
        ├── train/   ~6105 张带标注
        ├── test/    ~3071 张带标注
        ├── other/   ~7641 张带标注（官方划分里作为验证集）
        └── marks/   标志标准图（用于增强，本题不用）

每条标注形如::

    {"path": "train/12345.jpg",
     "objects": [{"category": "p5",
                 "bbox": {"xmin": 1097.11, "ymin": 968.116,
                          "xmax": 1116.309, "ymax": 987.6815}}, ...]}

本脚本做四件事：
1. 读 annotations.json，按 `path` 前缀（train/ test/ other/）落到 train/val(test)/ 划分；
2. 按训练集实例数筛出类别（默认 >=100 实例，即论文的 45 类基准子集；`--all-classes` 保留 221 类）；
3. 把 bbox 归一化成 YOLO 的 `cls cx cy w h`，做裁剪、退化框过滤、小目标过滤；
4. 生成 `images/ labels/ data.yaml`，默认用**软链接**引图（不占额外 18GB 磁盘）。

示例
----
    # 45 类基准子集 + other 作 val + 软链接
    python tt100k2yolo.py --src /data/TT100K/data --out /data/tt100k_yolo

    # 先做个 300 张的小数据集验证全流程（8 核 16G 机器强烈建议先跑这个）
    python tt100k2yolo.py --src /data/TT100K/data --out /data/tt100k_mini --limit 300 --limit-val 100

    # 保留全部 221 类，并把图长边预缩到 1024（省 CPU 解码时间，代价是小目标略损）
    python tt100k2yolo.py --src /data/TT100K/data --out /data/tt100k_yolo221 \
        --all-classes --resize-long-side 1024
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = "train", "val", "test"


# ---------------------------------------------------------------- 标注读取
def find_annotation_file(src: Path) -> Path:
    """在 src（及其 data/ 子目录）里找 annotations.json / annotations_all.json。"""
    candidates: List[Path] = []
    for root in (src, src / "data"):
        for name in ("annotations.json", "annotations_all.json", "annotation.json"):
            candidates.append(root / name)
    for c in candidates:
        if c.is_file():
            return c
    for pat in ("*.json", "data/*.json"):
        for p in sorted(src.glob(pat)):
            if p.is_file() and p.name.lower().startswith("annotation"):
                return p
    raise FileNotFoundError(
        f"在 {src} 下没找到 annotations.json。请让 --src 指向解压后包含 annotations.json 的 data 目录。"
    )


def iter_entries(data) -> Iterable[Tuple[str, dict]]:
    """统一 `imgs` 为 dict 或 list 两种写法。"""
    imgs = data.get("imgs") if isinstance(data, dict) else data
    if imgs is None:
        raise ValueError("annotations.json 里没有 'imgs' 字段")
    if isinstance(imgs, dict):
        for k, v in imgs.items():
            if isinstance(v, dict):
                yield str(k), v
    else:
        for v in imgs:
            if isinstance(v, dict):
                yield str(v.get("id", v.get("path", ""))), v


def build_image_index(src: Path) -> Dict[str, Path]:
    """文件名 → 绝对路径，兼容 path 字段缺失/与实际目录不一致的情况。"""
    idx: Dict[str, Path] = {}
    roots = [src, src / "data"]
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(IMG_EXTS):
                    idx.setdefault(fn, Path(dirpath) / fn)
    return idx


def split_from_rel(rel: str) -> Optional[str]:
    parts = [p.lower() for p in Path(rel).parts]
    for p in parts[:2]:
        if p in ("train", "training"):
            return SPLIT_TRAIN
        if p in ("other", "val", "valid", "validation"):
            return "other"
        if p in ("test", "testing"):
            return SPLIT_TEST
    return None


def collect_records(src: Path, ann_file: Path) -> Tuple[List[dict], List[str], int, int]:
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    types = list(data.get("types") or []) if isinstance(data, dict) else []
    index = build_image_index(src)

    records: List[dict] = []
    miss_img = miss_size = 0
    for key, ent in tqdm(list(iter_entries(data)), desc="解析标注"):
        rel = ent.get("path") or ent.get("file_name") or ""
        img_path: Optional[Path] = None
        if rel:
            for base in (src, src / "data", ann_file.parent):
                cand = base / rel
                if cand.is_file():
                    img_path = cand
                    break
        if img_path is None:
            for ext in IMG_EXTS:
                cand = index.get(f"{key}{ext}")
                if cand is not None:
                    img_path = cand
                    break
        if img_path is None:
            miss_img += 1
            continue
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            miss_size += 1
            continue

        boxes = []
        for obj in ent.get("objects") or []:
            cat = str(obj.get("category") or "").strip()
            bb = obj.get("bbox") or {}
            if not cat or not bb:
                continue
            x1, y1 = float(bb.get("xmin", 0.0)), float(bb.get("ymin", 0.0))
            x2, y2 = float(bb.get("xmax", 0.0)), float(bb.get("ymax", 0.0))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            boxes.append((cat, x1, y1, x2, y2))

        records.append(
            {
                "path": img_path,
                "rel_path": rel or img_path.name,
                "official_split": split_from_rel(rel) or split_from_rel(str(img_path)) or SPLIT_TRAIN,
                "width": width,
                "height": height,
                "boxes": boxes,
            }
        )
    return records, types, miss_img, miss_size


# ---------------------------------------------------------------- 划分与采样
def assign_splits(records: List[dict], args) -> None:
    if args.split_mode == "official":
        for r in records:
            sp = r["official_split"]
            if sp == "other":
                sp = SPLIT_VAL if args.val_from == "other" else SPLIT_TEST
            r["split"] = sp
        if args.val_from == "test":
            # test 被当作 val 用掉后，test 划分留空
            pass
    else:  # random：忽略官方目录，按比例重切（当心与官方 test 混用）
        rng = random.Random(args.seed)
        rng.shuffle(records)
        n = len(records)
        n_test = int(n * args.test_ratio)
        n_val = int(n * args.val_ratio)
        for i, r in enumerate(records):
            if i < n_test:
                r["split"] = SPLIT_TEST
            elif i < n_test + n_val:
                r["split"] = SPLIT_VAL
            else:
                r["split"] = SPLIT_TRAIN

    # 每个 split 限流（做快速验证用）
    rng = random.Random(args.seed)
    limits = {SPLIT_TRAIN: args.limit, SPLIT_VAL: args.limit_val, SPLIT_TEST: args.limit_test}
    for sp, lim in limits.items():
        if lim is None or lim <= 0:
            continue
        group = [r for r in records if r["split"] == sp]
        if len(group) > lim:
            keep = set(id(r) for r in rng.sample(group, lim))
            for r in group:
                if id(r) not in keep:
                    r["split"] = "drop"


def resolve_classes(records: List[dict], types: List[str], args) -> Tuple[List[str], Counter]:
    counts: Counter = Counter()
    for r in records:
        if r.get("split") == SPLIT_TRAIN:
            for b in r["boxes"]:
                counts[b[0]] += 1
    if not counts:  # 没有 train 划分（异常情况）时退化为全量统计
        for r in records:
            for b in r["boxes"]:
                counts[b[0]] += 1

    if args.classes_file:
        classes = [
            line.strip()
            for line in Path(args.classes_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.all_classes:
        base = types or sorted(counts)
        classes = sorted(base)
    else:
        # 论文定义：训练集中实例数 >= 100 的类别（45 类）。这里按实际数据复现，不硬编码类名。
        classes = sorted([c for c, n in counts.items() if n >= args.min_instances])

    excluded = set(args.exclude_classes or [])
    classes = [c for c in classes if c not in excluded]
    if not classes:
        raise SystemExit("筛选后类别为空，请降低 --min-instances 或检查 --src")
    return classes, counts


# ---------------------------------------------------------------- 写出
def link_or_copy(src: Path, dst: Path, mode: str, long_side: Optional[int], quality: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(str(src.resolve()), str(dst))
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        try:
            os.link(str(src), str(dst))
        except OSError:
            shutil.copy2(str(src), str(dst))
    elif mode == "copy":
        shutil.copy2(str(src), str(dst))
    elif mode == "resize":
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            if long_side and max(w, h) > long_side:
                scale = long_side / max(w, h)
                im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BICUBIC)
            im.save(dst, "JPEG", quality=quality, subsampling=0)
    else:
        raise ValueError(f"未知 --link-mode {mode}")


def write_dataset(records: List[dict], classes: List[str], counts: Counter, ann_types: List[str], args) -> dict:
    out = Path(args.out).expanduser().resolve()
    if args.clean and out.exists():
        print(f"清理输出目录 {out}")
        shutil.rmtree(out)
    for sp in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    stats = defaultdict(lambda: {"images": 0, "boxes": 0, "empty_images": 0})
    dropped_cls = Counter()   # 类别不在保留列表
    dropped_small = 0         # 面积/边长过小
    dropped_bad = 0           # 裁剪后宽高为 0

    for r in tqdm([x for x in records if x.get("split") in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)], desc="写数据集"):
        sp = r["split"]
        src: Path = r["path"]
        stem = src.stem
        # 同名文件跨划分不会冲突（各 split 独立目录）
        dst_img = out / "images" / sp / f"{stem}{src.suffix.lower()}"
        link_or_copy(src, dst_img, args.link_mode, args.resize_long_side, args.jpeg_quality)

        W, H = float(r["width"]), float(r["height"])
        lines: List[str] = []
        for cat, x1, y1, x2, y2 in r["boxes"]:
            idx = cls_to_idx.get(cat)
            if idx is None:
                dropped_cls[cat] += 1
                continue
            x1 = max(0.0, min(W, x1))
            y1 = max(0.0, min(H, y1))
            x2 = max(0.0, min(W, x2))
            y2 = max(0.0, min(H, y2))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                dropped_bad += 1
                continue
            if bw < args.min_width or bh < args.min_height or bw * bh < args.min_area:
                dropped_small += 1
                continue
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            nw = bw / W
            nh = bh / H
            lines.append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = out / "labels" / sp / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        stats[sp]["images"] += 1
        stats[sp]["boxes"] += len(lines)
        if not lines:
            stats[sp]["empty_images"] += 1

    # ---- data.yaml ----
    present_splits = [sp for sp in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST) if stats[sp]["images"] > 0]
    if SPLIT_TRAIN not in present_splits:
        raise SystemExit("train 划分为空，请检查 --src / --split-mode")
    data_yaml = {"path": str(out), "train": "images/train"}
    if SPLIT_VAL in present_splits:
        data_yaml["val"] = "images/val"
    if SPLIT_TEST in present_splits:
        data_yaml["test"] = "images/test"
    if SPLIT_VAL not in present_splits:  # 没有 val 时用 train 兜底，避免 trainer 起不来
        data_yaml["val"] = "images/train"
    data_yaml["nc"] = len(classes)
    data_yaml["names"] = {i: c for i, c in enumerate(classes)}
    yaml_path = out / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    summary = {
        "annotation_types_total": len(ann_types),
        "classes_kept": len(classes),
        "classes": classes,
        "min_instances_threshold": None if (args.all_classes or args.classes_file) else args.min_instances,
        "splits": {sp: dict(v) for sp, v in stats.items()},
        "dropped": {
            "small_boxes": dropped_small,
            "degenerate_boxes": dropped_bad,
            "class_not_kept": sum(dropped_cls.values()),
            "top_missing_classes": dropped_cls.most_common(10),
        },
        "filters": {
            "min_area": args.min_area,
            "min_width": args.min_width,
            "min_height": args.min_height,
            "link_mode": args.link_mode,
            "resize_long_side": args.resize_long_side,
        },
        "class_instance_counts_train": {c: int(counts.get(c, 0)) for c in classes},
    }
    (out / "dataset_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 打印 ----
    print("\n" + "=" * 66)
    print(f"输出目录 : {out}")
    print(f"data.yaml: {yaml_path}")
    print(f"类别数   : {len(classes)}  (原始 types: {len(ann_types)})")
    for sp in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        if stats[sp]["images"]:
            print(
                f"  {sp:<6}: {stats[sp]['images']:>6} 张, {stats[sp]['boxes']:>7} 个框, "
                f"空标注图 {stats[sp]['empty_images']}"
            )
    print(f"过滤掉的小框: {dropped_small}，退化框: {dropped_bad}，未保留类别框: {sum(dropped_cls.values())}")
    top = counts.most_common(10)
    print("训练集 Top10 类别:", ", ".join(f"{c}={n}" for c, n in top))
    print("=" * 66)
    print("\n下一步：")
    print(f"  python check_env.py --data {yaml_path}")
    print(f"  python train_lora.py --data {yaml_path}")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TT100K → YOLO 格式转换", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", required=True, help="解压后包含 annotations.json 的目录（TT100K/data）")
    p.add_argument("--out", required=True, help="输出 YOLO 数据集目录")
    p.add_argument("--link-mode", choices=["symlink", "hardlink", "copy", "resize"], default="symlink",
                   help="引图方式；symlink 不占空间，resize 会另存缩小后的图")
    p.add_argument("--resize-long-side", type=int, default=1024,
                   help="--link-mode resize 时的长边像素；2048 原图训练 640 时预缩可显著省 CPU")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--min-instances", type=int, default=100,
                   help="训练集实例数阈值；默认 100 复现论文 45 类基准子集")
    p.add_argument("--all-classes", action="store_true", help="保留全部 221 类（含极稀疏类）")
    p.add_argument("--classes-file", default=None, help="自定义类别清单（每行一个类名）")
    p.add_argument("--exclude-classes", nargs="*", default=["wo"], help="排除的类别名，默认排除 wo")
    p.add_argument("--min-area", type=float, default=100.0, help="过滤最小框面积（原图像素）")
    p.add_argument("--min-width", type=float, default=6.0, help="过滤最小框宽（原图像素）")
    p.add_argument("--min-height", type=float, default=6.0, help="过滤最小框高（原图像素）")
    p.add_argument("--split-mode", choices=["official", "random"], default="official")
    p.add_argument("--val-from", choices=["other", "test"], default="other",
                   help="official 模式下哪个目录当 val；other 为官方惯例")
    p.add_argument("--val-ratio", type=float, default=0.1, help="random 模式 val 比例")
    p.add_argument("--test-ratio", type=float, default=0.1, help="random 模式 test 比例")
    p.add_argument("--limit", type=int, default=0, help="train 最多取多少张（0=全部，做冒烟测试用）")
    p.add_argument("--limit-val", type=int, default=0, help="val 最多取多少张")
    p.add_argument("--limit-test", type=int, default=0, help="test 最多取多少张")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean", action="store_true", help="覆盖前清空 --out")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.link_mode == "resize" and not args.resize_long_side:
        raise SystemExit("--link-mode resize 需要 --resize-long-side")
    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"--src 目录不存在：{src}")

    ann_file = find_annotation_file(src)
    print(f"标注文件: {ann_file}")
    records, types, miss_img, miss_size = collect_records(src, ann_file)
    print(f"解析到 {len(records)} 张图（缺图 {miss_img}，读尺寸失败 {miss_size}）")

    assign_splits(records, args)
    records = [r for r in records if r.get("split") != "drop"]
    classes, counts = resolve_classes(records, types, args)

    summary = write_dataset(records, classes, counts, types, args)
    return summary


if __name__ == "__main__":
    main()
