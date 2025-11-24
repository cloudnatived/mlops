#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import gradio as gr
import pandas as pd

# ---------------- 配置Hugging Face镜像 ----------------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # 不启用离线模式

# ---------------- 设备 & 模型 ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device} | GPU数量: {torch.cuda.device_count()}")

#model_name = "microsoft/trocr-base-printed"
model_name = "/Data/DEMO/MODEL/microsoft/trocr-base-printed"

try:
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
except Exception as e:
    raise RuntimeError(f"加载模型失败: {str(e)}")

if torch.cuda.device_count() > 1:
    print(f"启用 {torch.cuda.device_count()} 张 GPU 并行")
    # 确保我们可以访问原始模型的generate方法
    #model.generate = model.module.generate    
    model = torch.nn.DataParallel(model)
model.to(device).eval()

# ---------------- 批处理函数 ----------------
BATCH_SIZE = 8  # 可根据显存调整
WORKERS = 4      # 数据加载线程数

def preprocess_batch(imgs: List[np.ndarray]) -> torch.Tensor:
    """把多张 numpy 图转成模型需要的 tensor"""
    try:
        pil_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB") for img in imgs]
        inputs = processor(images=pil_imgs, return_tensors="pt")
        return inputs["pixel_values"].to(device)
    except Exception as e:
        raise RuntimeError(f"预处理失败: {str(e)}")

def batch_infer(images: List[np.ndarray], progress=gr.Progress()) -> List[str]:
    """批量推理函数，返回每张图的OCR识别结果"""
    if not images:
        return []

    progress(0, desc="预处理中...")
    n = len(images)
    texts = []

    try:
        # 分批推理
        for start in progress.tqdm(range(0, n, BATCH_SIZE), desc="OCR推理"):
            end = min(start + BATCH_SIZE, n)
            batch = preprocess_batch(images[start:end])
            with torch.no_grad():
                #generated_ids = model.generate(batch)
                if isinstance(model, torch.nn.DataParallel):
                    generated_ids = model.module.generate(batch, max_new_tokens=512)  # 设置足够大的 max_new_tokens
                else:
                    generated_ids = model.generate(batch, max_new_tokens=512)
            
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            texts.extend(generated_texts)
        return texts
    except Exception as e:
        raise RuntimeError(f"OCR推理失败: {str(e)}")

# ---------------- Gradio UI ----------------
with gr.Blocks(title="TrOCR 批量图片 OCR 识别") as demo:
    gr.Markdown("# 批量图片 OCR 识别（支持多图并行上传）")
    gr.Markdown("一次拖拽多张图片，使用 microsoft/trocr-base-printed 进行 OCR 识别，进度实时可见。")

    with gr.Row():
        img_input = gr.File(
            file_count="multiple",
            file_types=["image"],
            label="上传图片（可多选）"
        )

    # 图片显示区域
    gallery = gr.Gallery(
        label="上传的图片",
        columns=4,
        height="auto",
        object_fit="contain"
    )

    run_btn = gr.Button("开始识别", variant="primary")
    progress_bar = gr.Progress()
    result_df = gr.Dataframe(
        headers=["文件名", "识别文本"],
        datatype=["str", "str"],
        interactive=False
    )
    csv_file = gr.File(label="下载 CSV 结果")

    def update_gallery(files: List[Any]) -> List[Tuple[np.ndarray, str]]:
        """更新图片展示区域"""
        if not files:
            return []
        
        gallery_content = []
        for file_info in files:
            path = file_info.name if hasattr(file_info, "name") else file_info  # 兼容Gradio版本
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            filename = os.path.basename(path)
            gallery_content.append((img_rgb, filename))
        
        return gallery_content

    def handle_files(files: List[Any]) -> Tuple[List[List[str]], str]:
        """处理上传文件并返回 DataFrame 和 CSV"""
        if not files:
            return [], None

        try:
            images = []
            paths = []
            for file_info in files:
                path = file_info.name if hasattr(file_info, "name") else file_info
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"无法读取图片: {path}")
                images.append(img)
                paths.append(os.path.basename(path))

            # 批量OCR推理
            texts = batch_infer(images, progress_bar)

            # 生成 DataFrame
            rows = [[p, t] for p, t in zip(paths, texts)]
            csv_path = f"/tmp/ocr_result_{int(time.time())}.csv"
            pd.DataFrame(rows, columns=["文件名", "识别文本"]).to_csv(csv_path, index=False)

            return rows, csv_path
        except Exception as e:
            raise gr.Error(f"处理文件时出错: {str(e)}")

    # 设置交互逻辑
    img_input.change(
        fn=update_gallery,
        inputs=img_input,
        outputs=gallery
    )

    run_btn.click(
        fn=handle_files,
        inputs=img_input,
        outputs=[result_df, csv_file]
    )

# ---------------- 启动 ----------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)
