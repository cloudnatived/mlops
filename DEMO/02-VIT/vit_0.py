import os
import time
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoImageProcessor, AutoModelForImageClassification
import gradio as gr

# ---------------- 设备 & 模型 ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device} | GPU数量: {torch.cuda.device_count()}")

local_model_path = "/Data/DEMO/MODEL/google/vit-base-patch16-224"
if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"模型路径不存在: {local_model_path}")

try:
    processor = AutoImageProcessor.from_pretrained(local_model_path)
    model = AutoModelForImageClassification.from_pretrained(local_model_path)
except Exception as e:
    raise RuntimeError(f"加载模型失败: {str(e)}")

if torch.cuda.device_count() > 1:
    print(f"启用 {torch.cuda.device_count()} 张 GPU 并行")
    model = torch.nn.DataParallel(model)
model.to(device).eval()

# ---------------- 批处理函数 ----------------
BATCH_SIZE = 8  # 可根据显存调整
WORKERS = 4      # 数据加载线程数

def preprocess_batch(imgs: List[np.ndarray]) -> torch.Tensor:
    """把多张 numpy 图转成模型需要的 tensor（固定尺寸）"""
    try:
        pil_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs]
        inputs = processor(images=pil_imgs, return_tensors="pt")
        return inputs["pixel_values"].to(device)
    except Exception as e:
        raise RuntimeError(f"预处理失败: {str(e)}")

def batch_infer(images: List[np.ndarray], progress=gr.Progress()) -> List[str]:
    """真正的批推理函数，返回每张图的识别结果"""
    if not images:
        return []

    progress(0, desc="预处理中...")
    n = len(images)
    labels = []

    try:
        # 分批推理
        for start in progress.tqdm(range(0, n, BATCH_SIZE), desc="推理"):
            end = min(start + BATCH_SIZE, n)
            batch = preprocess_batch(images[start:end])
            with torch.no_grad():
                outputs = model(batch)
            
            # 正确处理模型输出
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs  # 某些模型直接返回logits
            
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

            # 解决 DataParallel 下的 config 获取问题
            cfg = model.module.config if isinstance(model, torch.nn.DataParallel) else model.config
            labels.extend([cfg.id2label[i] for i in pred_ids])
        return labels
    except Exception as e:
        raise RuntimeError(f"推理失败: {str(e)}")

# ---------------- Gradio UI ----------------
with gr.Blocks(title="双 V100 批量图片识别") as demo:
    gr.Markdown("# 批量图片识别（支持多图并行上传）")
    gr.Markdown("一次拖拽多张图片，后台自动批量推理，进度实时可见。")

    with gr.Row():
        img_input = gr.File(
            file_count="multiple",
            file_types=["image"],
            label="上传图片（可多选）"
        )

    # 新增：图片显示区域
    gallery = gr.Gallery(
        label="上传的图片",
        columns=4,
        height="auto",
        object_fit="contain"
    )

    run_btn = gr.Button("开始识别", variant="primary")
    progress_bar = gr.Progress()
    result_df = gr.Dataframe(
        headers=["文件名", "识别结果"],
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
            if hasattr(file_info, "name"):  # 兼容不同Gradio版本
                path = file_info.name
            else:
                path = file_info  # 旧版本可能是直接路径字符串
            
            # 读取图片
            img = cv2.imread(path)
            if img is None:
                continue
            
            # 转换为RGB格式用于显示
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            filename = os.path.basename(path)
            gallery_content.append((img_rgb, filename))
        
        return gallery_content

    def handle_files(files: List[Any]) -> Tuple[List[List[str]], str]:
        """包装函数：把结果转成 DataFrame + CSV"""
        if not files:
            return [], None

        try:
            # 正确处理上传的文件对象
            images = []
            paths = []
            for file_info in files:
                if hasattr(file_info, "name"):  # 兼容不同Gradio版本
                    path = file_info.name
                else:
                    path = file_info  # 旧版本可能是直接路径字符串
                
                # 读取图片
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"无法读取图片: {path}")
                
                images.append(img)
                paths.append(os.path.basename(path))

            # 批量推理
            labels = batch_infer(images, progress_bar)

            # 生成 DataFrame
            rows = [[p, l] for p, l in zip(paths, labels)]
            csv_path = f"/tmp/result_{int(time.time())}.csv"
            import pandas as pd
            pd.DataFrame(rows, columns=["文件名", "识别结果"]).to_csv(csv_path, index=False)

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
    demo.launch(server_name="0.0.0.0", server_port=7860)
