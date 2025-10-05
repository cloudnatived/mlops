import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
import gradio as gr

# 设置环境变量
os.environ["GRADIO_FRONTEND_SRC"] = "local"

# ---------------- 设备 & YOLO模型配置 ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device} | GPU数量: {torch.cuda.device_count()}")

MODEL_PATH = "/Data/DEMO/CODE/YOLO/yolov13x.pt"

try:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    model.to(device)
    print(f"✅ 成功加载YOLO模型: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ 加载YOLO模型失败: {str(e)}") from e

# ---------------- 核心配置参数 ----------------
BATCH_SIZE = 4
CONF_THRESHOLD = 0.3
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------- 1. 图片读取工具函数 ----------------
def read_image(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"图片不存在: {file_path}")
    if not file_path.lower().endswith(IMAGE_EXTENSIONS):
        raise ValueError(f"不支持的图片格式: {file_path}")
    
    img = cv2.imread(file_path)
    if img is None:
        raise RuntimeError(f"无法读取图片: {file_path}")
    return img

# ---------------- 2. 批量检测函数 ----------------
def batch_detect(image_paths: list, progress=gr.Progress()) -> tuple:
    if not image_paths:
        return [], []
    
    progress(0, desc="初始化检测任务...")
    total_imgs = len(image_paths)
    all_detections = []
    vis_image_paths = []

    vis_temp_dir = f"/tmp/yolo_vis_{int(time.time())}"
    os.makedirs(vis_temp_dir, exist_ok=True)

    try:
        batch_indices = range(0, total_imgs, BATCH_SIZE)
        
        for i, batch_idx in enumerate(batch_indices):
            progress(i/len(batch_indices), desc=f"处理批次 {i+1}/{len(batch_indices)}")
            
            batch_paths = image_paths[batch_idx: batch_idx + BATCH_SIZE]
            batch_imgs = [cv2.cvtColor(read_image(path), cv2.COLOR_BGR2RGB) for path in batch_paths]
            
            results = model(
                batch_imgs,
                conf=CONF_THRESHOLD,
                device=device,
                verbose=False
            )

            for idx, (result, img_path) in enumerate(zip(results, batch_paths)):
                img_filename = os.path.basename(img_path)
                dets = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    dets.append({
                        "filename": img_filename,
                        "class": model.names[cls_id],
                        "confidence": round(float(box.conf[0]), 4),
                        "x1": round(float(box.xyxy[0][0]), 2),
                        "y1": round(float(box.xyxy[0][1]), 2),
                        "x2": round(float(box.xyxy[0][2]), 2),
                        "y2": round(float(box.xyxy[0][3]), 2)
                    })
                all_detections.extend(dets)

                img_bgr = read_image(img_path)
                for det in dets:
                    cv2.rectangle(
                        img_bgr,
                        (int(det["x1"]), int(det["y1"])),
                        (int(det["x2"]), int(det["y2"])),
                        (0, 255, 0),
                        2
                    )
                    label = f"{det['class']} {det['confidence']:.2f}"
                    cv2.putText(
                        img_bgr,
                        label,
                        (int(det["x1"]), int(det["y1"]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                vis_img_path = os.path.join(vis_temp_dir, f"vis_{img_filename}")
                cv2.imwrite(vis_img_path, img_bgr)
                vis_image_paths.append(vis_img_path)

        return all_detections, vis_image_paths

    except Exception as e:
        raise RuntimeError(f"检测过程出错: {str(e)}") from e

# ---------------- 3. 简化的Gradio UI ----------------
def create_ui() -> gr.Blocks:
    """创建简化版UI，避免复杂数据类型"""
    with gr.Blocks(title="YOLO批量目标检测") as demo:
        gr.Markdown("# YOLO批量目标检测")
        
        with gr.Row():
            file_input = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="上传图片"
            )

        with gr.Row():
            run_btn = gr.Button("开始检测", variant="primary")

        with gr.Row():
            result_table = gr.Dataframe(
                headers=["文件名", "目标类别", "置信度"],
                label="检测结果"
            )
            vis_preview = gr.File(
                label="可视化结果",
                file_count="multiple"
            )
            csv_output = gr.File(label="下载结果CSV")

        def handle_detection(files: list) -> tuple:
            if not files:
                raise gr.Error("请先上传图片！")
            
            image_paths = [file.name for file in files]
            all_detections, vis_paths = batch_detect(image_paths)
            
            if not all_detections:
                raise gr.Error("未检测到任何目标！")
            
            table_data = [
                [det["filename"], det["class"], det["confidence"]] 
                for det in all_detections
            ]
            
            csv_path = f"/tmp/yolo_detection_result_{int(time.time())}.csv"
            pd.DataFrame(all_detections).to_csv(csv_path, index=False)
            
            return table_data, vis_paths, csv_path

        run_btn.click(
            fn=handle_detection,
            inputs=file_input,
            outputs=[result_table, vis_preview, csv_output],
            show_progress="minimal"
        )

    return demo

# ---------------- 4. 启动程序 ----------------
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,  # 设置为False避免localhost访问问题
        debug=False
    )
