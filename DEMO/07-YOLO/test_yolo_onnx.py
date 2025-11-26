import onnxruntime as ort
import numpy as np
import cv2

ONNX_MODEL = "/Data/MODEL/YOLO/yolov13x.onnx"
IMAGE_PATH = "test.jpg"   # 换成你要测试的图片

def preprocess(img_path, img_size=640):
    """
    图像预处理：resize + normalize
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"❌ 找不到图片: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # resize 到网络输入大小
    img_resized = cv2.resize(img, (img_size, img_size))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC -> CHW
    img_resized = np.expand_dims(img_resized, 0)        # NCHW

    return img_resized, (h, w), img


def run_inference(onnx_model, img_path):
    """
    运行 ONNX 模型推理
    """
    # 初始化 ONNX Runtime
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_model, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 图像预处理
    img_tensor, orig_shape, orig_img = preprocess(img_path)
    print(f"输入图像 shape: {img_tensor.shape}")

    # 推理
    outputs = session.run([output_name], {input_name: img_tensor})
    preds = outputs[0]
    print(f"✅ 模型输出 shape: {preds.shape}")

    # 这里 preds 的格式要看你训练时的 YOLO 实现，通常是 [batch, num_boxes, xywh+conf+cls]
    return preds, orig_img


if __name__ == "__main__":
    preds, orig_img = run_inference(ONNX_MODEL, IMAGE_PATH)

    # 打印前 5 个预测框
    print("前 5 个预测结果:")
    print(preds[0][:5])
