#!/bin/bash
# 脚本功能：将图片转为FP32张量，发送Triton推理请求
# 依赖：需安装python3、opencv-python、numpy

# 1. 定义参数（根据实际情况修改）
MODEL_NAME="densenet_onnx"
TRITON_URL="http://172.18.8.208:8000"
IMG_PATH="/workspace/images/mug.jpg"
INPUT_NAME="data_0"  # 模型真实输入名
#SHAPE="[1,3,224,224]"
SHAPE="[3,224,224]"

# 2. 检查依赖
if ! command -v python3 &> /dev/null; then
    echo "错误：未安装python3，请先执行：apt install python3 python3-pip -y"
    exit 1
fi

# 3. 安装python依赖（首次执行需要）
pip3 install opencv-python numpy --quiet

# 4. 生成张量请求文件（通过Python处理图片）
python3 - << EOF
import cv2
import numpy as np
import json

# 图片预处理（适配DenseNet输入要求）
img = cv2.imread("$IMG_PATH")
if img is None:
    print(f"错误：无法读取图片 {IMG_PATH}")
    exit(1)
# 缩放+转RGB+CHW格式+归一化+增加batch维度
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))  # HWC -> CHW
img = img.astype(np.float32) / 255.0  # 归一化到0-1
img = np.expand_dims(img, axis=0)     # [3,224,224] -> [1,3,224,224]

# 转为一维数组（Triton要求JSON中数据为一维）
data = img.flatten().tolist()

# 生成推理请求JSON
request = {
    "inputs": [{
        "name": "$INPUT_NAME",
        "shape": $SHAPE,
        "datatype": "FP32",
        "data": data
    }]
}

# 保存请求文件
with open("densenet_infer_request.json", "w") as f:
    json.dump(request, f)
print("✅ 张量请求文件已生成：densenet_infer_request.json")
EOF

# 5. 发送推理请求
echo -e "\n🚀 发送推理请求到 $TRITON_URL/v2/models/$MODEL_NAME/infer..."
curl -X POST "$TRITON_URL/v2/models/$MODEL_NAME/infer" \
    -H "Content-Type: application/json" \
    -d @densenet_infer_request.json | python3 -m json.tool  # 格式化输出结果
