YOLO11


```
https://www.modelscope.cn/datasets/Genius-Society/tt100k/tree/master/data
https://huggingface.co/datasets/Genius-Society/tt100k/blob/main/README.md
https://huggingface.co/datasets/Genius-Society/tt100k/tree/main/default/train


/Data/MODEL/YOLO11

docker run -it -d --shm-size=8G --gpus all --network host --cap-add SYS_ADMIN -v /Data:/Data nvcr.io/nvidia/pytorch:24.10-py3 # 适用V100

pip download modelscope --dest /Data/whl
pip install modelscope --find-links=/Data/whl --no-index

解决 ubuntu24.04 使用pip时的信息提醒：error: externally-managed-environment
mv /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3.12/EXTERNALLY-MANAGED.bk

设置清华大学的 pip 源:
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple;
mkdir -p /root/.config/pip;
cat > /root/.config/pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF

python3 -m pip install --upgrade pip

pip download -r yolo_requirements.txt --dest /Data/whl
pip install -r yolo_requirements.txt --find-links=/Data/whl --no-index
pip uninstall numpy -y
pip install numpy==1.26.4


# 0 环境和 LoRA 数学自检（会下载 yolo11n.pt）
python3 check_env.py --selftest

# 1 数据转换（45 类，other 作 val，软链接不占额外 18G）
python3 tt100k2yolo.py --src /data/TT100K/data --out /data/tt100k_yolo --min-instances 100
python3 tt100k2yolo.py --src /Data/MODEL/tt100k_data --out /Data/MODEL/tt100k_yolo --min-instances 100
python3 tt100k2yolo.py --src /Data/MODEL/tt100k_data --out /Data/MODEL/tt100k_yolo --min-instances 10000

# 2 LoRA 冒烟（10% 数据、8 epoch、单卡）
python3 train_lora.py --data /data/tt100k_yolo/data.yaml --device 0 --batch 16 --nbs 16 --epochs 8 --fraction 0.1
python3 train_lora.py --model /Data/MODEL/Ultralytics/YOLO11/yolo11n.pt --data /Data/MODEL/tt100k_yolo/data.yaml --device 0 --batch 16 --nbs 16 --epochs 8 --fraction 0.1

# 3 LoRA 正式训练（双卡），再合并验证
python train_lora.py --data /data/tt100k_yolo/data.yaml --device 0,1 --batch 64 --nbs 32
python3 train_lora.py --model /Data/MODEL/Ultralytics/YOLO11/yolo11n.pt --data /Data/MODEL/tt100k_yolo/data.yaml --device 0,1 --batch 64 --nbs 32

python merge_lora.py --lora-ckpt runs/tt100k/lora_r8/weights/best.pt --out yolo11n_tt100k_lora.pt --check --data /data/tt100k_yolo/data.yaml
python3 merge_lora.py --lora-ckpt runs/detect/runs/tt100k/lora_r8/weights/best.pt --out yolo11n_tt100k_lora.pt --check --data /Data/MODEL/tt100k_yolo/data.yaml

# 4 全量基线 + 对齐评估
python train_full.py --data /data/tt100k_yolo/data.yaml --device 0,1 --preset 640
python3 train_full.py --data /Data/MODEL/tt100k_yolo/data.yaml --device 0,1 --preset 640

```
