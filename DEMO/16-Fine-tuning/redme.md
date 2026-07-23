




```
finetune_test/
├── configs/
│   ├── gemma2_2b_config.yaml
│   ├── gemma2_9b_config.yaml
│   └── gemma2_27b_config.yaml
├── data/
│   └── sample_data.jsonl  # 你的训练数据
├── scripts/
│   ├── train_full.py      # 全量微调
│   ├── train_lora.py      # LoRA微调
│   ├── train_qlora.py     # QLoRA微调
│   ├── train_ptuning.py   # P-Tuning微调
│   └── utils.py           # 工具函数
└── run_experiments.sh     # 批量运行脚本



google/gemma-4-E2B
huggingface-cli download google/gemma-4-E2B --local-dir /Data/MODEL/google/gemma-4-E2B

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download google/gemma-2-2b --local-dir /Data/MODEL/google/gemma-2-2b

/Data/MODEL/google/gemma-4-E2B/    10.3 GB




https://unsloth.ai/docs/zh/mo-xing/gemma-4/train
docker run -it -d --shm-size=8G --gpus all --network host --cap-add SYS_ADMIN -v /Data:/Data nvcr.io/nvidia/pytorch:23.10-py3     # 推荐使用这个镜像
pip download -r vit_0.py.requirements.txt --dest /Data/whl -v   # 把requirements.txt文件中列出的包全都先下载到/Data/IMAGES/whl目录，不安装。
pip install -r vit_0.py.requirements.txt --find-links=/Data/whl --no-index #从/Data/IMAGES/whl寻找安装包安装，不要連接到 PyPI。


finetune_test/
├── configs/
│   ├── gemma2_2b_config.yaml
│   ├── gemma2_9b_config.yaml
│   └── gemma2_27b_config.yaml
├── data/
│   └── sample_data.jsonl  # 你的训练数据
├── scripts/
│   ├── train_full.py      # 全量微调
│   ├── train_lora.py      # LoRA微调
│   ├── train_qlora.py     # QLoRA微调
│   ├── train_ptuning.py   # P-Tuning微调
│   └── utils.py           # 工具函数
└── run_experiments.sh     # 批量运行脚本

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

pip download transformers datasets peft --dest /Data/whl -v
pip install transformers datasets peft --find-links=/Data/whl --no-index
python3 train_full.py


docker run -it -d --shm-size=8G --gpus all --network host --cap-add SYS_ADMIN -v /Data:/Data nvcr.io/nvidia/pytorch:25.09-py3
/Data/DEMO/14-LLaMA-Factory/finetune_test/scripts
python3 train_full.py



```
