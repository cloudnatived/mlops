```
第0步
git clone https://github.com/jingyaogong/minimind.git

测试已有模型效果
1.环境准备
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

2.下载模型
git clone https://huggingface.co/jingyaogong/MiniMind2

3.命令行问答
# load=0: load from pytorch model, load=1: load from transformers-hf model
python3 eval_model.py --load 1 --model_mode 2

4.或启动WebUI
# 可能需要`python>=3.10` 安装 `pip install streamlit`
# cd scripts
streamlit run web_demo.py

# 从0开始自己训练
1.环境准备
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

2.数据下载



3.开始训练
3.1 预训练（学知识）
python train_pretrain.py

3.2 监督微调（学对话方式）
python3 train_full_sft.py

4.测试模型效果
确保需要测试的模型*.pth文件位于./out/目录下。 也可以直接去此处下载使用我训练的*.pth文件。
python3 eval_model.py --model_mode 1 # 默认为0：测试pretrain模型效果，设置为1：测试full_sft模型效果

单机N卡启动训练方式 (DDP, 支持多机多卡集群)
torchrun --nproc_per_node N train_xxx.py

Experiment
Ⅰ 训练开销


Ⅱ 主要训练步骤
1. 预训练(Pretrain):

2. 有监督微调(Supervised Fine-Tuning):

Ⅲ 其它训练步骤
3. 人类反馈强化学习(Reinforcement Learning from Human Feedback, RLHF)

4. 知识蒸馏(Knowledge Distillation, KD)

5. LoRA (Low-Rank Adaptation)

6. 训练推理模型 (Reasoning Model)


Ⅳ 模型参数设定

```
