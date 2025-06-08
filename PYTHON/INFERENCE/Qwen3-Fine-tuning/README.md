
# Qwen3大模型微调入门实战（完整代码，显存要求约32GB）

参考文档： https://github.com/Zeyi-Lin/Qwen3-Medical-SFT

Qwen3是阿里通义实验室最近开源的大语言模型，发布时便登顶了开源LLM榜单第一名。同时，Qwen系列模型也超越LLaMA，成为了HuggingFace上最受欢迎的开源LLM。
可以说，不论是进行研究学习，还是应用落地，Qwen已经逐渐成为开发者的最优选项之一。

以Qwen3作为基座大模型，通过全参数微调的方式，实现垂直专业领域聊天，甚至支持DeepSeek R1 / QwQ式的带推理过程的对话，是学习LLM微调的入门任务。

本文使用 Qwen3-1.7b 模型在 delicate_medical_r1_data 数据集上做全参数微调训练，实现让微调后的Qwen3支持对医学问题进行DeepSeek R1式的推理回复。
训练中用到了transformers、datasets等工具，同时使用SwanLab监控训练过程、评估模型效果。

全参数微调需要大约32GB显存，如果你的显存大小不足，可以使用Qwen3-0.6b，或Lora微调。
代码：完整代码直接看本文第5节 或 Github：https://github.com/Zeyi-Lin/Qwen3-Medical-SFT
实验日志过程：qwen3-1.7B-linear - SwanLab，或 SwanLab基线社区 搜索“qwen3-sft-medical”
模型：Modelscope
数据集：delicate_medical_r1_data
SwanLab：https://swanlab.cn

知识点：什么是全参数微调？
大模型全参数微调是指对预训练大模型的所有参数进行更新和优化，区别于部分参数微调和LoRA微调。
这种方法通过将整个模型权重（包括底层词嵌入、中间特征提取层和顶层任务适配层）在下游任务数据上进行梯度反向传播，使模型整体适应新任务的需求。相比仅微调部分参数，全参数微调能更充分地利用预训练模型的泛化能力，并针对特定任务进行深度适配，通常在数据差异较大或任务复杂度较高的场景下表现更优。

不过，全参数微调往往需要更高的计算资源和存储开销，且存在过拟合风险（尤其在小数据集上）。实际应用中常结合学习率调整、参数分组优化或正则化技术来缓解这些问题。

全参数微调多用于对模型表现性能要求较高的场景，例如专业领域知识问答或高精度文本生成。

更多微调技术可参考：https://zhuanlan.zhihu.com/p/682082440

```
1. 环境安装

modelscope==1.22.0
transformers>=4.50.0
datasets==3.2.0
accelerate
pandas
addict


2. 准备数据集
本示例使用的是 delicate_medical_r1_data 数据集，该数据集主要被用于医学对话模型。

该数据集由2000多条数据组成，每条数据包含Instruction、question、think、answer、metrics五列：

本示例只取question、think、answer这三列：
question：用户提出的问题，即模型的输入
think：模型的思考过程。大家如果用过DeepSeek R1的话，回复中最开始的思考过程就是这个。
answer：模型思考完成后，回复的内容。
本示例的训练任务，便是希望微调后的大模型，能够根据question，给用户一个think+answer的组合回复，并且think和answer在网页上的展示有区分。

3. 加载模型


4. 配置训练记录工具


5. 完整代码

python3 medical_dataset_splitter.py

python3 medical_dataset_splitter.py \
  --dataset_name krisfu/delicate_medical_r1_data \
  --output_dir ./data_split \
  --train_ratio 0.85 \
  --seed 42

python3 medical_dataset_splitter_DOUBAO.py \
  --dataset_name krisfu/delicate_medical_r1_data \
  --output_dir ./data_split \
  --train_ratio 0.85 \
  --seed 42

全参数微调：
python3 train_DOUBAO_optimization.py

LoRA微调：
python3 train_lora_DOUBAO_optimization.py






```
