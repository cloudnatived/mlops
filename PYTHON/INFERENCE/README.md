


modelscope download --model 'Qwen/Qwen2-7b' --local_dir /Data/modelscope/hub/models/Qwen/Qwen2-7b
modelscope download --model 'Qwen/QwQ-32b' --local_dir /Data/modelscope/hub/models/Qwen/QwQ-32b
modelscope download --model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B' --local_dir /Data/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
modelscope download --model 'iic/nlp_structbert_word-segmentation_chinese-base' --local_dir /Data/modelscope/hub/models/iic/nlp_structbert_word-segmentation_chinese-base

python3 -m sglang.check_env
