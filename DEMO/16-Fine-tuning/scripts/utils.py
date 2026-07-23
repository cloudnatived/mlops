import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PromptEncoderConfig, get_peft_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tokenizer(model_name):
    """加载tokenizer并处理Gemma特殊设置"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Gemma-2 特殊处理：添加EOS token
    if "gemma" in model_name.lower():
        tokenizer.add_eos_token = True
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    return tokenizer

def load_model_for_full(model_name, device_map="auto"):
    """加载模型用于全量微调"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # V100不支持bfloat16
        device_map=device_map,
        trust_remote_code=True,
        use_cache=False,  # 配合gradient_checkpointing
    )
    model.gradient_checkpointing_enable()
    return model

def load_model_for_lora(model_name, lora_config, device_map="auto"):
    """加载模型用于LoRA微调"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_model_for_qlora(model_name, lora_config, device_map="auto"):
    """加载模型用于QLoRA微调（4-bit量化）"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_model_for_ptuning(model_name, ptuning_config, device_map="auto"):
    """加载模型用于P-Tuning"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = get_peft_model(model, ptuning_config)
    model.print_trainable_parameters()
    return model

def load_dataset(data_path, tokenizer, max_length=512):
    """加载JSONL格式数据集"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    
    def format_instruction(example):
        """格式化指令数据，适配Gemma的对话模板"""
        # 简单格式：指令+输出
        text = f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n<start_of_turn>model\n{example['output']}<end_of_turn>"
        return {"text": text}
    
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_instruction)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

def get_lora_target_modules(model_name):
    """根据模型获取LoRA target_modules"""
    if "gemma" in model_name.lower():
        # Gemma-2 的线性层命名
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "llama" in model_name.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        return ["q_proj", "v_proj"]  # 默认

def get_gemma_target_modules(model_name):
    """Gemma专用的target_modules（确保覆盖所有线性层）"""
    if "2b" in model_name.lower():
        # 2B模型的层命名可能略有不同
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
