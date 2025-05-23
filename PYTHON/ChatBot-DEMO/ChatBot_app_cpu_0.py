from typing import List, Tuple, Iterator
import gradio as gr
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4096  # 可根据需要修改

# 使用较小的模型并配置为 CPU
#model_id = "qwen/Qwen-1_8B-Chat"
model_id = "Qwen/Qwen-1_8B-Chat"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.use_default_system_prompt = False

def generate(
    message: str,
    chat_history: List[Tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    conversation = []

    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([input_text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=input_ids.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    yield response

chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS),
        gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6),
        gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9),
        gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50),
        gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2),
    ],
    stop_btn=None,
    examples=[
        ["你好！你是谁？"],
        ["请简单介绍一下大语言模型?"],
        ["请讲一个小人物成功的故事."],
        ["浙江的省会在哪里?"],
        ["写一篇100字的文章，题目是'人工智能开源的优势'"]
    ],
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown("""<p align="center"><img src="https://modelscope.cn/api/v1/models/qwen/Qwen-VL-Chat/repo?Revision=master&FilePath=assets/logo.jpg&View=true" style="height: 80px"/></p>""")
    gr.Markdown("""<center><font size=8>通义千问 Qwen-1.8B-Chat Bot</font></center>""")
    gr.Markdown("""<center><font size=4>通义千问 Qwen-1.8B（qwen/Qwen-1_8B-Chat） 是阿里云研发的通义千问大模型系列中轻量版本，支持 CPU 部署。</font></center>""")
    chat_interface.render()

if __name__ == "__main__":
    demo.queue(max_size=20).launch()

