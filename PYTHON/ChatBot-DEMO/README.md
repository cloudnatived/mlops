
#     ChatBot-DEMO
  
```

1. 模型地址
魔搭 官方文档链接 https://modelscope.cn/models/Qwen/QwQ-32B
还有2月初新发布的更加轻量级的通义千问Qwen/QwQ-32B， 感兴趣也可以试试～

2. 环境依赖
首先是主要的python库依赖和版本，requirement.txt
####################################################################
python>=3.8
transformers>=4.37.0
accelerate>=0.27.0
modelscope>=1.9.5
numpy>=1.22.3
torch>=1.11.0
gradio>=4.8.0
####################################################################
然后是GPU依赖，咱们需要有显卡（无论是本地机还是服务器），安装好GPU版本的torch
没有安装好的小伙伴可以参考这篇教程：呆呆兽：Pytorch安装（保姆级教学 真·满血·GPU版）CUDA更新？torch版本？一文全搞定！

# 检查cuda是否可用
####################################################################
torch.cuda.is_available()
# out：True

这些依赖都搞定以后，咱们就可以通过下面这部分的代码自验，并且开始下载对应的Qwen模型了（耗时较久）
from threading import Thread
from typing import Iterator

import gradio as gr
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import  TextIteratorStreamer

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU This demo does not work on CPU.</p>"


if torch.cuda.is_available():
    model_id = "qwen/Qwen1.5-1.8B-Chat"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False
####################################################################
3.模型调用
然后就是模型调用部分的代码，这段代码定义了一个名为 generate 的函数，主要用来生成文本响应，以用于后续咱们ChatBot聊天机器人的交互。
####################################################################
from typing import List, Tuple
def generate(
    message: str, # 用户输入的消息。
    chat_history: List[Tuple[str, str]], # 一个包含聊天历史的列表，其中每个元素是一个包含用户消息和助手（机器人）响应的元组。
    system_prompt: str, # 系统提示，可以作为对话的一部分
    max_new_tokens: int = 1024, # 生成响应时最大的新token数
    temperature: float = 0.6, # 控制生成文本的随机性
    top_p: float = 0.9, # 用于概率限制的参数，有助于控制生成文本的多样性
    top_k: int = 50, # 控制生成过程中每一步考虑的最可能token的数量
    repetition_penalty: float = 1.2, # 用于惩罚重复生成相同token的参数
) -> Iterator[str]:
    conversation = []

    # 函数开始时，会基于system_prompt（如果存在）和chat_history构建一个会话记录。
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    # 使用tokenizer将整个会话转换成模型可以理解的input_ids，并将这些input_ids输入到模型
    input_ids = tokenizer.apply_chat_template(conversation, tokenize=False,add_generation_prompt=True)
    input_ids = tokenizer([input_ids],return_tensors="pt").to(model.device)

    # TextIteratorStreamer对象，用于流式处理文本
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids.input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    # 接下来，创建一个线程Thread来调用模型的generate方法，用于生成文本响应
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    #dictionary update sequence element #0 has length 19; 2 is required

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs) # 每次生成的文本都会yield返回，这样调用者就可以实时地获取机器人的回答～

    # 最后，函数打印出outputs列表，这包含了生成的所有文本片段
    #outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(outputs)
    #yield outputs
####################################################################
4.基于gradio的界面构建

可以调用通模型以后，咱们还差一个WebUI界面方便我们和大模型进行直接的交互，业界主流的框架有gradio、streamlit、Dash等方便用户快速生成AI应用的框架，今天以gradio为例：
通过以下代码可以快速构建一个Chat Interface
####################################################################
chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=None,
    examples=[
        ["你好！你是谁？"],
        ["请简单介绍一下大语言模型?"],
        ["请讲一个小人物成功的故事."],
        ["浙江的省会在哪里?"],
        ["写一篇100字的文章，题目是'人工智能开源的优势'"],
    ],
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown("""<p align="center"><img src="https://modelscope.cn/api/v1/models/qwen/Qwen-VL-Chat/repo?Revision=master&FilePath=assets/logo.jpg&View=true" style="height: 80px"/><p>""")
    gr.Markdown("""<center><font size=8>Qwen1.5-1.8B-Chat Bot </center>""")
    gr.Markdown("""<center><font size=4>通义千问1.5-1.8B（Qwen1.5-1.8B） 是阿里云研发的通义千问大模型系列的70亿参数规模的模型。</center>""")
    chat_interface.render()

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
####################################################################

至此，在本地机器上运行的话（把上述所有代码都粘贴到一个ChatBot_app.py脚本，运行即可）
本地访问http://127.0.0.1:7860 

但如果有些小伙伴的GPU环境在远处服务器上，还需做一道端口转发，例如：
# 端口转发/SSH隧道
（本地执行）ssh -L 9000:127.0.0.1:7860 用户ID@远程机器IP

然后本地访问 http://127.0.0.1:9000/ 即可！
```
