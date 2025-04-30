# Stable Diffusion文生图
  
  
  
多模态大模型实战篇——本地部署文生图应用（以Stable Diffusion XL 1.0为例）

```

部署教程
1. 模型地址

    模型链接：stable-diffusion-xl-base-1.0
    SD论文：《High-Resolution Image Synthesis with Latent Diffusion Models》

Stable Diffusion（简称SD）是一个由Stability AI公司研发与其他研究者和合作者合作开发的，基于潜在扩散模型 Latent Diffusion Models（LDMs）的多模态领域（text-to-image）开源生成模型，能够根据给定的文本提示来合成高分辨率的图像
Stable Diffusion XL

而Stable Diffusion XL是在SD的基础上的一个二阶段的级联扩散模型（Latent Diffusion Model），包括Base模型和Refiner模型。其中Base模型的主要工作和Stable Diffusion 1.x-2.x一致，具备文生图（txt2img）、图生图（img2img）、图像inpainting等能力。在Base模型之后，级联了Refiner模型，对Base模型生成的图像Latent特征进行精细化提升，其本质上是在做图生图的工作。
如果想要了解更多技术原理细节可以参考：Rocky Ding：深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识1634 赞同 · 246 评论文章

2. 环境依赖
python库依赖和版本，requirements.txt
###############################################
transformers>=4.37.0
accelerate>=0.27.0
modelscope>=1.9.5
numpy>=1.22.3
torch>=1.11.0
gradio>=4.8.0
diffusers>=0.26.3
opencv-python>=4.9.0.80
safetensors>=0.4.2
addict
datasets==2.21.0
simplejson
sortedcontainers
###############################################

然后是GPU依赖，咱们需要有安装好GPU版本的torch。

# 检查cuda是否可用
torch.cuda.is_available()
# out：True

这些依赖都搞定以后，咱们就可以通过下面这部分的代码自验，并且开始下载对应的stable-diffusion-xl-base-1.0模型了（耗时较久）

3.模型调用
###############################################
import cv2 # pip install opencv-python
import torch
import gradio as gr
import numpy as np
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline

# 内嵌风格中的正面提示词定义
prompt_dict = {
    "None": "{prompt}",
    "Enhance": "breathtaking {prompt} . award-winning, professional, highly detailed",
    "Anime": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime, highly detailed",
    "Photographic": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    "Digital Art": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
    "Comic Book": "comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "Fantasy Art": "ethereal fantasy concept art of {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
    "Analog Film": "analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage",
    "Neon Punk": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
    "Isometric": "isometric style {prompt} . vibrant, beautiful, crisp, detailed, ultra detailed, intricate",
    "Low Poly": "low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
    "Origami": "origami style {prompt} . paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition",
    "Line Art": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
    "Craft Clay": "play-doh style {prompt} . sculpture, clay art, centered composition, Claymation",
    "Cinematic": "cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    "3D Model": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
    "Pixel Art": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
    "Texture": "texture {prompt} top down close-up"
}

# 内嵌风格中的负面提示词定义
negative_prompt_dict = {
    "None": "{negative_prompt}",
    "Enhance": "{negative_prompt} ugly, deformed, noisy, blurry, distorted, grainy",
    "Anime": "{negative_prompt} photo, deformed, black and white, realism, disfigured, low contrast",
    "Photographic": "{negative_prompt} drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    "Digital Art": "{negative_prompt} photo, photorealistic, realism, ugly",
    "Comic Book": "{negative_prompt} photograph, deformed, glitch, noisy, realistic, stock photo",
    "Fantasy Art": "{negative_prompt} photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    "Analog Film": "{negative_prompt} painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    "Neon Punk": "{negative_prompt} painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    "Isometric": "{negative_prompt} deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic",
    "Low Poly": "{negative_prompt} noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    "Origami": "{negative_prompt} noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    "Line Art": "{negative_prompt} anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic",
    "Craft Clay": "{negative_prompt} sloppy, messy, grainy, highly detailed, ultra textured, photo",
    "Cinematic": "{negative_prompt} anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    "3D Model": "{negative_prompt} ugly, deformed, noisy, low poly, blurry, painting",
    "Pixel Art": "{negative_prompt} sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    "Texture": "{negative_prompt} ugly, deformed, noisy, blurry"
}


torch.cuda.empty_cache()
# 定义清空函数，返回一组预设的默认值
def clear_fn(value):
    return "", "", "None", 768, 768, 10, 50, None

# 定义拼接图片的函数，接受一个图片列表作为输入
def concatenate_images(images):
    # 得到每张图片的高度，并存储在列表中
    heights = [img.shape[0] for img in images]
    # 计算所有图片宽度的总和
    max_width = sum([img.shape[1] for img in images])
    # 创建一个新的空白图像，大小为最大高度和总宽度
    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0  # 初始化偏移量为0
    for img in images:  # 遍历所有图片
        # 将图片复制到新图像的相应位置上
        concatenated_image[0:img.shape[0], x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]  # 更新偏移量为下一张图片的起始位置
    return concatenated_image  # 返回拼接后的图片

# 下载模型，并初始化模型管道
pipe = pipeline(task=Tasks.text_to_image_synthesis,
                model='AI-ModelScope/stable-diffusion-xl-base-1.0',
                use_safetensors=True,
                model_revision='v1.0.0')

# 定义一个展示管道的函数，该函数使用文本提示生成图片
def display_pipeline(prompt: str,
                     negative_prompt: str,
                     style: str = 'None',
                     height: int = 768,
                     width: int = 768,
                     scale: float = 10,
                     steps: int = 50,
                     seed: int = 0):
    # 如果提示为空，则抛出异常
    if not prompt:
        raise gr.Error('The validation prompt is missing.')
    # 打印预设风格字典中的样式
    print(prompt_dict[style])
    # 使用预设风格格式化正面提示语
    prompt = prompt_dict[style].format(prompt=prompt)
    # 使用预设风格格式化负面提示语
    negative_prompt = negative_prompt_dict[style].format(negative_prompt=negative_prompt)
    # 创建一个随机数生成器，并设定种子以方便复现结果
    generator = torch.Generator(device='cuda').manual_seed(seed)
    # 调用模型管道生成图片
    output = pipe({'text': prompt,
                   'negative_prompt': negative_prompt,
                   'num_inference_steps': steps,
                   'guidance_scale': scale,
                   'height': height,
                   'width': width,
                   'generator': generator
                   })
    # 获取输出结果中的图片
    result = output['output_imgs'][0]
    # 定义存储图片的路径
    image_path = './lora_result.png'
    # 将图片写入文件
    cv2.imwrite(image_path, result)
    # 读取图片文件，并将其从BGR格式转换为RGB格式
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # 返回处理后的图片
    return image
###############################################

如果你的GPU内存不够出现OOM的报错，可以尝试以下解决方案：
缓存清理：
###############################################
# 在报错的位置前看情况做一下GPU缓存清理，比如：
gc.collect()
torch.cuda.empty_cache()
###############################################
2. 换个小点的模型：
###############################################
# 因为 SDXL 1.0 在stable 1.5的基础上做了改进，参数数量从0.98B扩大到6.6B，所以对GPU显存要求比较高
# 可以尝试参数少点的模型比如stable-diffusion-v1-5 
pipe = pipeline(task=Tasks.text_to_image_synthesis,
                model='AI-ModelScope/stable-diffusion-v1-5',
                use_safetensors=True,
                model_revision='v1.0.0')
###############################################

4.基于gradio的界面构建
同样还是以gradio框架为例，通过以下代码快速构建一个用户和模型的交互界面：
###############################################
# 使用Gradio的Blocks API创建一个交互式界面
with gr.Blocks() as demo:
    # 创建一个水平排列的容器，即一行
    with gr.Row():
        # 在行中创建一个列容器，该容器的大小是默认的两倍
        with gr.Column(scale=2):
            # 创建一个多行文本框，用于输入提示词
            prompt = gr.Textbox(label='提示词', lines=3)
            # 创建一个多行文本框，用于输入负向提示词
            negative_prompt = gr.Textbox(label='负向提示词', lines=3)
            # 创建一个下拉菜单，用于选择风格，列出各种风格选项
            style = gr.Dropdown(
                ['None', 'Enhance', 'Anime', 'Photographic', 'Digital Art', 'Comic Book', 'Fantasy Art', 'Analog Film',
                 'Cinematic', '3D Model', 'Neon Punk', 'Pixel Art', 'Isometric', 'Low Poly', 'Origami', 'Line Art',
                 'Craft Clay', 'Texture'], value='None', label='风格')
            with gr.Row():
                # 创建一个滑块，用于选择图片高度
                height = gr.Slider(512, 1024, 768, step=128, label='高度')
                # 创建一个滑块，用于选择图片宽度
                width = gr.Slider(512, 1024, 768, step=128, label='宽度')
            with gr.Row():
                # 创建一个滑块，用于选择引导系数
                scale = gr.Slider(1, 15, 10, step=.25, label='引导系数')
                # 创建一个滑块，用于选择迭代步数
                steps = gr.Slider(25, maximum=100, value=50, step=5, label='迭代步数')

            # 创建一个滑块，用于选择随机数种子，并且有一个随机化按钮
            seed = gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True, label='随机数种子')
            # 创建一个水平排列的容器，即一行
            with gr.Row():
                # 创建一个按钮，用于清除输入
                clear = gr.Button("清除")
                # 创建一个按钮，用于提交输入并生成图片
                submit = gr.Button("提交")
        # 在行中创建另一个列容器，该容器的大小是默认的三倍
        with gr.Column(scale=3):
            # 创建一个用于显示输出图片的组件
            output_image = gr.Image()
    # 当提交按钮被点击时，调用display_pipeline函数并将输入参数传递给它，将结果输出到output_image组件
    submit.click(fn=display_pipeline, inputs=[prompt, negative_prompt, style, height, width, scale, steps, seed],
                 outputs=output_image)
    # 当清除按钮被点击时，调用clear_fn函数，将clear作为输入，输出到指定的组件上，并将它们重置为初始状态
    clear.click(fn=clear_fn, inputs=clear,
                outputs=[prompt, negative_prompt, style, height, width, scale, steps, output_image])

# 运行demo
demo.queue(status_update_rate=1).launch(share=False)
###############################################
至此，在本地机器上运行的话（把上述所有代码都粘贴到一个StableDiffusion_DEMO.py脚本，运行即可）

咱们本地访问http://127.0.0.1:7860 就可以看到效果页面了！

但有些小伙伴的GPU环境在远处服务器上，还需做一道端口转发，例如：

# 端口转发/SSH隧道
（本地执行）ssh -L 9000:127.0.0.1:7860 用户ID@远程机器IP

然后本地访问 http://127.0.0.1:9000/ 即可！

```
