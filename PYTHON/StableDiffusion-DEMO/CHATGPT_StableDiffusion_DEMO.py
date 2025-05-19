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
                #model='AI-ModelScope/stable-diffusion-xl-base-1.0',
                model='AI-ModelScope/tiny-stable-diffusion-v1.0'         # 建议替代模型（如需更轻量）
                use_safetensors=True,
                model_revision='v1.0.0')
                device=device  # 加载模型到 CPU

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                # 支持 CPU 运行（如适配低配机器）
    #generator = torch.Generator(device='cuda').manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)           # 生成器绑定到 CPU
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

# 在报错的位置前看情况做一下GPU缓存清理，比如：
#gc.collect()
#torch.cuda.empty_cache()


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
