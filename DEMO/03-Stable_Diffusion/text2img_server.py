#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import gradio as gr
#from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline # 
from diffusers import DPMSolverMultistepScheduler
from accelerate import Accelerator
import os
from datetime import datetime

# 初始化 accelerate
accelerator = Accelerator()

# 检查模型路径
#base_model_path = "./stable-diffusion-v1-5/stable-diffusion-v1-5"
base_model_path = "/Data/DEMO/MODEL/stable-diffusion-v1-5/stable-diffusion-v1-5"
#refiner_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
refiner_model_path = "/Data/DEMO/MODEL/stable-diffusion-v1-5/stable-diffusion-v1-5"

print(f"基础模型路径: {base_model_path}, 存在: {os.path.exists(base_model_path)}")
print(f"优化模型路径: {refiner_model_path}, 存在: {os.path.exists(refiner_model_path)}")

# 加载基础模型
#pipe = StableDiffusionXLPipeline.from_pretrained(
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(accelerator.device)

# 加载优化模型
#refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
refiner = StableDiffusionImg2ImgPipeline.from_pretrained(
    refiner_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(accelerator.device)

# 优化配置
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()  # 减少显存占用

# 尝试启用 xformers（可选）
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers 加速已启用")
except:
    print("xformers 不可用，使用常规模式")

# 使用 accelerate 准备模型
pipe, refiner = accelerator.prepare(pipe, refiner)

def generate_image(prompt, negative_prompt, steps=30, guidance_scale=7.5, width=1024, height=1024, strength=0.3):
    try:
        # 基础模型生成
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            output_type="latent"
        ).images[0]
        
        # 优化模型增强
        image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image[None, :],
            num_inference_steps=steps,
            strength=strength,
        ).images[0]
        
        # 保存结果
        if accelerator.is_main_process:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"outputs/{timestamp}.png"
            os.makedirs("outputs", exist_ok=True)
            image.save(save_path)
            return image, f"图像已保存至: {save_path}"
        else:
            return None, ""
            
    except Exception as e:
        accelerator.print(f"生成失败: {str(e)}")
        return None, f"生成失败: {str(e)}"

# 主进程启动 Gradio
if accelerator.is_main_process:
    with gr.Blocks(title="分布式问生图服务") as demo:
        gr.Markdown("# 文本生成图像服务 (Stable Diffusion XL)")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="提示词", lines=4)
                negative_prompt = gr.Textbox(label="反向提示词", value="low quality, blurry, distorted", lines=2)
                generate_btn = gr.Button("生成图像", variant="primary")
            
            with gr.Column(scale=1):
                output_image = gr.Image(label="生成结果")
                status_text = gr.Textbox(label="状态信息")
        
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt],
            outputs=[output_image, status_text]
        )

    demo.launch(server_port=7861, server_name="0.0.0.0")
else:
    while True:
        pass  # 非主进程保持运行
