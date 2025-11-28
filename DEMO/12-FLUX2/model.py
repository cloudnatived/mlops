# model.py
import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from diffusers import FluxPipeline
import json
import io
from PIL import Image
import base64

class TritonPythonModel:
    def initialize(self, args):
        """模型初始化"""
        self.model_dir = args['model_repository']
        
        # 加载FLUX-2模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 优化配置
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.vae.enable_tiling()
        
        print("FLUX-2模型加载完成")

    def execute(self, requests):
        """处理推理请求"""
        responses = []
        
        for request in requests:
            # 解析输入
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt")
            num_inference_steps = pb_utils.get_input_tensor_by_name(request, "num_inference_steps")
            guidance_scale = pb_utils.get_input_tensor_by_name(request, "guidance_scale")
            
            prompt_text = prompt.as_numpy()[0].decode('utf-8')
            steps = int(num_inference_steps.as_numpy()[0]) if num_inference_steps else 28
            guidance = float(guidance_scale.as_numpy()[0]) if guidance_scale else 3.5
            
            # 生成图像
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                images = self.pipeline(
                    prompt=prompt_text,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=256,
                    width=256
                ).images
            
            # 转换为numpy数组
            image_array = np.stack([np.array(img) for img in images])
            image_array = image_array.transpose(0, 3, 1, 2)  # NHWC to NCHW
            image_array = image_array.astype(np.float32) / 255.0
            
            # 创建输出张量
            output_tensor = pb_utils.Tensor(
                "generated_images", 
                image_array.astype(np.float32)
            )
            
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        
        return responses

    def finalize(self):
        """清理资源"""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        torch.cuda.empty_cache()
