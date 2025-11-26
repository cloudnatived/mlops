#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
import numpy as np
import pandas as pd
import gradio as gr
from TTS.api import TTS
import logging
import pkg_resources

# ---------------- 配置日志 ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- 配置Hugging Face缓存 ----------------
os.environ["HF_HOME"] = "/Data/MODEL/cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TTS_HOME"] = "/Data/MODEL/coqui/XTTS-v2"

# ---------------- 设备 & XTTS模型配置 ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用设备: {device} | GPU数量: {torch.cuda.device_count()}")

MODEL_PATH = "/Data/MODEL/coqui/XTTS-v2"

try:
    model = TTS(model_path=MODEL_PATH, config_path=os.path.join(MODEL_PATH, "config.json"), progress_bar=True)
    model.to(device)
    logger.info(f"✅ 成功加载XTTS模型: {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ 加载XTTS模型失败: {str(e)}", exc_info=True)
    raise RuntimeError(f"❌ 加载XTTS模型失败: {str(e)}") from e

# ---------------- 核心配置参数 ----------------
BATCH_SIZE = 2  # V100 16GB 建议小批量
LANGUAGE = "zh"  # 支持中文
AUDIO_EXTENSIONS = (".wav", ".mp3")  # 支持的参考音频格式

# ---------------- 1. 文本和音频处理工具函数 ----------------
def validate_text(text: str) -> str:
    """验证并清理输入文本"""
    if not text.strip():
        raise ValueError("输入文本不能为空")
    return text.strip()

def validate_audio(file_path: str) -> str:
    """验证参考音频文件"""
    if not file_path:
        raise ValueError("请上传参考音频文件（WAV或MP3）")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    if not file_path.lower().endswith(AUDIO_EXTENSIONS):
        raise ValueError(f"不支持的音频格式: {file_path}（仅支持{AUDIO_EXTENSIONS}）")
    return file_path

# ---------------- 2. 批量TTS函数 ----------------
def batch_tts(texts: list, speaker_wav: str, progress=gr.Progress()) -> tuple:
    """
    批量文本转语音核心函数
    参数：texts (输入文本列表), speaker_wav (参考音频路径)
    返回：(TTS结果列表, 音频文件路径列表)
    """
    if not texts:
        return [], []
    
    progress(0, desc="初始化TTS任务...")
    total_texts = len(texts)
    all_results = []
    audio_paths = []

    audio_temp_dir = f"/tmp/xtts_audio_{int(time.time())}"
    os.makedirs(audio_temp_dir, exist_ok=True)

    try:
        # 验证参考音频
        speaker_wav = validate_audio(speaker_wav) if speaker_wav else None
        
        for i, batch_idx in enumerate(range(0, total_texts, BATCH_SIZE)):
            progress(i / (total_texts // BATCH_SIZE + 1), desc=f"处理批次 {i+1}/{(total_texts // BATCH_SIZE + 1)}")
            batch_texts = texts[batch_idx:batch_idx + BATCH_SIZE]
            
            for idx, text in enumerate(batch_texts):
                try:
                    validated_text = validate_text(text)
                    audio_path = os.path.join(audio_temp_dir, f"tts_{batch_idx + idx}.wav")
                    
                    model.tts_to_file(
                        text=validated_text,
                        file_path=audio_path,
                        speaker_wav=speaker_wav,
                        language=LANGUAGE
                    )
                    
                    all_results.append({
                        "text": validated_text,
                        "audio_path": audio_path
                    })
                    audio_paths.append(audio_path)
                
                except Exception as inner_e:
                    logger.error(f"TTS处理文本失败: {text} | 错误: {str(inner_e)}", exc_info=True)
                    raise RuntimeError(f"TTS处理文本失败: {text} | 错误: {str(inner_e)}") from inner_e

        return all_results, audio_paths

    except Exception as e:
        logger.error(f"TTS过程出错: {str(e)}", exc_info=True)
        raise RuntimeError(f"TTS过程出错: {str(e)}") from e

# ---------------- 3. Gradio UI交互逻辑 ----------------
def create_ui() -> gr.Blocks:
    """创建Gradio UI，兼容 Gradio 3.x，添加参考音频上传"""
    with gr.Blocks(title="XTTS批量文本转语音（双V100优化版）") as demo:
        gr.Markdown("# 🚀 XTTS批量文本转语音（双V100加速）")
        gr.Markdown(f"""
        - 支持输入：多行文本（每行一条）
        - 支持参考音频：WAV/MP3（用于声音克隆）
        - 配置：批量大小{BATCH_SIZE} | 语言{LANGUAGE}
        - 设备：{device}（{torch.cuda.device_count()}张GPU）
        """)

        with gr.Row():
            text_input = gr.Textbox(
                lines=10,
                placeholder="输入文本，每行一条（例如：\n读一段台词试试\n这是一个测试）",
                label="📤 输入文本（支持多行）",
                elem_id="text-input"
            )
            audio_input = gr.File(
                file_count="single",
                file_types=[".wav", ".mp3"],
                label="🎙️ 上传参考音频（WAV或MP3，用于声音克隆）"
            )

        with gr.Row():
            run_btn = gr.Button("▶️ 开始批量转换", variant="primary", size="lg")

        with gr.Row():
            result_table = gr.Dataframe(
                headers=["输入文本", "音频文件"],
                datatype=["str", "str"],
                label="📊 TTS结果表格"
            )
            audio_preview = gr.Audio(
                label="🖼️ 音频结果预览（可播放/下载）",
                interactive=False
            )
            zip_output = gr.File(label="💾 下载所有音频（ZIP）")

        def handle_tts(input_text: str, audio_file) -> tuple:
            """处理TTS请求：解析文本和参考音频→批量转换→生成结果"""
            if not input_text.strip():
                raise gr.Error("请先输入文本！")
            
            texts = [line.strip() for line in input_text.split("\n") if line.strip()]
            speaker_wav = audio_file.name if audio_file else None
            all_results, audio_paths = batch_tts(texts, speaker_wav)
            if not all_results:
                raise gr.Error("未生成任何音频，请检查输入文本和参考音频！")
            
            pandas_version = pkg_resources.get_distribution("pandas").version
            if pkg_resources.parse_version(pandas_version) >= pkg_resources.parse_version("2.2.0"):
                with pd.option_context("future.no_silent_downcasting", True):
                    table_data = [[res["text"], res["audio_path"]] for res in all_results]
            else:
                table_data = [[res["text"], res["audio_path"]] for res in all_results]
            
            zip_path = f"/tmp/xtts_audio_result_{int(time.time())}.zip"
            import zipfile
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for audio in audio_paths:
                    zipf.write(audio, os.path.basename(audio))

            return table_data, audio_paths[0] if audio_paths else None, zip_path

        run_btn.click(
            fn=handle_tts,
            inputs=[text_input, audio_input],
            outputs=[result_table, audio_preview, zip_output],
            show_progress=True
        )

    return demo

# ---------------- 4. 启动程序 ----------------
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=False,
        debug=False
    )
