#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whisper_gradio_asr_with_lang_mic.py

Whisper-large-v3 本地语音识别服务
新增：
  1. 语言选择下拉菜单：自动检测 / 中文 / 英文
  2. 在线语音输入（麦克风）
"""

import os
import tempfile
import logging
from typing import List, Dict, Any

import numpy as np
import torch
import gradio as gr
import librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ============ 日志 ============
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("whisper-asr")

# ============ 配置 ============
MODEL_DIR = "/Data/DEMO/MODEL/openai/whisper-large-v3"   # 本地模型路径
CHUNK_LENGTH_S = 30                           # 分片长度
SAMPLE_RATE = 16000                           # Whisper 输入采样率

# 设备选择
if torch.cuda.is_available():
    device = "cuda"
    logger.info(f"使用 GPU: {torch.cuda.device_count()} 个，device={device}")
else:
    device = "cpu"
    logger.info("使用 CPU 进行推理（速度较慢）")

# ============ 加载模型 ============
def load_whisper_model(model_dir: str, device: str = device):
    logger.info(f"从路径加载模型: {model_dir}")
    processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    logger.info("模型加载完成")
    return processor, model

processor, model = load_whisper_model(MODEL_DIR, device=device)

# ============ 工具函数 ============
def load_audio_to_array(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, orig_sr = librosa.load(file, sr=sr, mono=True)
    return y

def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    def fmt_ts(s: float) -> str:
        h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02d}:{m:02d}:{int(sec):02d},{ms:03d}"
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}")
        lines.append(seg['text'])
        lines.append("")
    return "\n".join(lines)

def segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    def fmt_ts(s: float) -> str:
        h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02d}:{m:02d}:{int(sec):02d}.{ms:03d}"
    lines = ["WEBVTT\n"]
    for seg in segments:
        lines.append(f"{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}")
        lines.append(seg['text'])
        lines.append("")
    return "\n".join(lines)

# ============ 推理函数 ============
def transcribe_file(audio_path: str, language: str = "auto", chunk_length_s: int = CHUNK_LENGTH_S) -> Dict[str, Any]:
    waveform = load_audio_to_array(audio_path, sr=SAMPLE_RATE)
    total_sec = len(waveform) / SAMPLE_RATE
    logger.info(f"加载音频: {audio_path}, 长度 {total_sec:.2f}s")

    chunks = []
    if total_sec <= chunk_length_s:
        chunks.append((0.0, total_sec, waveform))
    else:
        start = 0.0
        while start < total_sec:
            end = min(start + chunk_length_s, total_sec)
            start_idx = int(start * SAMPLE_RATE)
            end_idx = int(end * SAMPLE_RATE)
            chunk_wav = waveform[start_idx:end_idx]
            chunks.append((start, end, chunk_wav))
            start += chunk_length_s

    segments = []
    full_text_parts = []

    for (start_s, end_s, chunk_wav) in chunks:
        inputs = processor(chunk_wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(device=model.device, dtype=model.dtype)

        gen_kwargs = {}
        if language != "auto":
            lang_token = f"<|{language}|>"
            forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
            gen_kwargs["forced_decoder_ids"] = forced_ids

        with torch.no_grad():
            generated_ids = model.generate(input_features, **gen_kwargs)
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        text = decoded.strip()
        if text:
            segments.append({"start": float(start_s), "end": float(end_s), "text": text})
            full_text_parts.append(text)

    full_text = " ".join(full_text_parts).strip()
    return {"text": full_text, "segments": segments, "duration": total_sec}

# ============ Gradio 接口 ============
def transcribe_endpoint(audio_file, lang_choice):
    if audio_file is None:
        raise gr.Error("请上传音频或使用麦克风录音")

    if isinstance(audio_file, dict) and "name" in audio_file:
        audio_path = audio_file["name"]
    else:
        audio_path = audio_file

    lang_map = {"自动检测": "auto", "中文": "zh", "英文": "en"}
    lang = lang_map.get(lang_choice, "auto")

    res = transcribe_file(audio_path, language=lang)
    text = res["text"]
    segments = res["segments"]

    tmpdir = tempfile.mkdtemp(prefix="whisper_asr_")
    srt_text = segments_to_srt(segments)
    vtt_text = segments_to_vtt(segments)

    srt_path = os.path.join(tmpdir, "transcript.srt")
    vtt_path = os.path.join(tmpdir, "transcript.vtt")
    txt_path = os.path.join(tmpdir, "transcript.txt")

    with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_text)
    with open(vtt_path, "w", encoding="utf-8") as f: f.write(vtt_text)
    with open(txt_path, "w", encoding="utf-8") as f: f.write(text)

    return text, segments, srt_path, vtt_path, txt_path

with gr.Blocks(title="Whisper ASR Service") as demo:
    gr.Markdown("## Whisper 语音识别 (local whisper-large-v3)")
    with gr.Row():
        audio_input = gr.Audio(label="上传音频或录音", type="filepath", sources=["upload", "microphone"])
        lang_choice = gr.Dropdown(choices=["自动检测", "中文", "英文"], value="自动检测", label="语言选择")
    trans_btn = gr.Button("转录", variant="primary")
    out_text = gr.Textbox(label="转录文本", lines=8)
    out_segments = gr.JSON(label="分段结果")
    srt_file = gr.File(label="SRT 文件")
    vtt_file = gr.File(label="VTT 文件")
    txt_file = gr.File(label="纯文本")

    trans_btn.click(fn=transcribe_endpoint,
                    inputs=[audio_input, lang_choice],
                    outputs=[out_text, out_segments, srt_file, vtt_file, txt_file])

if __name__ == "__main__":
    #demo.launch(server_name="0.0.0.0", server_port=8000, show_error=True)
    demo.launch(server_name="0.0.0.0", server_port=7863, show_error=True, share=True)
