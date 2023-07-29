'''
Model Tester from HF Fine tune Whisper Tiny Model

Author: Sabian Hibbs
University of Derby
United Kingdom, England

Licence MIT
'''
from transformers import pipeline
import gradio as gr
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the paths to ffmpeg and ffprobe executables
ffmpeg_path = os.path.join(current_dir, "ffmpeg.exe")
ffprobe_path = os.path.join(current_dir, "ffprobe.exe")

# Set the environment variables
os.environ["GRADIO_AUDIO_FFMPEG"] = ffmpeg_path
os.environ["GRADIO_AUDIO_FFPROBE"] = ffprobe_path

# Pipeline setup
pipe = pipeline(model="Tesseract3D/LMT2-Tuned-S")  # change to "your-username/the-name-you-picked"

def transcribe(audio_file):
    text = pipe(audio_file.name)["text"]
    return text

audio_input = gr.inputs.Audio(label="Select an audio file")

iface = gr.Interface(
    fn=transcribe,
    inputs=audio_input,
    outputs="text",
    title="Whisper LM-S2T-TINY-2",
    description="Real-time demo for English speech recognition using a fine-tuned Whisper tiny model.",
)

iface.launch(share=True, debug=True)
