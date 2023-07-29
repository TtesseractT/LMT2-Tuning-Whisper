from transformers import pipeline
import torch
import argparse
import librosa as lr
import warnings
import os
import csv
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device name:", torch.cuda.get_device_name(0))  # Prints the name of the first GPU
    print("CUDA device count:", torch.cuda.device_count())  # Prints the number of available GPUs
else:
    print("CUDA is not available.")

pipe = pipeline("automatic-speech-recognition", model='./Validation', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
tokenizer = pipe.tokenizer

def transcribe(audio):
    out = pipe(audio)
    return out["text"]

def LoadAudio(file_path):
    x, sr = lr.load(file_path, sr=16000)
    return x

def split_audio(audio, duration):
    sample_rate = 16000
    audio_duration = len(audio) / sample_rate
    num_segments = int(audio_duration / duration)
    segments = []
    for i in range(num_segments):
        start = int(i * duration * sample_rate)
        end = int((i + 1) * duration * sample_rate)
        segment = audio[start:end]
        segments.append(segment)
    
    # Check if there is remaining audio that doesn't fit into a full segment
    remaining_audio = audio[num_segments * duration * sample_rate:]
    if len(remaining_audio) > 0:
        segments.append(remaining_audio)
    
    return segments

warnings.filterwarnings("ignore")
folder_path = './AudioFiles'
output_folder = './Logs'
segment_duration = 29  # Segment duration in seconds

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all files in the input folder
files = os.listdir(folder_path)

# Prepare the output TSV file
tsv_file = os.path.join(output_folder, 'transcriptions.tsv')
with open(tsv_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['File_Name', 'TS_Data'])  # Write the headers
    
    # Process each file and write to the TSV file
    for file in tqdm(files, desc="Transcribing files", unit="file"):
        # Construct the full path of the audio file
        audio_path = os.path.join(folder_path, file)
        
        # Load the audio
        audio = LoadAudio(audio_path)
        
        # Calculate the actual duration of the audio in seconds
        audio_duration = len(audio) / 16000
        
        # Split the audio into segments
        audio_segments = split_audio(audio, segment_duration)
        
        # Transcribe each audio segment
        transcripts = []
        for i, segment in enumerate(audio_segments):
            segment_transcript = transcribe(segment)
            transcripts.append(segment_transcript)
        
        # Merge the transcripts from all segments
        full_transcript = " ".join(transcripts)
        
        # Write the file name and transcription to the TSV file
        writer.writerow([file, full_transcript])

print("Transcription completed. TSV file created.")


'''
folder_path = './Audio'
output_folder = './Logs'
'''