import os
import pandas as pd
from tqdm import tqdm

tsv_file = 'New-Val-Ref-LONGFORM-20k.tsv'
audio_folder = 'clips'


# Read the TSV file into a pandas DataFrame
df = pd.read_csv(tsv_file, delimiter='\t')

# Get the list of audio files in the folder
audio_files = os.listdir(audio_folder)

# Check for unreferenced audio files
unreferenced_files = []
for audio_file in audio_files:
    if audio_file not in df['path'].values:
        unreferenced_files.append(audio_file)

# Print unreferenced audio files
if len(unreferenced_files) > 0:
    print("Unreferenced audio files:")
    for file in unreferenced_files:
        print(file)
else:
    print("No unreferenced audio files found.")

# Delete unreferenced audio files with progress bar
with tqdm(total=len(unreferenced_files), desc='Deleting audio files', unit='file') as pbar:
    for file in unreferenced_files:
        file_path = os.path.join(audio_folder, file)
        os.remove(file_path)
        pbar.update(1)

print("Deletion complete.")
