import os
import csv
from tqdm import tqdm

tsv_file = 'validated.tsv'  # Replace with the path to your TSV file
tsv_file1 = 'new-val.tsv'  # Replace with the path to your new TSV file
folder_path = 'clips'  # Replace with the path to the folder containing the files

# Step 1: Read the TSV file and extract the 'path' column
paths_to_keep = set()
with open(tsv_file1, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in tqdm(reader, desc="Processing rows", unit="row"):
        path = row['path']
        sentence = row['sentence']
        if any(char in sentence for char in "',!-;’£$%^&*()|~@→‘\\?'\"") or len(sentence) < 15 or len(sentence) > 30:
            file_path = os.path.join(folder_path, path)
            if os.path.isfile(file_path):
                os.remove(file_path)
        else:
            paths_to_keep.add(path)

# Step 2: Edit the TSV file to keep only the 'path' and 'sentence' columns and remove files with invalid sentences
tsv_output_file = 'New-Val-Ref-15-30.tsv'  # Replace with the desired output file name
with open(tsv_file1, 'r', encoding='utf-8') as input_file, open(tsv_output_file, 'w', newline='', encoding='utf-8') as output_file:
    reader = csv.reader(input_file, delimiter='\t')
    writer = csv.writer(output_file, delimiter='\t')
    header = next(reader)
    path_index = header.index('path')
    sentence_index = header.index('sentence')
    writer.writerow(['path', 'sentence'])
    for row in tqdm(reader, desc="Writing rows", unit="row"):
        path = row[path_index]
        sentence = row[sentence_index]
        if path in paths_to_keep:
            writer.writerow([path, sentence])

print("File parsing and filtering completed.")
