import os
import csv
from tqdm import tqdm

tsv_file = 'validated.tsv'  # Replace with the path to your TSV file
tsv_file1 = 'new-val.tsv'  # Replace with the path to your new TSV file
folder_path = 'clips'  # Replace with the path to the folder containing the files

# Step 1: Read the TSV file and extract the 'path' column
paths_to_keep = set()
unique_sentences = set()  # Track unique sentences
with open(tsv_file1, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in tqdm(reader, desc="Processing rows", unit="row"):
        path = row['path']
        sentence = row['sentence']
        if (
            any(char in sentence for char in "',!-;£$%^&*()|~@→‘\\?'\"")
            or len(sentence) < 15  # Check if the sentence is less than 5 characters long
            or len(sentence) > 30 # Check if the sentence is more than 15 characters long
            or sentence.count('.') > 2  # Check if the sentence has more than 2 full stops
            or sum(1 for c in sentence if c.isupper() and c.isalpha()) > 1  # Check if the sentence has more than 1 capital letter
            or sentence in unique_sentences  # Check if the sentence has already been encountered
        ):
            file_path = os.path.join(folder_path, path)
            if os.path.isfile(file_path):
                os.remove(file_path)
        else:
            paths_to_keep.add(path)
            unique_sentences.add(sentence)

# Step 2: Edit the TSV file to keep only the 'path' and 'sentence' columns and remove files with invalid sentences
tsv_output_file = 'New-Val-Ref-LONGFORM.tsv'  # Replace with the desired output file name
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
