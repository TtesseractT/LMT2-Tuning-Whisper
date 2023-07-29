import os
import csv
from tqdm import tqdm

tsv_file = 'validated.tsv'  # Replace with the path to your TSV file
tsv_file1 = 'new-val.tsv'  # Replace with the path to your new TSV file
folder_path = 'clips'  # Replace with the path to the folder containing the files

# Step 1: Read the TSV file and extract the 'path' column
paths_to_keep = set()
with open(tsv_file, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        path = row['path']
        paths_to_keep.add(path)

# Step 2: Remove files in the folder that are not referenced in the TSV file
removed_files = []
for filename in tqdm(os.listdir(folder_path), desc='Removing Files'):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and filename not in paths_to_keep:
        os.remove(file_path)
        removed_files.append(filename)

# Step 3: Edit the TSV file to keep only the 'path' and 'sentence' columns
tsv_output_file = 'new-val.tsv'  # Replace with the desired output file name
with open(tsv_file, 'r', encoding='utf-8') as input_file, open(tsv_output_file, 'w', newline='', encoding='utf-8') as output_file:
    reader = csv.reader(input_file, delimiter='\t')
    writer = csv.writer(output_file, delimiter='\t')
    header = next(reader)
    path_index = header.index('path')
    sentence_index = header.index('sentence')
    writer.writerow(['path', 'sentence'])
    for row in reader:
        path = row[path_index]
        sentence = row[sentence_index]
        writer.writerow([path, sentence])

print("File parsing and filtering completed.")
#print("Removed files:", removed_files)

# Step 1: Read the TSV file and extract the 'path' column
paths_in_tsv = set()
with open(tsv_file1, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        path = row['path']
        paths_in_tsv.add(path)

# Step 2: Get the list of files in the folder
files_in_folder = os.listdir(folder_path)

# Step 3: Check for files in the folder that are not referenced in the TSV file
unreferenced_files = [filename for filename in files_in_folder if filename not in paths_in_tsv]

# Step 4: Print the list of unreferenced files
if unreferenced_files:
    print("Unreferenced files found in the folder:")
    for file in unreferenced_files:
        print(file)
else:
    print("No unreferenced files found in the folder.")
    
'''
# Step 1: Read the TSV file and extract the 'path' column
paths_to_keep = set()
with open(tsv_file1, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        path = row['path']
        sentence = row['sentence']
        if any(char in sentence for char in "',‘\\?'\""):
            file_path = os.path.join(folder_path, path)
            if os.path.isfile(file_path):
                os.remove(file_path)
        else:
            paths_to_keep.add(path)

# Step 2: Edit the TSV file to keep only the 'path' and 'sentence' columns and remove files with invalid sentences
tsv_output_file = 'new-validated.tsv'  # Replace with the desired output file name
with open(tsv_file1, 'r', encoding='utf-8') as input_file, open(tsv_output_file, 'w', newline='', encoding='utf-8') as output_file:
    reader = csv.reader(input_file, delimiter='\t')
    writer = csv.writer(output_file, delimiter='\t')
    header = next(reader)
    path_index = header.index('path')
    sentence_index = header.index('sentence')
    writer.writerow(['path', 'sentence'])
    for row in reader:
        path = row[path_index]
        sentence = row[sentence_index]
        if path in paths_to_keep:
            writer.writerow([path, sentence])

print("File parsing and filtering completed.")
'''
