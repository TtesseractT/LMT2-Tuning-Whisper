import random
import csv
import tqdm

tsv_file = 'New-Val-Ref-LONGFORM.tsv'  # Replace with the path to your TSV file
output_file = 'New-Val-Ref-LONGFORM-20k.tsv'  # Replace with the desired output file name

# Step 1: Read the TSV file and extract all the lines
lines = []
with open(tsv_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='\t')
    header = next(reader)
    lines = list(reader)

# Step 2: Randomly select 5,000 lines with tqdm progress monitoring
selected_lines = random.sample(lines, 20000)

# Step 3: Save the selected lines in the new TSV file with tqdm progress monitoring
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(header)
    for line in tqdm.tqdm(selected_lines, desc="Saving lines", unit="line"):
        writer.writerow(line)

print("Random selection and saving completed.")