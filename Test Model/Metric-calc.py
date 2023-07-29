import csv
import random
from nltk.metrics.distance import edit_distance
from nltk.metrics import f_measure
import os
from tqdm import tqdm

def calculate_wer(reference, hypothesis):
    # Calculate the Word Error Rate
    wer = edit_distance(reference.split(), hypothesis.split()) / len(reference.split())
    return wer

def calculate_cer(reference, hypothesis):
    # Calculate the Character Error Rate
    cer = edit_distance(reference, hypothesis) / len(reference)
    return cer

def calculate_word_accuracy(reference, hypothesis):
    # Calculate Word Accuracy
    num_correct_words = sum(1 for ref_word, hyp_word in zip(reference.split(), hypothesis.split()) if ref_word == hyp_word)
    word_accuracy = num_correct_words / len(reference.split())
    return word_accuracy

def calculate_sentence_error_rate(reference, hypothesis):
    # Calculate Sentence Error Rate
    ref_set = set(reference.split())
    hyp_set = set(hypothesis.split())
    f1 = f_measure(ref_set, hyp_set)
    if f1 is not None:
        ser = 1 - f1  # Sentence Error Rate is 1 minus the F1 score
    else:
        ser = None
    return ser

def calculate_f1_score(reference, hypothesis):
    # Calculate F1-Score
    if not reference or not hypothesis:
        return 0.0
    
    ref_set = set(reference.split())
    hyp_set = set(hypothesis.split())
    f1 = f_measure(ref_set, hyp_set)
    return f1

def compare_transcriptions(validated_file, speech_recognition_file, num_samples=4000, num_tests=10):
    # Read and parse TSV files
    validated_data = {}
    speech_recognition_data = {}

    with open(validated_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            validated_data[row[0]] = row[1]  # Map file name to sentence

    with open(speech_recognition_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            speech_recognition_data[row[0]] = row[1]  # Map file name to sentence

    total_results = []

    with tqdm(total=num_samples*num_tests, desc="Running Tests") as pbar:
        for _ in range(num_tests):
            results = []
            selected_references = random.sample(list(validated_data.keys()), num_samples)
            for reference in selected_references:
                validated_sentence = validated_data.get(reference, '')
                hypothesis = speech_recognition_data.get(reference, '')

                # Calculate metrics
                wer = calculate_wer(validated_sentence, hypothesis)
                cer = calculate_cer(validated_sentence, hypothesis)
                word_accuracy = calculate_word_accuracy(validated_sentence, hypothesis)
                ser = calculate_sentence_error_rate(validated_sentence, hypothesis)
                f1_score = calculate_f1_score(validated_sentence, hypothesis)

                # Append the metric values to results only if hypothesis is not None
                if hypothesis is not None:
                    results.append([wer, cer, ser, word_accuracy, f1_score])
                pbar.update(1)

            total_results.extend(results)

    # Calculate averages
    metric_sums = [sum([value if value is not None else 0 for value in metric_values]) for metric_values in zip(*total_results)]
    metric_counts = [len([value for value in metric_values if value is not None]) for metric_values in zip(*total_results)]
    avg_results = [metric_sum / metric_count if metric_sum is not None and metric_count != 0 else None for metric_sum, metric_count in zip(metric_sums, metric_counts)]

    return avg_results


# Usage
script_directory = os.path.dirname(os.path.abspath(__file__))
validated_tsv_file = os.path.join(script_directory, 'new-val-ref-Longform-20K.tsv')
speech_recognition_tsv_file = os.path.join(script_directory, 'transcriptions.tsv')

# Run the test 10 times and calculate averages
num_tests = 1
num_samples = 5000
avg_results = compare_transcriptions(validated_tsv_file, speech_recognition_tsv_file, num_samples=num_samples, num_tests=num_tests)

# Save averages to a file in the same directory as the script
results_file = os.path.join(script_directory, 'results.tsv')

with open(results_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['Metric', 'Average'])
    writer.writerow(['Word Error Rate (WER)', f'{avg_results[0] * 100:.2f}%'])
    writer.writerow(['Character Error Rate (CER)', f'{avg_results[1] * 100:.2f}%'])
    writer.writerow(['Sentence Error Rate (SER)', f'{avg_results[2] * 100:.2f}%'])
    writer.writerow(['Word Accuracy', f'{avg_results[3] * 100:.2f}%'])
    writer.writerow(['F1-Score', f'{avg_results[4]:.4f}'])

print("Results saved successfully!")
