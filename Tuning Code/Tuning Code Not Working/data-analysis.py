
import os, wave, re, csv, math, statistics, soundfile as sf
from collections import Counter
from tqdm import tqdm
from transformers import pipeline
from pydub import AudioSegment

# Function to get the duration of a WAV file in seconds
def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return round(duration, 2)

# Function to count the number of words in a sentence
def count_words(sentence):
    return len(re.findall(r'\w+', sentence))

# Function to calculate the entropy of a list
def calculate_entropy(lst):
    counts = Counter(lst)
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

# Function to calculate the average word frequency in a sentence
def calculate_avg_word_frequency(word_counts):
    total_words = sum(word_counts.values())
    unique_words = len(word_counts)
    if unique_words > 0:
        return total_words / unique_words
    else:
        return 0

# Function to get the top N most common words from a word frequency counter
def get_top_n_words(word_counts, n=5):
    most_common_words = word_counts.most_common(n)
    return [word for word, _ in most_common_words]

# Function to get the file type (extension) of an audio file
def get_file_type(file_path):
    return os.path.splitext(file_path)[1]

# Function to get the sample rate, bit depth, bit rate, number of channels, and number of audio samples of an audio file
def get_audio_properties(file_path):
    audio = AudioSegment.from_file(file_path)
    sample_rate = audio.frame_rate
    bit_depth = audio.sample_width * 8
    bit_rate = audio.frame_rate * audio.sample_width * audio.channels
    num_channels = audio.channels
    num_samples = len(audio.get_array_of_samples())
    return sample_rate, bit_depth, bit_rate, num_channels, num_samples

# Path to your audio dataset folder
dataset_folder = 'clips'

# Path to the validated.tsv file
validated_file_path = 'validated.tsv'

# Output TSV file path
tsv_file_path = 'analysis.tsv'

# Check if the TSV file already exists
if not os.path.exists(tsv_file_path):
    # Load the sentiment analysis model from Hugging Face Transformers
    sentiment_analysis_model = pipeline("sentiment-analysis")

    # Read the validated file and extract the audio file names and sentences
    audio_data = {}
    with open(validated_file_path, 'r') as validated_file:
        # Skip the header row
        next(validated_file)
        for line in validated_file:
            line_parts = line.strip().split('\t')
            file_name = line_parts[1]  # Extract the 'path' column
            sentence = line_parts[2]  # Extract the 'sentence' column
            audio_data[file_name] = sentence

    # Count the number of audio files
    num_files = len(audio_data)

    # Create a progress bar
    progress_bar = tqdm(total=num_files, desc="Processing")

    # Open the TSV file in write mode to create it
    with open(tsv_file_path, 'w', newline='') as tsv_file:
        # Create a CSV writer
        writer = csv.writer(tsv_file, delimiter='\t')

        # Write the header row
        writer.writerow(['Audio_File', 'Length', 'Words', 'Sentiment', 'Num_Characters', 'Num_Sentences',
                         'Avg_Word_Length', 'Min_Word_Length', 'Max_Word_Length', 'Avg_Sentence_Length',
                         'Min_Sentence_Length', 'Max_Sentence_Length', 'Unique_Words', 'Vocabulary_Size',
                         'Word_Entropy', 'Sentence_Entropy', 'Avg_Word_Frequency', 'Min_Word_Frequency',
                         'Max_Word_Frequency', 'Top_5_Words', 'File_Type', 'Sample_Rate', 'Bit_Depth',
                         'Bit_Rate', 'Num_Channels', 'Num_Samples'])

        # Iterate over the audio files
        for file_name, sentence in audio_data.items():
            file_path = os.path.join(dataset_folder, file_name)
            duration = get_wav_duration(file_path)
            num_words = count_words(sentence)
            sentiment = sentiment_analysis_model(sentence)[0]['label']
            num_characters = len(sentence)
            num_sentences = len(re.findall(r'[.!?]+', sentence))
            words = re.findall(r'\w+', sentence.lower())
            word_counts = Counter(words)
            unique_words = len(word_counts)
            vocabulary_size = len(word_counts)
            word_entropy = calculate_entropy(list(word_counts.values()))
            sentence_entropy = calculate_entropy([num_words])
            avg_word_frequency = calculate_avg_word_frequency(word_counts)
            min_word_frequency = min(word_counts.values())
            max_word_frequency = max(word_counts.values())
            top_5_words = get_top_n_words(word_counts, n=5)
            file_type = get_file_type(file_path)
            sample_rate, bit_depth, bit_rate, num_channels, num_samples = get_audio_properties(file_path)

            avg_sentence_length = num_words / num_sentences if num_sentences != 0 else 0

            writer.writerow([
                file_name, duration, num_words, sentiment, num_characters, num_sentences,
                statistics.mean([len(word) for word in words]), min([len(word) for word in words]),
                max([len(word) for word in words]), avg_sentence_length,
                min([len(sentence) for sentence in re.split(r'[.!?]+', sentence.strip()) if sentence.strip()]),
                max([len(sentence) for sentence in re.split(r'[.!?]+', sentence.strip()) if sentence.strip()]),
                unique_words, vocabulary_size, word_entropy, sentence_entropy, avg_word_frequency,
                min_word_frequency, max_word_frequency, ', '.join(top_5_words),
                file_type, sample_rate, bit_depth, bit_rate, num_channels, num_samples
            ])

            # Update the progress bar
            progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()