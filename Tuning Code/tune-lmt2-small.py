'''
Model Trainer from HF Fine tune Whisper small Model

Author: Sabian Hibbs
University of Derby
United Kingdom, England

Licence MIT
'''

import os
os.environ["HF_DATASETS_CACHE"] = "E:\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "E:\huggingface"

# Loading checkpoints from hugging face
from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from datasets import Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import evaluate
import subprocess

# Huggingface pylance token
# Make sure you run cmd first and use

#subprocess.run(['huggingface-cli', 'login'])

# huggingface-cli login
# login to hugging face and get a token
# paste the token below
# accept and sign the agreement for 13_0 dataset 

if __name__ == '__main__':

    notebook_login()

    common_voice = DatasetDict()
    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train+validation")
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test")
    # Print for debug
    #print(common_voice)

    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    # Print for debug
    #print(common_voice)

    # --------------------------
    # Feature Extraction Process - small
    # --------------------------

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    # --------------------------
    # Load Whisper Tokenizer - small
    # --------------------------

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")

    # --------------------------
    # Combine WhisperProcessor - small
    # --------------------------

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")

    # --------------------------
    # Prepare Data
    # --------------------------

    print(common_voice["train"][0])
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    # Re-loading the first audio sample in the Common Voice dataset will resample
    print(common_voice["train"][0])

    def prepare_dataset(batch, feature_extractor, tokenizer):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch


    common_voice = common_voice.map(prepare_dataset, fn_kwargs={'feature_extractor': feature_extractor, 'tokenizer': tokenizer}, remove_columns=common_voice.column_names["train"], num_proc=30)

    # --------------------------
    # Define Data Collector
    # --------------------------

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch
        
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # --------------------------
    # Evaluation Metrics
    # --------------------------

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # --------------------------
    # Load Pretrained Checkpoints
    # --------------------------

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # --------------------------
    # Define the Training Config
    # --------------------------

    training_args = Seq2SeqTrainingArguments(
        output_dir="./LMT2-Small-10",  # change to a repo name of your choice
        per_device_train_batch_size=10,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-6,
        warmup_steps=500,
        max_steps=10000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)

    ''' 
    ----------------------------------------------
    ##### Model Training Whisper SMALL Model #####
    ----------------------------------------------
    '''

    trainer.train()

    kwargs = {
        "dataset_tags": "asr, speech-to-text, lmt2, tesseract3d, whisper-small, mozilla, common-voice, en",
        "dataset": "Common Voice 13.0",
        "dataset_args": "config: en, split: test",
        "language": "en",
        "model_name": "Tesseract3D/LMT2-Tuned-S",  
        "finetuned_from": "openai/whisper-small",
        "tasks": "automatic-speech-recognition",
        "tags": "asr, speech-to-text, lmt2, tesseract3d, whisper-small, mozilla, common-voice, en",
    }

    trainer.push_to_hub(**kwargs)

    print("TRAINING COMPLETE!")