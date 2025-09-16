"""
training.py
-----------
Script to train and evaluate NER models for skill extraction from job descriptions.

This script:
1. Loads the processed dataset (from ETL pipeline).
2. Tokenizes text and aligns BIO labels.
3. Trains transformer models (BERT, DistilBERT, RoBERTa, SpanBERT).
4. Logs hyperparameters, metrics, and artifacts to MLflow.
5. Saves the best model locally for inference.

Usage:
    python model_training.py --model_name bert-base-cased --epochs 2
"""

import os
import json
import mlflow
import mlflow.transformers
import argparse
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Train NER model for Job Skill Extraction")
parser.add_argument("--model_name", type=str, required=True,
                    help="HuggingFace model name (e.g., bert-base-cased, distilbert-base-cased, roberta-base, SpanBERT)")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
args = parser.parse_args()

MODEL_NAME = args.model_name
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

# -------------------------------
# Load Processed Dataset
# -------------------------------
print("Loading processed dataset...")
with open("data/processed/processed_job_data.json", "r") as f:
    raw_data = json.load(f)

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(raw_data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
datasets = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# -------------------------------
# Label Mapping
# -------------------------------
unique_tags = sorted({tag for sample in raw_data for tag in sample["ner_tags"]})
tag2id = {tag: i for i, tag in enumerate(unique_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}

print(f"Labels: {tag2id}")

# -------------------------------
# Tokenizer Setup
# -------------------------------
tokenizer_kwargs = {}
if "roberta" in MODEL_NAME.lower():
    tokenizer_kwargs["add_prefix_space"] = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_kwargs)

def tokenize_and_align_labels(batch):
    tokenized = tokenizer(batch["tokens"], is_split_into_words=True,
                          truncation=True, padding="max_length", max_length=128)
    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(tag2id[label[word_id]])
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# -------------------------------
# Model Setup
# -------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(tag2id), id2label=id2tag, label2id=tag2id
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# -------------------------------
# Metrics Function
# -------------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = [
        [id2tag[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
        "accuracy": accuracy_score(true_labels, true_preds),
    }

# -------------------------------
# Training Setup
# -------------------------------
training_args = TrainingArguments(
    output_dir=f"models/{MODEL_NAME}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to=[]  # disables W&B, only logs to MLflow
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------------------------------
# MLflow Logging
# -------------------------------
mlflow.set_experiment(f"NER__{MODEL_NAME}")

with mlflow.start_run():
    # Log Hyperparameters
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "num_train_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "max_seq_length": 128
    })

    # Train Model
    train_result = trainer.train()

    # Log Metrics
    metrics = trainer.evaluate()
    mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})

    # Save Model + Label Mappings
    trainer.save_model(f"models/{MODEL_NAME}")
    with open(f"models/{MODEL_NAME}/label_mappings.json", "w") as f:
        json.dump({"id2tag": id2tag, "tag2id": tag2id}, f)

    # Log Model to MLflow
    mlflow.transformers.log_model(trainer.model, "ner_model")

print("Training complete!")