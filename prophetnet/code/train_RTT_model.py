import torch
import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    ProphetNetForConditionalGeneration,
    ProphetNetTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from rouge_score import rouge_scorer

# =========================
# GPU CHECK
# =========================
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# =========================
# LOAD DATA
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "../data/tokenized_final_data")   # IMPORTANT (RTT dataset)

datasets = load_from_disk(data_path)

print("=" * 80)
print("DATA LOADED")
print("=" * 80)
print(datasets)

# =========================
# MODEL + TOKENIZER
# =========================

model_name = "microsoft/prophetnet-large-uncased"

tokenizer = ProphetNetTokenizer.from_pretrained(model_name)
model = ProphetNetForConditionalGeneration.from_pretrained(model_name)

# move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model device:", model.device)

# =========================
# DATA COLLATOR
# =========================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# =========================
# ROUGE METRIC
# =========================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, label in zip(decoded_preds, decoded_labels):
        result = scorer.score(label, pred)

        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(scores["rouge1"]),
        "rouge2": np.mean(scores["rouge2"]),
        "rougeL": np.mean(scores["rougeL"]),
    }

# =========================
# TRAINING ARGS (FIXED)
# =========================

training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(BASE_DIR, "../models/prophetnet_rtt"),

    do_train=True,
    do_eval=True,

    eval_strategy="epoch",   # FIXED (instead of evaluation_strategy)
    save_strategy="epoch",

    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,

    weight_decay=0.01,
    predict_with_generate=True,

    logging_dir=os.path.join(BASE_DIR, "logs_rtt"),
    save_total_limit=1,

    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True
)

# =========================
# TRAINER
# =========================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2)
    ]
)

# =========================
# TRAIN
# =========================

print("\nStarting RTT training...\n")
trainer.train()

# =========================
# SAVE MODEL
# =========================

trainer.save_model(os.path.join(BASE_DIR, "../models/prophetnet_rtt_model"))

print("\nTraining complete.")