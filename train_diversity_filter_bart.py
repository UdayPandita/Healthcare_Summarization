import torch
from tqdm import tqdm
import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
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
data_path = os.path.join(BASE_DIR, "tokenized_final_v2_bart")   # Diversity filtered dataset (BART tokenized)

datasets = load_from_disk(data_path)

print("=" * 80)
print("DATA LOADED (DIVERSITY - BART)")
print("=" * 80)
print(datasets)

# =========================
# MODEL + TOKENIZER
# =========================

model_name = "facebook/bart-large"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

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

    # Replace -100 with pad token for predictions
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 with pad token for labels
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
# TRAINING ARGS (IDENTICAL TO PROPHETNET DIVERSITY)
# =========================

training_args = Seq2SeqTrainingArguments(
    output_dir="bart_diversity",

    do_train=True,
    do_eval=True,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,

    weight_decay=0.01,
    predict_with_generate=True,

    logging_dir="./logs_diversity_bart",
    logging_steps=50,
    save_total_limit=1,

    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    report_to=["tensorboard"]
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

print("\nStarting diversity training (BART)...\n")
trainer.train()

# =========================
# SAVE MODEL
# =========================

trainer.save_model("bart_diversity_model")

print("\nDiversity training complete (BART).")
