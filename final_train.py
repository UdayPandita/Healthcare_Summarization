import os
import torch
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
# LOAD TOKENIZED DATA
# =========================
BASE_DIR = os.getcwd()
data_path = os.path.join(BASE_DIR, "tokenized_final_v2")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("CUDA available:", torch.cuda.is_available())
print("Model device:", model.device)

# =========================
# DATA COLLATOR
# =========================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# =========================
# ROUGE METRICS
# =========================
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    r1, r2, rl = [], [], []
    for p, l in zip(decoded_preds, decoded_labels):
        s = scorer.score(l, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(r1),
        "rouge2": np.mean(r2),
        "rougeL": np.mean(rl),
    }

# =========================
# TRAINING ARGS (SAFE)
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir="prophetnet_diversity",

    do_train=True,
    do_eval=True,

    eval_strategy="epoch",     # your env-safe name
    save_strategy="epoch",

    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,

    weight_decay=0.01,
    predict_with_generate=True,

    logging_dir="./logs_div",
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
    processing_class=tokenizer,   # IMPORTANT for your version
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# =========================
# TRAIN
# =========================
print("\nStarting Diversity training...\n")
trainer.train()

# =========================
# SAVE FINAL MODEL
# =========================
trainer.save_model("prophetnet_diversity_model")

print("\nTraining complete. Model saved.")