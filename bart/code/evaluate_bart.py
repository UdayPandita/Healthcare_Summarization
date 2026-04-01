import numpy as np
import json
from datasets import load_from_disk
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# CHANGE THIS EACH TIME
# =========================

MODEL_PATH = os.path.join(BASE_DIR, "../models/bart_diversity_model")
# change to:
# "../models/bart_baseline_model"
# "../models/bart_rtt_model"
# "../models/bart_diversity_model"

TOKENIZED_DATA_PATH = os.path.join(BASE_DIR, "../data/tokenized_data_bart")
# change to:
# "../data/tokenized_data_bart"
# "../data/tokenized_final_data_bart"
# "../data/tokenized_final_v2_bart"

# =========================
# LOAD DATA
# =========================

datasets = load_from_disk(TOKENIZED_DATA_PATH)

# =========================
# LOAD MODEL
# =========================

tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)

# =========================
# DATA COLLATOR
# =========================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# =========================
# METRICS
# =========================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
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
# EVAL SETTINGS
# =========================

args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(BASE_DIR, "../models/eval_tmp_bart"),
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    do_train=False,
    do_eval=True
)

# =========================
# TRAINER
# =========================

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    eval_dataset=datasets["test"],   # IMPORTANT
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# =========================
# RUN EVAL
# =========================

metrics = trainer.evaluate()

print(metrics)

# =========================
# SAVE RESULTS
# =========================

filename = MODEL_PATH + "_results.json"

with open(filename, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Saved {filename}")
