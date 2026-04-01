import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BartTokenizer

# =========================
# LOAD RTT DATA
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("LOADING RTT DATA FOR BART TOKENIZATION")
print("=" * 80)

train_path = os.path.join(BASE_DIR, "../../filtered_augmented_train.csv")
val_path = os.path.join(BASE_DIR, "../../validation.csv")
test_path = os.path.join(BASE_DIR, "../../test.csv")

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

print("\nTrain (RTT):", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

# =========================
# TO HF DATASET
# =========================

datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df),
})

print("\n" + "=" * 80)
print("DATASETS LOADED")
print("=" * 80)
print(datasets)

# =========================
# BART TOKENIZER
# =========================

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# =========================
# TOKENIZATION FUNCTION
# =========================

def tokenize_function(example):
    model_input = tokenizer(
        example["question"],
        max_length=256,
        padding="max_length",
        truncation=True
    )

    labels = tokenizer(
        text_target=example["summary"],
        max_length=64,
        padding="max_length",
        truncation=True
    )

    model_input["labels"] = labels["input_ids"]
    return model_input

# =========================
# APPLY TOKENIZATION
# =========================

print("\nTokenizing RTT data...")
tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["question", "summary"]
)

print("\n" + "=" * 80)
print("TOKENIZATION COMPLETE")
print("=" * 80)
print(tokenized_datasets)

# -------------------------
# SAVE TOKENIZED DATASET
# -------------------------

save_path = os.path.join(BASE_DIR, "../data/tokenized_final_data_bart")
tokenized_datasets.save_to_disk(save_path)

print(f"\nTokenized RTT dataset saved to: {save_path}")
print("RTT tokenization complete.")
