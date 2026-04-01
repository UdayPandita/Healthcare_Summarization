import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BartTokenizer

# =========================
# STEP 4: LOAD + TOKENIZE DATA
# =========================

# -------------------------
# Get current script directory
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("CURRENT WORKING DIRECTORY:")
print(BASE_DIR)
print("=" * 80)

# -------------------------
# Load CSV splits
# -------------------------
train_path = os.path.join(BASE_DIR, "train.csv")
val_path = os.path.join(BASE_DIR, "validation.csv")
test_path = os.path.join(BASE_DIR, "test.csv")

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into DatasetDict
datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

print("\nDATASETS LOADED")
print(datasets)

# -------------------------
# Load BART tokenizer
# -------------------------
tokenizer = BartTokenizer.from_pretrained(
    "facebook/bart-large"
)

# -------------------------
# Tokenization function
# -------------------------
def tokenize_function(example):
    # Tokenize input question
    model_input = tokenizer(
        example["question"],
        max_length=256,
        padding="max_length",
        truncation=True
    )

    # Tokenize target summary (new HF style)
    labels = tokenizer(
        text_target=example["summary"],
        max_length=64,
        padding="max_length",
        truncation=True
    )

    model_input["labels"] = labels["input_ids"]
    return model_input

# -------------------------
# Apply tokenization
# -------------------------
tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["question", "summary"]
)

print("\n" + "=" * 80)
print("TOKENIZATION COMPLETE")
print("=" * 80)

print("\nSample tokenized example:")
print(tokenized_datasets["train"][0])

# -------------------------
# Save tokenized dataset
# -------------------------
save_path = os.path.join(BASE_DIR, "tokenized_data_bart")
tokenized_datasets.save_to_disk(save_path)

print(f"\nTokenized dataset saved to: {save_path}")
print("Step 4 complete.")
