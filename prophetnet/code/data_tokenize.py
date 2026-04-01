import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import ProphetNetTokenizer

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(BASE_DIR, "../../final_train_v2.csv")
val_path   = os.path.join(BASE_DIR, "../../validation.csv")
test_path  = os.path.join(BASE_DIR, "../../test.csv")

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)
test_df  = pd.read_csv(test_path)

print("Train:", train_df.shape)
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

# =========================
# TOKENIZER
# =========================
model_name = "microsoft/prophetnet-large-uncased"
tokenizer = ProphetNetTokenizer.from_pretrained(model_name)

# =========================
# TOKENIZE FUNCTION
# =========================
def tokenize_function(batch):
    inputs = tokenizer(
        batch["question"],
        max_length=256,
        padding="max_length",
        truncation=True
    )
    targets = tokenizer(
        text_target=batch["summary"],
        max_length=64,
        padding="max_length",
        truncation=True
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

# =========================
# APPLY
# =========================
tokenized = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["question", "summary"]
)

print("\nTokenization complete")
print(tokenized)

# =========================
# SAVE
# =========================
save_path = os.path.join(BASE_DIR, "../data/tokenized_final_v2")
tokenized.save_to_disk(save_path)

print("Saved to:", save_path)