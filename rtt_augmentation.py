import pandas as pd
import os
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# =========================
# LOAD TRAIN DATA
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "train.csv")

df = pd.read_csv(train_path)

print("=" * 80)
print("TRAIN DATA LOADED:", df.shape)
print("=" * 80)

# =========================
# LOAD TRANSLATION MODELS
# =========================

# English → Spanish
en_es_model_name = "Helsinki-NLP/opus-mt-en-es"
en_es_tokenizer = MarianTokenizer.from_pretrained(en_es_model_name)
en_es_model = MarianMTModel.from_pretrained(en_es_model_name)

# Spanish → English
es_en_model_name = "Helsinki-NLP/opus-mt-es-en"
es_en_tokenizer = MarianTokenizer.from_pretrained(es_en_model_name)
es_en_model = MarianMTModel.from_pretrained(es_en_model_name)

print("\nTranslation models loaded.")

# =========================
# TRANSLATION FUNCTIONS
# =========================

def translate(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# =========================
# APPLY RTT
# =========================

augmented_questions = []

print("\nStarting RTT augmentation...\n")

batch_size = 8  # keep small for CPU safety

questions = df["question"].tolist()

for i in tqdm(range(0, len(questions), batch_size)):
    batch = questions[i:i + batch_size]

    # Step 1: English → Spanish
    spanish = translate(batch, en_es_model, en_es_tokenizer)

    # Step 2: Spanish → English
    back_to_english = translate(spanish, es_en_model, es_en_tokenizer)

    augmented_questions.extend(back_to_english)

# =========================
# CREATE AUGMENTED DATASET
# =========================

aug_df = pd.DataFrame({
    "question": augmented_questions,
    "summary": df["summary"]
})

print("\nAugmentation complete.")
print("Augmented dataset shape:", aug_df.shape)

print("\nSample augmented data:")
print(aug_df.head())

# =========================
# SAVE AUGMENTED DATA
# =========================

save_path = os.path.join(BASE_DIR, "augmented_train.csv")
aug_df.to_csv(save_path, index=False)

print(f"\nSaved augmented data to: {save_path}")
print("Step 5 complete.")