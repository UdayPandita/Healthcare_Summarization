import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# =========================
# LOAD DATA
# =========================

original_df = pd.read_csv("train.csv")
aug_df = pd.read_csv("augmented_train.csv")

print("Original:", original_df.shape)
print("Augmented:", aug_df.shape)

# =========================
# LOAD EMBEDDING MODEL
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# COMPUTE SIMILARITY
# =========================

original_questions = original_df["question"].tolist()
aug_questions = aug_df["question"].tolist()

original_emb = model.encode(original_questions, convert_to_tensor=True)
aug_emb = model.encode(aug_questions, convert_to_tensor=True)

# cosine similarity
similarities = util.cos_sim(original_emb, aug_emb).diagonal()

# =========================
# FILTER
# =========================

filtered_questions = []
filtered_summaries = []

for i, sim in enumerate(similarities):
    sim_val = sim.item()

    # keep moderately similar samples
    if 0.7 <= sim_val <= 0.95:
        filtered_questions.append(aug_questions[i])
        filtered_summaries.append(aug_df["summary"][i])

print("Filtered samples:", len(filtered_questions))

# =========================
# SAVE
# =========================

filtered_df = pd.DataFrame({
    "question": filtered_questions,
    "summary": filtered_summaries
})

filtered_df.to_csv("filtered_augmented_train.csv", index=False)

print("Saved filtered dataset")