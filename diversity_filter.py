import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# =========================
# LOAD DATA
# =========================

original_df = pd.read_csv("train.csv")
aug_df = pd.read_csv("augmented_train.csv")

# =========================
# MODEL
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

original_q = original_df["question"].tolist()
aug_q = aug_df["question"].tolist()

original_emb = model.encode(original_q, convert_to_tensor=True)
aug_emb = model.encode(aug_q, convert_to_tensor=True)

# =========================
# STEP 1 — similarity filter
# =========================

sim_scores = util.cos_sim(original_emb, aug_emb).diagonal()

filtered_idx = []

for i, s in enumerate(sim_scores):
    val = s.item()
    if 0.7 <= val <= 0.95:
        filtered_idx.append(i)

# =========================
# STEP 2 — diversity filter
# =========================

selected = []
selected_embs = []

for idx in filtered_idx:
    emb = aug_emb[idx]

    if len(selected_embs) == 0:
        selected.append(idx)
        selected_embs.append(emb)
        continue

    # check similarity with already selected samples
    sims = util.cos_sim(emb, torch.stack(selected_embs))

    max_sim = sims.max().item()

    # keep only if not too similar
    if max_sim < 0.85:
        selected.append(idx)
        selected_embs.append(emb)

# =========================
# SAVE
# =========================

final_questions = [aug_q[i] for i in selected]
final_summaries = [aug_df["summary"][i] for i in selected]

final_df = pd.DataFrame({
    "question": final_questions,
    "summary": final_summaries
})

print("Final filtered size:", final_df.shape)

final_df.to_csv("filtered_diverse_train.csv", index=False)

print("Saved filtered_diverse_train.csv")