import torch
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = "prophetnet_rtt_model"  
# try also:
# "prophetnet_rtt_model"
# "prophetnet_baseline_model"

tokenizer = ProphetNetTokenizer.from_pretrained(MODEL_PATH)
model = ProphetNetForConditionalGeneration.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using device:", device)

# =========================
# INFERENCE FUNCTION
# =========================

def summarize(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# =========================
# TEST EXAMPLES
# =========================

examples = [
    "I have had a persistent cough and fever for the past 3 days, what could be the cause?",
    "What are the symptoms of diabetes and how can it be managed?",
    "Where can I get genetic testing for rare diseases?",
    "I feel chest pain when breathing deeply, should I be worried?",
    
]

# =========================
# RUN
# =========================

for i, text in enumerate(examples):
    print("\n" + "="*60)
    print(f"INPUT {i+1}:\n{text}")
    
    output = summarize(text)
    
    print(f"\nSUMMARY:\n{output}")