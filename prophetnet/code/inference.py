import torch
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LOAD MODEL
# Change this to evaluate different models
MODEL_PATH = os.path.join(BASE_DIR, "../models/prophetnet_rtt_model")  
# try also:
# "../models/prophetnet_rtt_model"
# "../models/prophetnet_baseline_model"

tokenizer = ProphetNetTokenizer.from_pretrained(MODEL_PATH)
model = ProphetNetForConditionalGeneration.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using device:", device)

# INFERENCE FUNCTION
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

# TEST EXAMPLES
examples = [
    """I have had a persistent cough and fever for the past 3 days, which has been getting progressively worse. 
    I also have body aches and fatigue. My temperature has been hovering around 101-102 degrees Fahrenheit. 
    I tried some over-the-counter medications but they don't seem to be helping much. 
    I'm wondering if this could be the flu or if I should be more concerned about something else. 
    Should I see a doctor or just wait it out?""",
    
    """What are the symptoms of diabetes and how can it be managed? I'm asking because my family history shows 
    several relatives with diabetes, and I want to know the early warning signs to watch out for. 
    Also, are there any lifestyle changes or dietary modifications that can help prevent or manage the condition? 
    What are the differences between Type 1 and Type 2 diabetes in terms of symptoms and treatment? 
    Are there any new treatments available besides the traditional insulin therapy?""",
    
    """My mother was recently diagnosed with a genetic disorder and we're unsure about where to get genetic testing done. 
    We want to know if other family members are at risk. We live in a small town and don't have immediate access to 
    specialized medical centers. What are the options for genetic counseling and testing? Can we do this remotely? 
    How much does genetic testing typically cost and is it covered by insurance? 
    What should we prepare or know before undergoing genetic testing?""",
    
    """I've been experiencing chest pain when I breathe deeply, and it's been happening on and off for about a week now. 
    Along with the chest pain, I also feel shortness of breath during physical activity. I don't have a history of heart problems, 
    but I'm quite anxious about it. The pain is usually sharp and localized on the left side of my chest. 
    Sometimes it goes away after a few hours but comes back later. Should I be worried and seek immediate medical attention?""",
]

# RUN
for i, text in enumerate(examples):
    print("\n" + "="*60)
    print(f"INPUT {i+1}:\n{text}")
    
    output = summarize(text)
    
    print(f"\nSUMMARY:\n{output}")