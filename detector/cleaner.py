import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1) Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
model     = AutoModelForSeq2SeqLM.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def restore_punctuation(text: str, max_length: int = 512) -> str:
    # encode + move to device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    # generate output ids
    outputs = model.generate(**inputs)
    # decode back to string
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# usage:
punctuated = restore_punctuation("so in this position we want to control the center and maybe play e4 then knight f3")
print(punctuated)
