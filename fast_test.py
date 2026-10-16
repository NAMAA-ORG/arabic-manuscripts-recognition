# pip install -U torch transformers accelerate pillow pandas jiwer qwen-vl-utils

import pandas as pd
import torch
from PIL import Image
from jiwer import cer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

CSV = "dataset_relative.csv"
ROOT = "."  # Dataset root; relative image paths resolve from here
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

df = pd.read_csv(CSV)
processor = AutoProcessor.from_pretrained(MODEL)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype="auto", device_map="auto"
)

predictions, references = [], []

for _, row in df.iterrows():
    image = Image.open(f"{ROOT}/{row['path']}").convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Transcribe the Arabic manuscript exactly. Output only the main text, no explanation."}
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512)

    pred = processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    predictions.append(pred)
    references.append(str(row["main_text"]))
    print(f"\nPRED: {pred}\nREF:  {row['main_text']}")

print(f"\nCER: {cer(references, predictions):.4f}")
df["prediction"] = predictions
df.to_csv("results.csv", index=False, encoding="utf-8-sig")
