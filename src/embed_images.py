# embed_images.py

import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import pickle
from pathlib import Path
import pandas as pd

# Paths
DATA_DIR = Path("../Memes")
CSV_PATH = DATA_DIR / "labels" / "labels.csv"
OUTPUT_PATH = DATA_DIR / "embeddings.pkl"

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Read CSV labels
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()  # Clean column names

print("CSV columns:", df.columns.tolist())

# Define the correct column names from your CSV
filename_col = "image_name"
sentiment_col = "overall_sentiment"

# Store embeddings and metadata
embeddings = []
filenames = []
sentiments = []

# Loop through all rows
for _, row in tqdm(df.iterrows(), total=len(df), desc="🔄 Embedding images"):
    filename = row.get(filename_col)
    if not filename:
        print("⚠️ Skipping row without filename:", row)
        continue

    sentiment = row.get(sentiment_col, "Any")
    image_path = DATA_DIR / filename

    if not image_path.exists():
        print(f"⚠️ Image not found: {image_path}")
        continue

    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).cpu().squeeze(0)
        embeddings.append(embedding)
        filenames.append(filename)
        sentiments.append(sentiment)
    except Exception as e:
        print(f"⚠️ Skipping {filename}: {e}")

if len(embeddings) == 0:
    print("❌ No embeddings saved. Check your image files and CSV.")
else:
    # Save to pickle
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "embeddings": torch.stack(embeddings),
            "filenames": filenames,
            "sentiments": sentiments
        }, f)
    print(f"✅ Saved {len(embeddings)} embeddings to {OUTPUT_PATH}")
