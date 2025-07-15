from pathlib import Path

DATA_DIR   = Path("./Memes")
LABEL_CSV  = DATA_DIR / "labels" / "labels.csv"
MODEL_NAME = "ViT-B/32"
TOP_K      = 5
SENTIMENT_OPTIONS = ["Any", "positive", "negative", "neutral"]
