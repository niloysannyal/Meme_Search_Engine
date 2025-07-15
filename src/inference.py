# src/inference.py

import torch
import pandas as pd
from PIL import Image, ImageFile
import torch.nn.functional as F
import clip
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MemeSearch:
    def __init__(self, meme_paths, model_name, label_csv=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        self.meme_paths = meme_paths
        self.labels = {}
        self.sentiments = {}

        if label_csv:
            df = pd.read_csv(label_csv)
            df = df.dropna(subset=['image_name', 'text_corrected', 'overall_sentiment'])
            self.labels = dict(zip(df['image_name'], df['text_corrected']))
            self.sentiments = dict(zip(df['image_name'], df['overall_sentiment']))

        self.embeddings, self.image_names = self._embed_images(meme_paths)

    def _embed_images(self, paths):
        image_embeds = []
        image_names = []

        for path in tqdm(paths, desc="🔄 Embedding images"):
            try:
                image = Image.open(path).convert("RGB")
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    image_feat = self.model.encode_image(image_input)

                name = path.name
                # Optional label fusion
                if name in self.labels:
                    label = self.labels[name]
                    text_input = clip.tokenize([label]).to(self.device)
                    with torch.no_grad():
                        text_feat = self.model.encode_text(text_input)
                    combined_feat = 0.7 * image_feat + 0.3 * text_feat
                else:
                    combined_feat = image_feat

                combined_feat /= combined_feat.norm(dim=-1, keepdim=True)
                image_embeds.append(combined_feat)
                image_names.append(name)

            except Exception as e:
                print(f"Error processing {path.name}: {e}")
                continue

        return torch.cat(image_embeds), image_names

    def search(self, query, top_k=5, sentiment_filter="Any"):
        # Encode query text
        text_input = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(text_input)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # Compute cosine similarities
        results = []
        for i, img_emb in enumerate(self.embeddings):
            name = self.image_names[i]

            # Filter sentiment
            if sentiment_filter != "Any":
                if self.sentiments.get(name, "").lower() != sentiment_filter.lower():
                    continue

            sim = F.cosine_similarity(text_feat.cpu(), img_emb.unsqueeze(0).cpu(), dim=1)
            results.append((name, sim.item()))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def search_by_image(self, uploaded_image, top_k=5):
        try:
            image = uploaded_image.convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                query_feat = self.model.encode_image(image_input)
            query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)

            results = []
            for i, img_emb in enumerate(self.embeddings):
                sim = F.cosine_similarity(query_feat.cpu(), img_emb.unsqueeze(0).cpu(), dim=1)
                results.append((self.image_names[i], sim.item()))

            return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

        except Exception as e:
            print(f"Error processing uploaded image: {e}")
            return []
