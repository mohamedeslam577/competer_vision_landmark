# build_embeddings.py

import pandas as pd
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("clip-ViT-B-32")

print("Reading CSV...")

df = pd.read_csv("landmark_prompts.csv")

landmarks = df["landmark_prompt"].tolist()

print(f"Creating embeddings for {len(landmarks)} landmarks...")

text_embeddings = model.encode(
    landmarks,
    convert_to_numpy=True,
    show_progress_bar=True
)

np.save(
    "landmark_embeddings.npy",
    text_embeddings
)

with open(
    "landmark_names.pkl",
    "wb"
) as f:
    pickle.dump(
        landmarks,
        f
    )

print("Done!")