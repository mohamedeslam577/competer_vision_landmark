# main.py
from dotenv import load_dotenv
load_dotenv()

import os
import io
import numpy as np
import pickle
import httpx

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from groq import Groq



# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cometer Vision API",
    description="Landmark detection powered by CLIP + Groq enrichment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load CLIP model & embeddings on startup ────────────────────────────────────
print("Loading CLIP model...")
clip_model = SentenceTransformer("clip-ViT-B-32")

print("Loading landmark embeddings...")
landmark_embeddings = np.load("landmark_embeddings.npy")

with open("landmark_names.pkl", "rb") as f:
    landmark_names = pickle.load(f)

print(f"Ready — {len(landmark_names)} landmarks loaded.")

# ── Groq client ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

CONFIDENCE_THRESHOLD = 0.25


# ── Helper: ask Groq for landmark info ────────────────────────────────────────
def get_landmark_info(landmark_name: str) -> dict:
    """
    Given a landmark name, ask Groq to return:
    - history
    - best_visiting_time
    - travel_tips
    as a structured JSON response.
    """
    prompt = f"""
You are a knowledgeable travel guide specializing in Egyptian landmarks and attractions.
The user just identified the following landmark from a photo:

"{landmark_name}"

Return a JSON object with exactly these three keys (no extra keys, no markdown, pure JSON):
{{
  "history": "2-3 sentences about the history and significance of this place.",
  "best_visiting_time": "1-2 sentences on the best time of day / year to visit.",
  "travel_tips": "2-3 practical tips for visitors (entry fees, dress code, nearby spots, etc.)."
}}
"""

    chat = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    import json
    raw = chat.choices[0].message.content
    return json.loads(raw)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Cometer Vision API is running. POST an image to /detect",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "landmarks_loaded": len(landmark_names)}


@app.post("/detect")
async def detect_landmark(file: UploadFile = File(...)):
    """
    Upload an image and receive:
    - landmark name
    - confidence score
    - history
    - best visiting time
    - travel tips
    """
    # ── Validate file type ─────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WEBP.",
        )

    # ── Read & open image ──────────────────────────────────────────────────────
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    # ── CLIP embedding ─────────────────────────────────────────────────────────
    image_embedding = clip_model.encode(image, convert_to_tensor=True)
    scores = util.cos_sim(image_embedding, landmark_embeddings)
    scores = scores.cpu().numpy()[0]

    best_index = int(np.argmax(scores))
    confidence = float(scores[best_index])
    raw_landmark_name = landmark_names[best_index]

    # Strip the "A photo of … in … Egypt" wrapper to get a clean name
    # e.g. "A photo of Great Sphinx of Giza (تمثال أبو الهول) in Cairo Egypt"
    #   →  "Great Sphinx of Giza (تمثال أبو الهول)"
    clean_name = raw_landmark_name
    if raw_landmark_name.startswith("A photo of "):
        clean_name = raw_landmark_name[len("A photo of "):]
        if " in " in clean_name:
            clean_name, location = clean_name.rsplit(" in ", 1)
        else:
            location = "Egypt"
    else:
        location = "Egypt"

    # ── Low confidence → unknown landmark ────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "landmark": None,
            "confidence": round(confidence, 4),
            "history": None,
            "best_visiting_time": None,
            "travel_tips": None,
            "message": "Landmark not recognized. Please try a clearer photo.",
        }

    # ── Groq enrichment ───────────────────────────────────────────────────────
    try:
        info = get_landmark_info(f"{clean_name}, {location}")
    except Exception as e:
        # Still return detection result even if Groq fails
        info = {
            "history": None,
            "best_visiting_time": None,
            "travel_tips": f"Could not fetch info: {e}",
        }

    return {
        "landmark": clean_name,
        "location": location,
        "confidence": round(confidence, 4),
        "history": info.get("history"),
        "best_visiting_time": info.get("best_visiting_time"),
        "travel_tips": info.get("travel_tips"),
    }
