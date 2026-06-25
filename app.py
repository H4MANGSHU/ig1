from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import io

app = FastAPI()

model = joblib.load("xgb_face_rank_model.pkl")
clip_model = SentenceTransformer("clip-ViT-B-32")

@app.get("/")
def home():
    return FileResponse("whom-should-i-date.html")

def get_embedding(file: UploadFile):
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    emb = clip_model.encode(img)
    return np.array(emb).reshape(1, -1)

@app.post("/predict-upload")
def predict_upload(
    photo_a: UploadFile = File(...),
    photo_b: UploadFile = File(...)
):
    fa = get_embedding(photo_a)
    fb = get_embedding(photo_b)

    X = fa - fb

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    return {
        "winner_side": "A" if pred == 1 else "B",
        "winner_file": photo_a.filename if pred == 1 else photo_b.filename,
        "confidence": float(max(prob))
    }
