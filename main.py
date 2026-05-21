"""
// record audio and send
const formData = new FormData()
formData.append("file", audioBlob, "recording.wav")

const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    body: formData  // no Content-Type header — browser sets it automatically
})

const emotions = await response.json()
// emotions = { "Angry": 0.12, "Happy": 0.45, ... }
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import io
from baseline_model import EmotionClassifier
from data_processing import preprocess_image, resize_audio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model once on startup
model = EmotionClassifier(num_classes=7)
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

transform = transforms.ToTensor()

label_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

@app.get("/")
def root():
    return {"Hello": "World"}

@app.get("/labels")
def get_labels():
    return label_dict

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    processed_audio = preprocess_image(audio_bytes)
    probs = model.predict(processed_audio)
    
    return {label_dict[i]: float(probs[i]) for i in range(7)}