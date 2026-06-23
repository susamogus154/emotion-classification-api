from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import io
from baseline_model import EmotionHead, extract_features
import data_processing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model once on startup
model = EmotionHead(num_classes=7)
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

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

    processed_audio_bytes = data_processing.reformat_audio(audio_bytes)
    audio, _ = librosa.load(io.BytesIO(processed_audio_bytes))

    features = extract_features(audio, sr=16000)
    probs = model.predict(features)
    
    return {
        "raw audio shape": audio_bytes.__len__(),
        "features shape": features.shape,
        "emotions": {label_dict[i]: float(probs[i]) for i in range(7)}
    }