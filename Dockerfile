FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "from transformers import Wav2Vec2FeatureExtractor, HubertModel; \
Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960'); \
HubertModel.from_pretrained('facebook/hubert-base-ls960')"

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]