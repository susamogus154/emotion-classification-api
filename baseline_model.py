import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, HubertModel


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()
for param in hubert.parameters():
    param.requires_grad = False

def extract_features(audio_array, sr=16000):
    inputs = feature_extractor(audio_array, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = hubert(**inputs)
    # mean pool over time dimension
    return outputs.last_hidden_state.mean(dim=1).cpu()  # shape: [1, 768]


class EmotionHead(nn.Module): # dense dropout layer then final dense
    def __init__(self, input_dim=768, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
    def predict(self, features):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(features) # forward function
            print(outputs.shape) 
            probabilities = torch.nn.functional.softmax(outputs, dim=1) # final softmax activation function
            
        return probabilities.numpy().flatten()