import librosa
import numpy as np
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import subprocess
matplotlib.use('Agg')

def reformat_audio(audio_file):
    result = subprocess.run(
        ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-ar', '22050', 'pipe:1'],
        input=audio_file,
        capture_output=True
    )
    return result.stdout

def resize_audio(audio_array, target_length=40000):
    current_length = len(audio_array)

    if current_length < target_length:
        # Pad with zeros at the end
        pad_amount = target_length - current_length
        padded_audio = np.pad(audio_array, (0, pad_amount), mode='constant')

        return padded_audio

    elif current_length > target_length:
        # Slice to the target length
        sliced_audio = audio_array[:target_length]
        return sliced_audio

    return audio_array


def preprocess_image(audio_file):
    audio_file = reformat_audio(audio_file)

    audio, sr = librosa.load(io.BytesIO(audio_file))
    resized_audio = resize_audio(audio_array=audio)
    
    S = librosa.feature.melspectrogram(y=resized_audio, sr=sr, n_mels=128, n_fft=512)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # match training exactly — save with matplotlib then reload
    buf = io.BytesIO()
    plt.imsave(buf, S_db, format='png')
    buf.seek(0)
    
    img = Image.open(buf).convert("RGB")
    tensor = transforms.ToTensor()(img)
    return tensor.unsqueeze(0)

