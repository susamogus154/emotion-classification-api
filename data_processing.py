import librosa
import numpy as np
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

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
    audio, sr = librosa.load(io.BytesIO(audio_file))

    resized_audio = resize_audio(audio_array=audio)

    S = librosa.feature.melspectrogram(y=resized_audio, sr=sr, n_mels=128, n_fft=512)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    S_db = S_db.astype(np.uint8)

    img = Image.fromarray(S_db, mode='L').convert("RGB")
    tensor_transform = transforms.Compose([ transforms.ToTensor() ])
    tensor = tensor_transform(img)

    return tensor.unsqueeze(0) # make it look like a batch of size 1
    