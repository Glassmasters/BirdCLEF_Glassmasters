import os
import numpy as np
import pandas as pd
import librosa
import torch
from src.audio_to_image_model.model_audio_to_image import load_metadata
from src.preprocess.audio_to_image.preprocess import preprocess_audio


def split_audio(audio_path, segment_length=5, sample_rate=32000):
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = len(audio) // sample_rate
    segments = []

    for start in range(0, duration, segment_length):
        end = start + segment_length
        if end > duration:
            end = duration
        segment = audio[start * sample_rate:end * sample_rate]
        segments.append(segment)

    return segments


def predict(model, audio_segments, preprocess_fn, device):
    predictions = []

    for segment in audio_segments:
        mel_spectrogram = preprocess_fn(segment)
        mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0).to(device)
        output = model(mel_spectrogram)
        pred = torch.sigmoid(output).detach().cpu().numpy()
        predictions.append(pred)

    return np.array(predictions)


def generate_csv(predictions, threshold, row_id_prefix, label_to_bird_map):
    result_rows = []

    for i, pred in enumerate(predictions):
        birds_present = [label_to_bird_map[j] for j, p in enumerate(pred) if p >= threshold]
        row_id = f"{row_id_prefix}_{i * 5}"
        result_rows.append([row_id] + birds_present)

    result_df = pd.DataFrame(result_rows, columns=['row_id'] + [f'bird{i}' for i in range(1, len(label_to_bird_map) + 1)])
    return result_df


def inference(audio_path, model, preprocess_fn, label_to_bird_map, threshold=0.5, device='cuda'):
    audio_segments = split_audio(audio_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        predictions = predict(model, audio_segments, preprocess_fn, device)

    row_id_prefix = os.path.splitext(os.path.basename(audio_path))[0]
    result_df = generate_csv(predictions, threshold, row_id_prefix, label_to_bird_map)

    return result_df


# Load your model and preprocessing function here
model = torch.load("../../models/audio_to_image_model.pt")
preprocess_fn = preprocess_audio
metadata_df, num_classes = load_metadata('path/to/metadata.csv')
label_to_index_map = {label: index for index, label in enumerate(metadata_df['primary_label'].unique())}
label_to_bird_map = {index: label for label, index in label_to_index_map.items()}  # Assuming label_to_index_map is the dictionary mapping bird labels to indices

audio_path = 'path/to/10min_audio.wav'
result_df = inference(audio_path, model, preprocess_fn, label_to_bird_map)

result_df.to_csv('submission.csv', index=False)
