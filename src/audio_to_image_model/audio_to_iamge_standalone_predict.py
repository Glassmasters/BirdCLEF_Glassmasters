import os
import numpy as np
import pandas as pd
import librosa
import torch
from src.audio_to_image_model.model_audio_to_image import load_metadata
from src.preprocess.audio_to_image.preprocess import preprocess_audio
from src.audio_to_image_model.audio_to_image_model_def import CustomCNN


def split_audio(audio_path, segment_length=5, sample_rate=32000, n_fft=2048):
    mel_spectrogram_full = preprocess_audio(audio_path, inference=True)
    num_frames = mel_spectrogram_full.shape[2]
    duration = num_frames * (n_fft // 2) / sample_rate
    segments = []

    for start in range(0, int(duration), segment_length):
        end = start + segment_length
        if end > duration:
            end = duration
        start_frame = int(start * (sample_rate / (n_fft // 2) ))
        end_frame = int(end * (sample_rate / (n_fft // 2) ))
        segment = mel_spectrogram_full[..., start_frame:end_frame]
        segments.append(segment)

    return segments



def predict(model, audio_segments, device):
    predictions = []
    model.eval()
    for segment in audio_segments:
        segment_tensor = torch.tensor(segment).unsqueeze(0).to(device)
        output = model(segment_tensor)
        pred = torch.sigmoid(output).detach().cpu().numpy()
        predictions.append(pred)

    return np.array(predictions)


def generate_csv(predictions, threshold, row_id_prefix, label_to_bird_map):
    result_rows = []

    for i, pred in enumerate(predictions):
        print(f"Prediction {i}: {pred}")
        birds_present = [1 if p >= threshold else 0 for p in pred[0]]
        row_id = f"{row_id_prefix}_{i * 5}"
        result_rows.append([row_id] + birds_present)

    result_df = pd.DataFrame(result_rows, columns=['row_id'] + [f'bird{i}' for i in range(1, len(label_to_bird_map) + 1)])
    return result_df



def inference(audio_path, model, label_to_bird_map, threshold=0.6, device='cpu'):
    audio_segments = split_audio(audio_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        predictions = predict(model, audio_segments, device)

    row_id_prefix = os.path.splitext(os.path.basename(audio_path))[0]
    result_df = generate_csv(predictions, threshold, row_id_prefix, label_to_bird_map)

    return result_df


if __name__ == '__main__':
    # Set parameters here
    TEST_FILE_PATH = r"D:\kaggle_competition\birdclef-2023\test_soundscapes"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load your model and preprocessing function here
    preprocess_fn = preprocess_audio
    metadata_df, num_classes = load_metadata(r"../../data/local_subset.csv")
    model = CustomCNN(num_classes).to(device)
    model.load_state_dict(torch.load("../../models/model.pth"))

    label_to_index_map = {label: index for index, label in enumerate(metadata_df['primary_label'].unique())}
    label_to_bird_map = {index: label for label, index in label_to_index_map.items()}  # Assuming label_to_index_map is the dictionary mapping bird labels to indices

    audio_path = TEST_FILE_PATH + r"\soundscape_29201.ogg"
    print(audio_path)
    result_df = inference(audio_path, model, label_to_bird_map, device=device)

    result_df.to_csv('submission.csv', index=False)
