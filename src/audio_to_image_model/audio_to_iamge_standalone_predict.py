import glob
import os
import numpy as np
import pandas as pd
import librosa
import torch
from src.audio_to_image_model.model_audio_to_image import load_metadata
from src.preprocess.audio_to_image.preprocess import preprocess_audio
from src.audio_to_image_model.audio_to_image_model_def import *


def split_audio(audio_path, segment_length=5, sample_rate=32000, n_fft=2048):
    mel_spectrogram_full = preprocess_audio(audio_path, inference=True)
    num_frames = mel_spectrogram_full.shape[2]
    duration = num_frames * (n_fft // 2) / sample_rate
    segments = []

    for start in range(0, int(duration), segment_length):
        end = start + segment_length
        if end > duration:
            end = duration
        start_frame = int(start * (sample_rate / (n_fft // 2)))
        end_frame = int(end * (sample_rate / (n_fft // 2)))
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
        max_index = np.argmax(pred[0])
        # birds_present = {label_to_bird_map[index]: 1 if p >= threshold else 0 for index, p in enumerate(pred[0])}
        birds_present = {label_to_bird_map[index]: 1 if index == max_index else 0 for index, p in enumerate(pred[0])}
        row_id = f"{row_id_prefix}_{i * 5 + 5}"
        result_rows.append([row_id] + list(birds_present.values()))

    # Change the column names to bird names
    column_names = ['row_id'] + [label_to_bird_map[i] for i in range(0, len(label_to_bird_map))]
    result_df = pd.DataFrame(result_rows, columns=column_names)
    return result_df


# def generate_csv(predictions, threshold, row_id_prefix, label_to_bird_map):
#    result_rows = []
#    for i, pred in enumerate(predictions):
#        print(f"Prediction {i}: {pred}")
#        max_index = np.argmax(pred[0])
#        birds_present = {label_to_bird_map[index]: pred[0][index] for index, p in enumerate(pred[0])}
#        row_id = f"{row_id_prefix}_{i * 5 + 5}"
#        result_rows.append([row_id] + list(birds_present.values()))
#
#    # Change the column names to bird names
#    column_names = ['row_id'] + [label_to_bird_map[i] for i in range(0, len(label_to_bird_map))]
#    result_df = pd.DataFrame(result_rows, columns=column_names)
#    return result_df


def inference(audio_path, model, label_to_bird_map, threshold=0.5017, device='cpu'):
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
    TEST_FOLDER_PATH = r"D:\kaggle_competition\birdclef-2023\test_soundscapes"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load your model and preprocessing function here
    metadata_df, num_classes = load_metadata(r"C:\Users\phili\Projects\pythonProjects\BirdCLEF_Glassmasters\data\train_metadata_subset_balanced.csv")
    model = PretrainedBirdClassifier(num_classes).to(device)
    model.load_state_dict(torch.load("../../models/model_pretrained.pth", map_location=torch.device(device)))
    model.eval()

    label_to_index_map = {label: index for index, label in enumerate(sorted(metadata_df['primary_label'].unique()))}
    label_to_bird_map = {index: label for label, index in label_to_index_map.items()}
    # Create an empty DataFrame to store concatenated results
    submission_df = pd.DataFrame()

    for audio_path in list(glob.glob(rf"{TEST_FOLDER_PATH}/*.ogg")):
        print(audio_path)

        # Get the result DataFrame for the current audio file
        result_df = inference(audio_path, model, label_to_bird_map, device=device)

        # Concatenate the result_df to the main combined_result_df
        submission_df = pd.concat([submission_df, result_df], axis=0, ignore_index=True)

    submission_df.to_csv('submission.csv', index=False)

    # Load the submission file into pandas
    # submission = pd.read_csv('submission.csv')
    ## count how often the number 1 occurs in each row
    # submission['count'] = submission.iloc[:, 1:].sum(axis=1)
    ## select only the rows where the number 1 occurs more than once
    # submission = submission[submission['count'] > 0]
    # print(submission.head(100))
