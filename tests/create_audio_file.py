import pandas as pd
from pydub import AudioSegment
import os


def load_metadata(file_path):
    """
    Load metadata from a CSV file and return a DataFrame and the number of unique classes.
    Args:
        file_path (str): The path to the CSV file containing metadata.
    Returns:
        df (pd.DataFrame): A DataFrame containing the metadata.
        num_classes (int): The number of unique classes in the primary_label column.
    """

    df = pd.read_csv(file_path)
    num_classes = df['primary_label'].nunique()
    return df, num_classes


def create_mixed_audio(metadata, unique_classes, mix_length=600, segment_length=5, audio_id=1):
    """
    Create a mixed audio file and corresponding DataFrame with one-hot encoded class labels.
    Args:
        metadata (pd.DataFrame): A DataFrame containing metadata for audio samples.
        unique_classes (list): A list of unique audio classes.
        mix_length (int, optional): The length of the mixed audio file in seconds. Defaults to 600.
        segment_length (int, optional): The length of each audio segment in seconds. Defaults to 5.
        audio_id (int, optional): The ID to be used for the mixed audio file. Defaults to 1.
    Returns:
        mix_audio (AudioSegment): The mixed audio file as an AudioSegment object.
        mix_dataframe (pd.DataFrame): A DataFrame with one-hot encoded class labels for each audio segment.
    """
    mix_audio = AudioSegment.silent(duration=mix_length * 1000)
    mix_data = []

    for start_time in range(0, mix_length, segment_length):
        sample = metadata.sample(1).iloc[0]
        audio_file = sample['filename']
        audio_file = os.path.join(DATASET_BASE_FILE_PATH, audio_file)
        audio_class = sample['primary_label']

        audio = AudioSegment.from_file(audio_file)
        audio = audio[:segment_length * 1000]
        mix_audio = mix_audio.overlay(audio, position=start_time * 1000)

        one_hot_encoded = [int(cls == audio_class) for cls in unique_classes]
        mix_data.append(
            {'row_id': f'{audio_id +1}_10_minute_mixed_audio_{start_time + segment_length:02}', **dict(zip(unique_classes, one_hot_encoded))}
        )

    mix_dataframe = pd.DataFrame(mix_data)
    return mix_audio, mix_dataframe


def create_multiple_mixed_audios(metadata, unique_classes, n, mix_length=600, segment_length=5):
    """
    Create multiple mixed audio files and their corresponding metadata DataFrames.
    Args:
        metadata (pd.DataFrame): A DataFrame containing metadata for audio samples.
        unique_classes (list): A list of unique audio classes.
        n (int): The number of mixed audio files to create.
        mix_length (int, optional): The length of the mixed audio file in seconds. Defaults to 600.
        segment_length (int, optional): The length of each audio segment in seconds. Defaults to 5.
    Creates:
        Mixed audio files in 'test_audio_files' directory.
        CSV files with one-hot encoded class labels in 'test_audio_metadata' directory.
    """
    if not os.path.exists('test_audio_files'):
        os.mkdir('test_audio_files')
    if not os.path.exists('test_audio_metadata'):
        os.mkdir('test_audio_metadata')

    full_dataframe = pd.DataFrame()
    for i in range(n):
        mixed_audio, mixed_dataframe = create_mixed_audio(metadata, unique_classes, mix_length, segment_length, i)

        output_audio_file = f'test_audio_files/{i + 1}_mixed_audio.ogg'
        output_dataframe_file = f'test_audio_metadata/{i + 1}_mixed_audio_classes.csv'
        mixed_audio.export(output_audio_file, format='ogg')
        mixed_dataframe.to_csv(output_dataframe_file, index=False)
        full_dataframe = pd.concat([full_dataframe, mixed_dataframe], axis=0, ignore_index=True)
        print(f'Created {i + 1} mixed audio files')
    output_dataframe_file = f'test_audio_metadata/all_mixed_audio_classes.csv'
    full_dataframe.to_csv(output_dataframe_file, index=False)


if __name__ == '__main__':
    # Define the dataset base path and the train set file directory
    DATASET_BASE_FILE_PATH = r"D:\kaggle_competition\birdclef-2023\train_audio"
    metadata, num_classes = load_metadata(r"../data/local_subset_stratified.csv")

    unique_classes = sorted(metadata['primary_label'].unique())

    number_of_samples = 20
    create_multiple_mixed_audios(metadata, unique_classes, number_of_samples)

