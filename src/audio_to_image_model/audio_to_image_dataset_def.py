# This file defines the dataset structure to be used by the audio to image model
import random
from torch.utils.data import Dataset
from src.preprocess.audio_to_image.preprocess import preprocess_audio
import torch

torch.manual_seed(42)


# Audio Dataset
def pad_or_truncate(mel_spectrogram, length):
    """
   Pad or truncate the mel spectrogram to a fixed length
   Args:
       mel_spectrogram (torch.Tensor): The mel spectrogram tensor.
       length (int): The fixed length to pad or truncate the mel spectrogram to
   Returns:
       torch.Tensor: The padded or truncated mel spectrogram.
   """
    mel_spectrogram = torch.squeeze(mel_spectrogram)
    if mel_spectrogram.shape[1] < length:
        padding = length - mel_spectrogram.shape[1]
        mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding), 'constant', 0)
    else:
        mel_spectrogram = mel_spectrogram[:, :length]
    mel_spectrogram = torch.unsqueeze(mel_spectrogram, 0)
    return mel_spectrogram


def split_audio(mel_spectrogram_full, segment_length=5, sample_rate=32000, n_fft=2048):
    num_frames = mel_spectrogram_full.shape[2]
    duration = num_frames * (n_fft // 2) / sample_rate  # Doc correct?
    segments = []

    if int(duration) == 0:
        return mel_spectrogram_full

    for start in range(0, int(duration), segment_length):
        end = start + segment_length
        if end > duration:
            end = duration
        start_frame = int(start * (sample_rate / (n_fft // 2)))
        end_frame = int(end * (sample_rate / (n_fft // 2)))
        segment = mel_spectrogram_full[..., start_frame:end_frame]
        segments.append(segment)

    segment = random.choice(segments)

    return segment


class BirdDataset(Dataset):
    """
        A custom dataset class for loading and processing bird sound data.
    """

    def __init__(self, metadata_df, train_set_file_dir, fixed_length=5, num_classes=264, transform=None, efficientnet=False):
        """
        Initialize the dataset
        Args:
            metadata_df (pd.DataFrame): A dataframe containing metadata for the audio files.
            train_set_file_dir (str): The base directory for the training set files.
            fixed_length (int, optional): The fixed length to pad or truncate the mel spectrograms to. Defaults to 5 (seconds).
            num_classes (int, optional): The number of bird species. Defaults to 264.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.sample_rate = 32_000
        self.n_fft = 2048
        self.metadata_df = metadata_df
        self.transform = transform
        self.fixed_length = int(fixed_length * (self.sample_rate / (self.n_fft // 2)))
        self.num_classes = num_classes
        self.base_file_path = train_set_file_dir
        self.efficientnet = efficientnet
        self.label_to_index = {label: index for index, label in enumerate(sorted(self.metadata_df['primary_label'].unique()))}

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.metadata_df)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index
        Args:
            idx (int): Index of the sample to get
        Returns:
            tuple: A tuple containing the mel spectrogram and one-hot encoded label.
        """
        row = self.metadata_df.iloc[idx]
        file_path = row['filename']
        primary_label = row['primary_label']

        # Load and preprocess the audio
        full_file_path = self.base_file_path + "\\" + file_path
        mel_spectrogram = preprocess_audio(full_file_path, inference=True)
        # mel_spectrogram = torch.unsqueeze(torch.tensor(mel_spectrogram), 0)

        # Cut the mel spectrogram to pieces and return a random piece
        mel_spectrogram = split_audio(mel_spectrogram)
        # Pad or truncate the mel spectrogram to a fixed length
        mel_spectrogram = pad_or_truncate(mel_spectrogram, self.fixed_length)

        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        one_hot_label = self.label_to_onehot(primary_label, self.num_classes)

        # from src.plotting.plot_mel_spectogram import plot_mel_spectrogram
        # plot_mel_spectrogram(mel_spectrogram)

        # Normalize the mel spectrogram
        # mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

        # Repeat Dimension to create RGB from greayscale
        if self.efficientnet:
            mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)

        return mel_spectrogram, one_hot_label

    def label_to_onehot(self, label, num_classes):
        """
        Convert a label to a one-hot encoded tensor
        Args:
            label (str): The bird species label.
            num_classes (int): The number of bird species
        Returns:
            torch.Tensor: A one-hot encoded tensor representing the bird species.
        """
        one_hot = torch.zeros(num_classes)
        index = self.label_to_index[label]
        one_hot[index] = 1
        return one_hot
