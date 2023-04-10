from torch.utils.data import Dataset
from src.preprocess.audio_to_image.preprocess import preprocess_audio
import torch


# Audio Dataset
class BirdDataset(Dataset):
    def __init__(self, metadata_df, train_set_file_dir, fixed_length=300, num_classes=264, transform=None, ):
        self.metadata_df = metadata_df
        self.transform = transform
        self.fixed_length = fixed_length
        self.num_classes = num_classes
        self.base_file_path = train_set_file_dir
        self.label_to_index = {label: index for index, label in enumerate(sorted(self.metadata_df['primary_label'].unique()))}

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        file_path = row['filename']
        primary_label = row['primary_label']

        # Load and preprocess the audio
        full_file_path = self.base_file_path + "\\" + file_path
        mel_spectrogram = preprocess_audio(full_file_path)
        # mel_spectrogram = torch.unsqueeze(torch.tensor(mel_spectrogram), 0)
        # Pad or truncate the mel spectrogram to a fixed length
        mel_spectrogram = self.pad_or_truncate(mel_spectrogram, self.fixed_length)

        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        one_hot_label = self.label_to_onehot(primary_label, num_classes)
        return mel_spectrogram, one_hot_label

    def pad_or_truncate(self, mel_spectrogram, length):
        mel_spectrogram = torch.squeeze(mel_spectrogram)
        if mel_spectrogram.shape[1] < length:
            padding = length - mel_spectrogram.shape[1]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding), 'constant', 0)
        else:
            mel_spectrogram = mel_spectrogram[:, :length]

        mel_spectrogram = torch.unsqueeze(mel_spectrogram, 0)
        return mel_spectrogram

    def label_to_onehot(self, label, num_classes):
        one_hot = torch.zeros(num_classes)
        index = self.label_to_index[label]
        one_hot[index] = 1
        return one_hot
