from torch.utils.data import Dataset
import torchaudio
import torch

import tensorflow as tf
import tensorflow_hub as hub

torch.manual_seed(42)


class BirdDatasetAudioOnly(Dataset):
    """
        A custom dataset class for loading and processing bird sound data.
    """

    def __init__(self, metadata_df, train_set_file_dir, num_classes: int=264, use_pretrained_feature_extractor: bool=False):
        """
        Initialize the dataset
        Args:
            metadata_df (pd.DataFrame): A dataframe containing metadata for the audio files.
            train_set_file_dir (str): The base directory for the training set files.
            num_classes (int, optional): The number of bird species. Defaults to 264.
            use_pretrained_feature_extractor (bool, optional): Whether to use the pretrained bird classifier from tensorflow
            hub as a feature extractor.
        """
        self.metadata_df = metadata_df
        self.num_classes = num_classes
        self.base_file_path = train_set_file_dir
        self.use_pretrained_feature_extractor = use_pretrained_feature_extractor
        self.label_to_index = {label: index for index, label in enumerate(sorted(self.metadata_df['primary_label'].unique()))}

        if self.use_pretrained_feature_extractor:
            pretrained_model_handle = "https://tfhub.dev/google/bird-vocalization-classifier/1"
            self.pretrained_model = hub.load(pretrained_model_handle)


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
        audio_waveform = self.load_audio(full_file_path)

        one_hot_label = self.label_to_onehot(primary_label, self.num_classes)

        if self.use_pretrained_feature_extractor:
            audio_waveform_feature_extracted = self.pretrained_model.infer_tf(
                audio_waveform
            )
            return audio_waveform_feature_extracted[0], one_hot_label
        else:
            return audio_waveform, one_hot_label
    

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
    

    def load_audio(self, file_path, sample_rate=32000):
        """
        Load an audio file and resample it to the desired sample rate.
        Args:
            file_path (str): Path to the audio file.
            sample_rate (int, optional): Desired sample rate for the output waveform. Defaults to 32000.
        Returns:
            torch.Tensor: The resampled waveform.
        """
        waveform, _ = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        resampler = torchaudio.transforms.Resample(orig_freq=_, new_freq=sample_rate)
        waveform = resampler(waveform)
        return waveform
