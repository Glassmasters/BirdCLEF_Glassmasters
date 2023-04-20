import os
import soundfile as sf
import torch
from typing import List


def load_audio_waveform_from_folder(root_folder: str) -> List[torch.Tensor]:
    """
    Load audio waveforms from a folder of folders of .ogg files and split them into 5-second clips.

    Parameters:
        - root_folder: str, the path to the root folder.

    Returns:
        - audio_clips: List[torch.Tensor], a list of audio clips as PyTorch tensors.
    """
    audio_clips = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.ogg'):
                file_path = os.path.join(subdir, file)
                # Load the audio waveform using PySoundFile
                audio_data, sample_rate = sf.read(file_path, dtype='float32')

                # Calculate the duration of the audio clip in seconds
                audio_duration = audio_data.shape[0] / sample_rate

                # Split the audio waveform into 5-second clips
                clip_duration = 5
                num_clips = int(audio_duration / clip_duration)
                for i in range(num_clips):
                    start = i * clip_duration * sample_rate
                    end = (i + 1) * clip_duration * sample_rate
                    audio_clip = torch.from_numpy(audio_data[start:end])
                    audio_clips.append(audio_clip)

    return audio_clips


def audio_waveform_maximum(audio_data: torch.Tensor) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: max_value: float, the maximum value of the audio waveform.

    """
    max_value = torch.max(torch.abs(audio_data))
    return max_value.item()


def audio_waveform_minimum(audio_data: torch.Tensor) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: min_value: float, the minimum value of the audio waveform.

    """
    min_value = torch.min(torch.abs(audio_data))
    return min_value.item()


def audio_waveform_mean(audio_data: torch.Tensor) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: mean_value: float, the mean value of the audio waveform.

    """
    mean_value = torch.mean(torch.abs(audio_data))
    return mean_value.item()


def audio_waveform_std(audio_data: torch.Tensor) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: std_value: float, the standard deviation value of the audio waveform.

    """
    std_value = torch.std(audio_data)
    return std_value.item()


def audio_waveform_percentile(audio_data: torch.Tensor, percentile: int) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.
        percentile: int, the percentile to calculate.

    Returns: percentile_value: float, the percentile value of the audio waveform.

    """
    percentile_value = torch.quantile(audio_data, percentile / 100)
    return percentile_value.item()


def audio_waveform_to_db(audio_data: torch.Tensor) -> torch.Tensor:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: db_value: torch.Tensor, the audio waveform in decibels.

    """
    db_value = 20 * torch.log10(torch.abs(audio_data) + 1e-6)
    return db_value


def audio_waveform_fft(audio_data: torch.Tensor) -> torch.Tensor:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: fft_value: torch.Tensor, the audio waveform in the frequency domain.

    """
    fft_value = torch.fft.fft(audio_data)
    return fft_value


def audio_waveform_envelope(audio_data: torch.Tensor) -> torch.Tensor:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: envelope_value: torch.Tensor, the audio waveform envelope.

    """
    envelope_value = torch.abs(audio_data)
    return envelope_value
