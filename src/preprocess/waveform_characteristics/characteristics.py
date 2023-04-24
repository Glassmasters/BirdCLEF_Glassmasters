import torch
from torch.nn.functional import relu


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
    percentile_value = torch.quantile(audio_data, percentile / 100.0)
    return percentile_value.item()


def audio_waveform_to_db(audio_data: torch.Tensor) -> torch.Tensor:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: db_value: torch.Tensor, the audio waveform in decibels.

    """
    db_value = 20 * torch.log10(torch.abs(audio_data))
    return db_value


def audio_waveform_fft(audio_data: torch.Tensor) -> torch.Tensor:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: fft_value: torch.Tensor, the audio waveform in the frequency domain.

    """
    fft_value = torch.fft.fft(audio_data)
    return fft_value


def audio_waveform_envelope(audio_data: torch.Tensor, window_size: int = 1000) -> torch.Tensor:
    """
    Calculates the envelope of an audio waveform using PyTorch.

    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.
        window_size: int, the size of the sliding window used to compute the moving average.

    Returns:
        envelope_value: torch.Tensor, the audio waveform envelope.
    """
    rectified_signal = relu(audio_data)

    envelope = rectified_signal.unfold(0, window_size, window_size).mean(dim=-1)

    return envelope
