import matplotlib.pyplot as plt


def plot_waveform(waveform):
    """
    Plot the waveform of an audio signal.
    Args:
        waveform (torch.Tensor): The audio waveform.
    """
    plt.figure()
    plt.plot(waveform.t().numpy())
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.show()
