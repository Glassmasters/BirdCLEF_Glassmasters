import matplotlib.pyplot as plt
import numpy as np


def plot_waveforms(orig_wav: np.ndarray, transformed_wav: np.ndarray):
    assert orig_wav.shape[0] == transformed_wav.shape[0]
    x_axis_points = [x for x in range(len(orig_wav))]
    plt.subplot(1, 2, 1)
    plt.plot(x_axis_points, orig_wav)
    plt.title("Original waveform")
    plt.xlabel('X-axis ')
    plt.ylabel('Y-axis ')

    plt.subplot(1, 2, 2)
    plt.plot(x_axis_points, transformed_wav)
    plt.title("Transformed waveform")
    plt.xlabel('X-axis ')
    plt.ylabel('Y-axis ')

    plt.show()