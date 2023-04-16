import characteristics

if __name__ == '__main__':
    path = "/home/meri/Documents/GitHub/BirdCLEF_Glassmasters/data/XC363502.ogg"
    audio_data = characteristics.load_audio_waveform(path)
    print(audio_data)
    print("Maximum", characteristics.audio_waveform_maximum(audio_data))
    print("Minimum", characteristics.audio_waveform_minimum(audio_data))
    print("Mean", characteristics.audio_waveform_mean(audio_data))
    print("Std", characteristics.audio_waveform_std(audio_data))
    print("percentile", characteristics.audio_waveform_percentile(audio_data, 50))
    print("db", characteristics.audio_waveform_to_db(audio_data))
    print("fft", characteristics.audio_waveform_fft(audio_data))
    print("envelope", characteristics.audio_waveform_envelope(audio_data))
