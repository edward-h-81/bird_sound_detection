import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt

# bird sound file converted and displayed as a log mel spectrogram
bird_sound = "ff1010bird_1051.wav"

audio, sr = librosa.load(bird_sound, sr=22050)

mel_spectrogram = librosa.feature.melspectrogram(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=256)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

librosa.display.specshow(log_mel_spectrogram,
                         x_axis="time",
                         y_axis="mel",
                         sr=sr, cmap='gray') # cmap='gray' OR cmap='gray_r' - add for grayscale
plt.colorbar(format="%+2.f dB")
plt.show()

print(log_mel_spectrogram.shape)


