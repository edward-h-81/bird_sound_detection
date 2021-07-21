import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt

# bird sound file converted and displayed as a log mel spectrogram

# bird_sound = "ff1010bird_1051.wav"
# bird_sound = "BirdVox_bird.wav"
bird_sound = "warblr_bird.wav"

audio, sr = librosa.load(bird_sound, sr=22050)

mel_spectrogram = librosa.feature.melspectrogram(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=256)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

librosa.display.specshow(log_mel_spectrogram,
                         x_axis="time",
                         y_axis="mel",
                         sr=sr, cmap='gray_r') # cmap='gray' OR cmap='gray_r' - add for grayscale
plt.colorbar(format="%+2.f dB")
plt.title("Bird sound present")
plt.show()

print(log_mel_spectrogram.shape)

# wav file, without bird sound, converted and displayed as a log mel spectrogram

# sound = "ff1010nobird_1945.wav"
# sound = "BirdVox_nobird.wav"
sound = "warblr_nobird.wav"

audio, sr = librosa.load(sound, sr=22050)

mel_spectrogram = librosa.feature.melspectrogram(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=256)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

librosa.display.specshow(log_mel_spectrogram,
                         x_axis="time",
                         y_axis="mel",
                         sr=sr, cmap='gray_r') # cmap='gray' OR cmap='gray_r' - add for grayscale
plt.colorbar(format="%+2.f dB")
plt.title("Bird sound absent")
plt.show()

print(log_mel_spectrogram.shape)

