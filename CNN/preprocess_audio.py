import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from nlpaug.util.audio.loader import AudioLoader
from nlpaug.util.audio.visualizer import AudioVisualizer
import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf

SAMPE_RATE = 22050
N_MELS = 128
WINDOW_SIZE = 2048
HOP_LENGTH = 512

# bird sound file converted and displayed as a log mel spectrogram

bird_sound = "ff1010bird_1051.wav"
# bird_sound = "BirdVox_bird.wav"
# bird_sound = "warblr_bird.wav"

audio, sr = librosa.load(bird_sound, sr=SAMPE_RATE)

mel_spectrogram = librosa.feature.melspectrogram(audio, sr=SAMPE_RATE, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)


# librosa.display.specshow(log_mel_spectrogram,
#                          x_axis="time",
#                          y_axis="mel",
#                          sr=sr, cmap='gray_r') # cmap='gray' OR cmap='gray_r' - add for grayscale
# plt.colorbar(format="%+2.f dB")
# plt.title("Bird Presence")
# plt.show()
#
# print(log_mel_spectrogram.shape)

# wav file, without bird sound, converted and displayed as a log mel spectrogram

sound = "ff1010nobird_1945.wav"
# sound = "BirdVox_nobird.wav"
# sound = "warblr_nobird.wav"

audio, sr = librosa.load(sound, sr=SAMPE_RATE)

mel_spectrogram2 = librosa.feature.melspectrogram(audio, sr=SAMPE_RATE, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS)
log_mel_spectrogram2 = librosa.power_to_db(mel_spectrogram2, ref=np.max)

# librosa.display.specshow(log_mel_spectrogram2,
#                          x_axis="time",
#                          y_axis="mel",
#                          sr=sr, cmap='gray_r') # cmap='gray' OR cmap='gray_r' - add for grayscale
# plt.colorbar(format="%+2.f dB")
# plt.title("Bird Absence")
# plt.show()

print(log_mel_spectrogram2.shape)

###########################
# NLPAUG
###########################

# Frequency masking

# aug = nas.FrequencyMaskingAug(zone=(0, 3))
#
# aug_data = aug.substitute(log_mel_spectrogram)

# librosa.display.specshow(aug_data,
#                          x_axis="time",
#                          y_axis="mel",
#                          sr=sr, cmap='gray_r') # cmap='gray' OR cmap='gray_r' - add for grayscale
# plt.colorbar(format="%+2.f dB")
# plt.title("Frequency Masking")
# plt.show()

###########################
# Time Masking

# aug = nas.TimeMaskingAug(zone=(1, 3))
#
# aug_data = aug.substitute(log_mel_spectrogram)

# librosa.display.specshow(aug_data,
#                          x_axis="time",
#                          y_axis="mel",
#                          sr=sr, cmap='gray_r') # cmap='gray' OR cmap='gray_r' - add for grayscale
# plt.colorbar(format="%+2.f dB")
# plt.title("Time Masking")
# plt.show()

###########################
# Frequency and Time Masking

flow = naf.Sequential([
    nas.FrequencyMaskingAug(zone=(0,3.667), coverage=1., factor=(11, 33)),
    nas.TimeMaskingAug(zone=(0,3.667), coverage=0.1),
])
aug_data = flow.augment(log_mel_spectrogram)

librosa.display.specshow(aug_data,
                         x_axis="time",
                         y_axis="mel",
                         sr=sr, cmap='gray_r') # cmap='gray' OR cmap='gray_r' - add for grayscale
plt.colorbar(format="%+2.f dB")
plt.title("Frequency and Time Masking")
plt.show()
