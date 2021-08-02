import random

from nlpaug.util.audio.loader import AudioLoader
from nlpaug.util.audio.visualizer import AudioVisualizer
import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf

from nlpaug.util import Action, Logger
import nlpaug.model.spectrogram as nms

path = 'ff1010bird_1051.wav'

data = AudioLoader.load_mel_spectrogram(path, n_mels=80)
# AudioVisualizer.spectrogram('Original', data)

aug = nas.FrequencyMaskingAug(zone=(0, 4))

aug_data = aug.substitute(data)
AudioVisualizer.spectrogram('Frequency Masking', aug_data)