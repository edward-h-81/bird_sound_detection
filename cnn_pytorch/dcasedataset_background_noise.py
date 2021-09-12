from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import torch

import matplotlib
import matplotlib.pyplot as plt

import math

class DCASE_Dataset(Dataset):

  def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
    self.annotations = pd.read_csv(annotations_file)
    self.audio_dir = audio_dir
    self.device = device
    self.transformation = transformation.to(self.device)
    self.target_sample_rate = target_sample_rate
    self.num_samples = num_samples

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    audio_sample_path = self._get_audio_sample_path(index)
    label = self._get_audio_sample_label(index)
    signal, sr = torchaudio.load(audio_sample_path) 
    signal = signal.to(self.device)
    signal = self._resample_if_necessary(signal, sr)
    signal = self._mix_down_if_necessary(signal)
    signal = self._cut_if_necessary(signal)
    signal = self._right_pad_if_necessary(signal)
    signal = self._add_background_noise(signal)
    signal = self.transformation(signal) 
    return signal, label

  def _cut_if_necessary(self, signal):
      if signal.shape[1] > self.num_samples:
          signal = signal[:, :self.num_samples]
      return signal

  def _right_pad_if_necessary(self, signal):
      length_signal = signal.shape[1]
      if length_signal < self.num_samples:
          num_missing_samples = self.num_samples - length_signal
          last_dim_padding = (0, num_missing_samples)
          signal = torch.nn.functional.pad(signal, last_dim_padding)
      return signal

  def _resample_if_necessary(self, signal, sr):
    if sr != self.target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).cuda()
        signal = resampler(signal)
    return signal

  def _mix_down_if_necessary(self, signal):
    if signal.shape[0] > 1: 
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

  def _add_background_noise(self, signal):

    bird = signal
    noise, sr = torchaudio.load('/content/drive/My Drive/torchaudio/warblr_nobird.wav')
    noise = noise.to(self.device)
    noise = self._resample_if_necessary(noise, sr)
    noise = self._mix_down_if_necessary(noise)
    noise = self._cut_if_necessary(noise)
    noise = self._right_pad_if_necessary(noise)

    # plot_waveform(noise.cpu(), self.target_sample_rate, title="Background noise")
    # plot_specgram(noise.cpu(), self.target_sample_rate, title="Background noise")
    # play_audio(noise.cpu(), self.target_sample_rate)

    bird_power = bird.norm(p=2)
    noise_power = noise.norm(p=2)

    for snr_db in [20, 10, 3]:
      snr = math.exp(snr_db / 10)
      scale = snr * noise_power / bird_power
      noisy_bird = (scale * bird + noise) / 2

      # plot_waveform(noisy_bird.cpu(), self.target_sample_rate, title=f"SNR: {snr_db} [dB]")
      # plot_specgram(noisy_bird.cpu(), self.target_sample_rate, title=f"SNR: {snr_db} [dB]")
      # play_audio(noisy_bird.cpu(), self.target_sample_rate)

    return noisy_bird

  def _get_audio_sample_path(self, index):
    fold = f"{self.annotations.iloc[index, 1]}"
    path = os.path.join(self.audio_dir, fold, f"{self.annotations.iloc[index, 0]}.wav")
    print(path)
    return path

  def _get_audio_sample_label(self, index):
    return self.annotations.iloc[index, 2]

if __name__ == "__main__":

  ANNOTATIONS_FILE = '/content/drive/My Drive/DCASE_Datasets/labels/mini_metadata.csv'
  AUDIO_DIR = '/content/drive/My Drive/DCASE_Datasets/audio/'
  SAMPLE_RATE = 22050
  DURATION = 10
  NUM_SAMPLES = 22050 * DURATION

  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"
  print(f"Using device {device}")

  mel_spectrogram = torchaudio.transforms.MelSpectrogram(
      sample_rate=SAMPLE_RATE,
      n_fft=1024,
      hop_length=512,
      n_mels=64
      )

  dcase_data = DCASE_Dataset(ANNOTATIONS_FILE, 
                             AUDIO_DIR, 
                             mel_spectrogram, 
                             SAMPLE_RATE,
                             NUM_SAMPLES,
                             device)

  print(f"There are {len(dcase_data)} samples in the dataset.")

  signal, label = dcase_data[5]

  print(signal.shape, label)

  print(signal)


  

  


