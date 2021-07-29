import json
import os
import librosa
import math
# import matplotlib.pyplot as plt
# import librosa, librosa.display


DATASET_PATH = "Hold-out_warblr_1140/test"
SAMPLE_RATE = 22050
DURATION = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
JSON_PATH = "hold-out_warblr_test.json"

def save_log_mel_spectrogram(dataset_path, json_path, n_mels=128, n_fft=2048, hop_length=512, num_segments=1):

    # dictionary to store data
    data = {
        "mapping": [],
        "log mel spectrogram": [],
        "labels": [],
        "filenames": []
    }
    # maybe create a wav file mapping?

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_log_mel_spectrogram_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # ceil rounds up e.g., 1.2 -> 2

    # loop through bird/nobird
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure not at root level
        if dirpath is not dataset_path:

            # save the label
            dirpath_components = dirpath.split("/")[-1]
            semantic_label = dirpath_components
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # process files for bird/nobird
            for f in filenames:

                # load the audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting log mel spectrogram and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mel_spectrogram = librosa.feature.melspectrogram(signal[start_sample:finish_sample],
                                                                         sr=sr,
                                                                         n_fft=n_fft,
                                                                         n_mels=n_mels,
                                                                         hop_length=hop_length)

                    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

                    # librosa.display.specshow(log_mel_spectrogram,
                    #                          x_axis="time",
                    #                          y_axis="mel",
                    #                          sr=sr, cmap='gray_r')  # cmap='gray' OR cmap='gray_r' - add for grayscale
                    # plt.colorbar(format="%+2.f dB")
                    # plt.show()

                    log_mel_spectrogram = log_mel_spectrogram.T

                    # store log mel spectrogram for segment if it has the expected length
                    if len(log_mel_spectrogram) == expected_num_log_mel_spectrogram_vectors_per_segment:
                        data["log mel spectrogram"].append(log_mel_spectrogram.tolist())
                        data["labels"].append(i-1) # -1 as the first iteration was for the dataset path
                        data["filenames"].append(f)
                        print("{}, segment:{}".format(file_path, s+1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_log_mel_spectrogram(DATASET_PATH, JSON_PATH, num_segments=1)
