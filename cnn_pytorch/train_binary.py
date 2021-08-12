import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from dcasedataset import DCASE_Dataset
from cnnbinary import CNNNetwork

ANNOTATIONS_FILE = '/content/drive/My Drive/DCASE_Datasets/labels/mini_metadata.csv'
AUDIO_DIR = '/content/drive/My Drive/DCASE_Datasets/audio/'
SAMPLE_RATE = 22050
DURATION = 10
NUM_SAMPLES = 22050 * DURATION


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        print(target.shape)

        # calculate loss
        prediction = model(input)
        # print(prediction.shape)
        sigmoid = nn.Sigmoid()
        # prediction.unsqueeze(1)
        target = target.unsqueeze_(1)
        target = target.type(torch.cuda.FloatTensor)
        loss = loss_fn(sigmoid(prediction), target)
        print(loss)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiate dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    dcase = DCASE_Dataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    train_dataloader = create_data_loader(dcase, BATCH_SIZE)

    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.BCELoss()
    optimiser = torch.optim.Adam(cnn.parameters(), 
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "/content/drive/My Drive/MSc_Project_Colab/BAD_PyTorch/cnn.pth")
    print("Trained cnn saved at cnn.pth")