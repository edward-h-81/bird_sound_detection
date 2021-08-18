from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding='valid'
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.001),
            nn.MaxPool2d(kernel_size=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding='valid'
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.001),
            nn.MaxPool2d(kernel_size=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding='valid'
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.001),
            nn.MaxPool2d(kernel_size=(3,1), stride=(3,1), padding=0)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.001),
            nn.MaxPool2d(kernel_size=(3,1), stride=(3,1), padding=0)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.linearA = nn.Linear(752, 256)
        self.batchnormA = nn.BatchNorm1d(256)
        self.leakyrelu = nn.LeakyReLU(0.001)
        self.linearB = nn.Linear(256, 32)
        self.batchnormB = nn.BatchNorm1d(32)

        self.linear = nn.Linear(32, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linearA(x)
        x = self.batchnormA(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.linearB(x)
        x = self.batchnormB(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)


        logits = self.linear(x)
        # predictions = self.sigmoid(logits)
        return logits


if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (1, 64, 431))

