import torch.nn as nn


class ReseauMNIST(nn.Module):
    def __init__(self):
        super(ReseauMNIST, self).__init__()
        self.reseau = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.reseau(x)
