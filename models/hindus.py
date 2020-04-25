import torch
import torch.nn as nn


class Hindus(nn.Module):
    def __init__(self):
        super().__init__()
        # Common operations
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # First block
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(num_features=32),
            pool,
            nn.Dropout(p=0.2)  # Dropout2D could be better
        )
        # Second block
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(num_features=64),
            pool,
            nn.Dropout(p=0.3)
        )
        # Third block
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(num_features=128),
            pool,
            nn.Dropout(p=0.4)
        )
        # Dense layer
        self.n_features = 4 * 4 * 128
        self.fc = nn.Linear(self.n_features, 10)

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1, self.n_features)
        x = self.fc(x)
        return x
