import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    Implements convolutional neural network inspired by Yan LeCun's ideas from 1998.
    (see: Y. Lecun et.al, "Gradient-based learning applied to document recognition")
    Note, that the network in 1998 had only one channel (grayscale 32x32 images of digits)
    and generally had some other differences in the architecture as well.
    """
    def __init__(self):
        super().__init__()
        self.name = "LeCun"
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.avg_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.avg_pool2(x)
        x = x.view([-1, 16*5*5])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
