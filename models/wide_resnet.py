import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_2d(ni, nf, stride=1, ks=3):
    return nn.Conv2d(in_channels=ni, out_channels=nf,
                     kernel_size=ks, stride=stride,
                     padding=ks // 2, bias=False)


def bn_relu_conv(ni, nf):
    return nn.Sequential(nn.BatchNorm2d(ni),
                         nn.ReLU(inplace=True),
                         conv_2d(ni, nf))


class BasicBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d(ni, nf, stride)
        self.conv2 = bn_relu_conv(nf, nf)
        self.shortcut = lambda x: x
        if ni != nf:
            self.shortcut = conv_2d(ni, nf, stride, 1)

    def forward(self, x):
        x = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x)
        x = F.relu(self.conv1(x))  # nonlinearity
        x = self.conv2(x) * 0.2
        return x.add_(r)


def make_group(N, ni, nf, stride):
    start = BasicBlock(ni, nf, stride)
    rest = [BasicBlock(nf, nf) for j in range(1, N)]
    return [start] + rest


class Flatten(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, x): return x.view(x.size(0), -1)


class WideResNet(nn.Module):
    """
       Based on an article: https://bit.ly/2VlO7bu
       """
    def __init__(self, n_groups, N, n_classes, k=1, n_start=16):
        super().__init__()
        # Increase channels to n_start using conv layer
        layers = [conv_2d(3, n_start)]
        n_channels = [n_start]

        # Add groups of BasicBlock(increase channels & downsample)
        for i in range(n_groups):
            n_channels.append(n_start * (2 ** i) * k)
            stride = 2 if i > 0 else 1
            layers += [nn.Dropout2d(p=0.5)]
            layers += make_group(N, n_channels[i],
                                 n_channels[i + 1], stride)

        # Pool, flatten & add linear layer for classification
        layers += [nn.Dropout2d(p=0.5),
                   nn.BatchNorm2d(n_channels[3]),
                   nn.ReLU(inplace=True),
                   nn.AdaptiveAvgPool2d(1),
                   Flatten(),
                   nn.Linear(n_channels[3], n_classes)]

        self.features = nn.Sequential(*layers)

    def forward(self, x): return self.features(x)


class WideResNet22(WideResNet):
    def __init__(self):
        super().__init__(n_groups=3, N=3, n_classes=10, k=6)
