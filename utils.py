import torch
import numpy as np
import matplotlib.pyplot as plt


def lr_schedule(epoch: int) -> float:
    if epoch < 20:
        return 0.1
    elif epoch < 30:
        return 0.05
    elif epoch < 50:
        return 0.01
    elif epoch < 70:
        return 0.001
    else:
        return 0.0003


def imshow(tensor, ax=None):
    size = np.prod(tensor.shape)
    height = width = int((size // 3)**0.5)
    # permuting, because we want rgb channels as third dimension (not first)
    image = tensor.view(3, width, height)
    image = image.permute(1, 2, 0)
    if ax:
        ax.imshow(image)
    else:
        plt.imshow(image)
        plt.show()


def get_trainable(parameters):
    return (p for p in parameters if p.requires_grad)


def validate(net, criterion, loader, device):
    running_loss = 0.0
    n_correct = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_pred = net(x)
            running_loss += criterion(y_pred, y)
            n_correct += (torch.argmax(y_pred, axis=1) == y).sum().int().item()
    return running_loss / len(loader), n_correct / (len(loader)*loader.batch_size)
