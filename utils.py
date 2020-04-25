import torch
import numpy as np
import matplotlib.pyplot as plt


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003
    return lrate


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
            n_correct += (torch.argmax(y_pred, 1) == y).sum().int().item()
    return running_loss / len(loader), 100. * n_correct / (len(loader)*loader.batch_size)
