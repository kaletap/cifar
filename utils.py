import numpy as np
import matplotlib.pyplot as plt


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
