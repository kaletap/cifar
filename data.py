import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


def get_loader_splits(batch_size: int = 64, augment: bool = True, augment_valid: bool = False):
    stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    to_tensor_normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ]
    )

    augment_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                to_tensor_normalize
            ]
        )

    train_transform = augment_transform if augment else to_tensor_normalize

    valid_transform = to_tensor_normalize if not augment_valid else augment_transform

    # Built-in dataset (the same as Kaggle)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=valid_transform)

    # Trainloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    validloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=2)

    return trainloader, validloader
