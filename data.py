import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


class DatasetWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label
        
    def __len__(self):
        return len(self.dataset)


def get_loader_splits(augment: bool = True, augment_valid: bool = False):
    if augment:
        train_transform = transforms.Compose(
            [
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomAffine((-10, 10)),
                transforms.RandomResizedCrop(32, scale=(0.85, 1.0)),
                transforms.ToTensor(),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.04))
            ]
        )
    else:
        train_transform = transforms.ToTensor()

    if augment_valid:
        valid_transform = train_transform
    else:
        # not applying any transformations to validation dataset
        valid_transform = transforms.ToTensor()

    # Built-in dataset (the same as Kaggle)
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True)

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size

    trainset_raw, validset_raw = torch.utils.data.random_split(dataset, [train_size, valid_size])

    trainset = DatasetWrapper(trainset_raw, train_transform)
    validset = DatasetWrapper(validset_raw, valid_transform)

    # Trainloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)

    validloader = torch.utils.data.DataLoader(validset, batch_size=10, shuffle=True, num_workers=2)

    return trainloader, validloader
