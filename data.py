import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
from tqdm import tqdm


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
              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
              transforms.RandomAffine((-10, 10)),
              transforms.RandomCrop(32, padding=4),
              to_tensor_normalize
          ]
      )


def get_loader_splits(batch_size: int = 64, valid_batch_size: int = 128,
                      augment: bool = True, augment_valid: bool = False):

    train_transform = augment_transform if augment else to_tensor_normalize

    valid_transform = to_tensor_normalize if not augment_valid else augment_transform

    # Built-in dataset (the same as Kaggle)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=valid_transform)

    # Trainloader
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    validloader = DataLoader(testset, batch_size=valid_batch_size, shuffle=True, num_workers=2)

    return trainloader, validloader


class CifarKaggleTestset(Dataset):
    def __init__(self, data_root, augment: bool = True):
        self.augment = augment
        self.samples = []
        self.indexes = []
        for image_name in tqdm(os.listdir(data_root)):
            image_path = os.path.join(data_root, image_name)
            with open(image_path, 'r'):
                img = Image.open(image_path)
                self.samples.append(img.convert('RGB'))
                self.indexes.append(int(image_name[:-4]))
                img.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.augment:
            return augment_transform(self.samples[idx]), self.indexes[idx]
        else:
            return to_tensor_normalize(self.samples[idx]), self.indexes[idx]


def get_kaggle_testloader(data_root: str, augment: bool = True, batch_size: int = 128):
    testset = CifarKaggleTestset(data_root, augment=augment)
    return DataLoader(testset, batch_size=batch_size)
