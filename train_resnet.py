import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt; plt.style.use("fivethirtyeight")
from tqdm import tqdm, trange
from typing import List
from collections import Counter
import cv2
from datetime import datetime


# Transforms
def resize(img):
    return cv2.resize(img.numpy(), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
def permute(img):
    return img.permute(1, 2, 0)

def permute_back(img):
    return img.permute(2, 0, 1)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        permute,
        resize,
        transforms.ToTensor()
    ]
)

# Built-in dataset (the same as Kaggle)
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                       download=True, transform=transform)

train_size = int(0.8 * len(dataset))  # 40_000
test_size = len(dataset) - train_size  # 10_000
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Trainloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=400,
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_trainable(parameters):
    return (p for p in parameters if p.requires_grad)


net = torchvision.models.resnet18(pretrained=True, progress=True)
print("Loaded model")
net.fc = nn.Linear(512, 10) 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Moving network parameters to device
net.to(device)
# Tensorboard
writer = SummaryWriter('runs/cifar/transfer')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(get_trainable(net.parameters()), lr=0.001)

N_EPOCHS = 15
for epoch in range(N_EPOCHS):  
    running_loss = 0.0
    print("Epoch {} / {}".format(epoch + 1, N_EPOCHS))
    print("Time", datetime.now())
    for i, (x, y) in enumerate(tqdm(trainloader, leave=False)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = net(x)
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i)

            running_loss = 0.0
    # Saving model parameters
    torch.save(net.state_dict(), "states/epoch{}".format(epoch))

writer.add_graph(net, x)


def get_accuracy_and_predictions(data):
    predictions = []
    n = len(data)
    correct = 0
    for tensor, label in tqdm(data, leave=False):
        tensor = tensor.to(device)
        logits = net(tensor.view(-1, 3, 256, 256))
        y_pred = logits.argmax().item()
        predictions.append(y_pred)
        if (y_pred == label):
            correct += 1
    return correct / n, predictions


acc, preds = get_accuracy_and_predictions(testset)
print("Accuracy", acc)
print(Counter(preds))