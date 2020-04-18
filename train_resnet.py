import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm, trange
from typing import List
from collections import Counter
import cv2
from datetime import datetime

from data import get_loader_splits


torch.manual_seed(42)


trainloader, validloader = get_loader_splits()


net = torchvision.models.resnet18(pretrained=False, progress=True)
print("Loaded model")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Moving network parameters to device
net.to(device)
# Tensorboard
writer = SummaryWriter('runs/cifar/transfer')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(get_trainable(net.parameters()), lr=0.001, weight_decay=0.005)

def validate(loader):
    running_loss = 0.0
    n_correct = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_pred = net(x)
            running_loss += criterion(y_pred, y)
            n_correct += (torch.argmax(y_pred, axis=1) == y).sum().int().item()
    return running_loss / len(loader), n_correct / (len(loader)*loader.batch_size)

        

N_EPOCHS = 70
for epoch in range(N_EPOCHS):  
    running_loss = 0.0
    n_correct = 0
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
        with torch.no_grad():
            n_correct += (torch.argmax(y_pred, axis=1) == y).sum().int().item()
        
        if i % 1000 == 999:    # every 1000 mini-batches...
            step = epoch * len(trainloader) + i
            valid_loss, valid_accuracy = validate(validloader)
            train_accuracy = n_correct / (1000*trainloader.batch_size)

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            step)
            writer.add_scalar('training accuracy',
                            train_accuracy,
                            step)
            writer.add_scalar('validation loss',
                            valid_loss,
                            step)
            writer.add_scalar('validation accuracy',
                            valid_accuracy,
                            step)

            print("Training loss: {}, training accuracy: {}, validation loss: {}, validation accuracy: {}"\
                    .format(
                        running_loss / 1000, 
                        train_accuracy,
                        valid_loss,
                        valid_accuracy)
                    )

            running_loss = 0.0
            n_correct = 0
    # Saving model parameters
    torch.save(net.state_dict(), "states/epoch{}".format(epoch))

writer.add_graph(net, x)


def get_accuracy_and_predictions(data):
    predictions = []
    n = len(data)
    correct = 0
    for tensor, label in tqdm(data, leave=False):
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = net(tensor.view(-1, 3, 32, 32))
            y_pred = logits.argmax().item()
            predictions.append(y_pred)
            if (y_pred == label):
                correct += 1
    return correct / n, predictions


acc, preds = get_accuracy_and_predictions(testset)
print("Accuracy", acc)
print(Counter(preds))

acc, preds = get_accuracy_and_predictions(trainset)
print("Training accuracy", acc)
print(Counter(preds))
