import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import argparse
import os

from data import get_loader_splits
from utils import get_trainable, validate, lr_schedule
from models import LeNet

torch.manual_seed(42)


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, required=True, help="Name of the run. Used for creating directories "
                                                                  "with tensorboard files and states of the network.")
parser.add_argument("--no_augmentation", action="store_false", help="Whether to use data augmentation.")
parser.add_argument("-e", "--epochs", type=int, default=70, help="Number of epochs.")
args = parser.parse_args()

states_dir = "states/{}".format(args.name)
if not os.path.exists(states_dir):
    os.makedirs(states_dir, exist_ok=True)

trainloader, validloader = get_loader_splits(augment=not args.no_augmentation)

net = LeNet()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Moving network parameters to device
net.to(device)
# Tensorboard
writer = SummaryWriter('runs/{}'.format(args.name))
# Optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(get_trainable(net.parameters()), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

# Training loop
n_epochs = args.epochs
for epoch in range(n_epochs):
    running_loss = 0.0
    n_correct = 0
    print("Epoch {} / {}".format(epoch + 1, n_epochs))
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
            n_correct += (torch.argmax(y_pred, 1) == y).sum().int().item()

        if i % 1000 == 999:  # every 1000 mini-batches...
            step = epoch * len(trainloader) + i
            valid_loss, valid_accuracy = validate(net, criterion, validloader, device)
            train_accuracy = n_correct / (1000 * trainloader.batch_size)

            # ...log the running loss
            writer.add_scalar('training loss', running_loss / 1000, step)
            writer.add_scalar('training accuracy', train_accuracy, step)
            writer.add_scalar('validation loss', valid_loss, step)
            writer.add_scalar('validation accuracy', valid_accuracy, step)

            print("Training loss: {:.4f}, training accuracy: {:.2f}, validation loss: {:.4f}, validation accuracy: {:.2f}"
                  .format(running_loss / 1000, 100*train_accuracy, valid_loss, 100*valid_accuracy))

            running_loss = 0.0
            n_correct = 0
    # Saving model parameters
    if epoch % 3 == 0:
        torch.save(net.state_dict(), "states/{}/epoch{}".format(args.name, epoch))
    scheduler.step()

writer.add_graph(net, x)

_, acc = validate(net, criterion, validloader, device)
print("Evaluation accuracy", acc)

_, acc = validate(net, criterion, trainloader, device)
print("Training accuracy", acc)
