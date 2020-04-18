import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from datetime import datetime

from data import get_loader_splits
from utils import get_trainable, validate, lr_schedule

torch.manual_seed(42)

trainloader, validloader = get_loader_splits(augment=True)

net = torchvision.models.resnet18(pretrained=False, progress=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Moving network parameters to device
net.to(device)
# Tensorboard
writer = SummaryWriter('runs/cifar/resnet')
# Optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(get_trainable(net.parameters()), lr=0.001, weight_decay=0.005)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

# Training loop
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

        if i % 1000 == 999:  # every 1000 mini-batches...
            step = epoch * len(trainloader) + i
            valid_loss, valid_accuracy = validate(net, criterion, validloader, device)
            train_accuracy = n_correct / (1000 * trainloader.batch_size)

            # ...log the running loss
            writer.add_scalar('training loss', running_loss / 1000, step)
            writer.add_scalar('training accuracy', train_accuracy, step)
            writer.add_scalar('validation loss', valid_loss, step)
            writer.add_scalar('validation accuracy', valid_accuracy, step)

            print("Training loss: {}, training accuracy: {}, validation loss: {}, validation accuracy: {}"
                  .format(running_loss / 1000, train_accuracy, valid_loss, valid_accuracy))

            running_loss = 0.0
            n_correct = 0
    # Saving model parameters
    torch.save(net.state_dict(), "states/epoch{}".format(epoch))
    scheduler.step()

writer.add_graph(net, x)

_, acc = validate(net, criterion, validloader, device)
print("Evaluation accuracy", acc)

_, acc = validate(net, criterion, trainloader, device)
print("Training accuracy", acc)
