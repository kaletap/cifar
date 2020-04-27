"""
Script to evaluate ensemble of WideResNet models created by augmenting test data.
Each predictions is made multiple times on randomly modified images. Then, the average of all
predictions is taken as a final prediction.
"""

import torch
import argparse
import matplotlib.pyplot as plt; plt.style.use("fivethirtyeight")
from tqdm import tqdm, trange

from data import get_kaggle_testloader
from models import WideResNet22


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, required=True, help="Path to the Kaggle test folder.")
parser.add_argument("-s", "--state_path", type=str, required=True)
parser.add_argument("--augment", action="store_true", help="Whether to use data augmentation.")
parser.add_argument("-e", "--ensemble", type=int, default=1, help="Number of ensembles to use")
parser.add_argument("--save_path", type=str, default="submission.csv", help="Path in which to save submission.")
parser.add_argument("-bs", "--batch_size", type=int, default=1000, help="Test batch size.")
args = parser.parse_args()
print(args)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("Reading kaggle data into memory...")
testloader = get_kaggle_testloader(args.path, augment=args.augment, batch_size=args.batch_size)

device = torch.device("cuda")

# Reading net
net = WideResNet22()
net.load_state_dict(torch.load(args.state_path, map_location=device))
net.to(device).eval()

print("Making predictions...")
# Making predictions
ensemble_preds = []
for _ in trange(args.ensemble):
    indexes = []
    preds = []
    n_correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, idx) in enumerate(tqdm(testloader, leave=False)):
            x = x.to(device)
            y_pred = net(x)
            preds.append(y_pred)
            indexes.append(idx)
    y_pred = torch.cat(preds)
    indexes = torch.cat(indexes).numpy()
    ensemble_preds.append(y_pred)

mean_pred = torch.mean(torch.stack(ensemble_preds), 0)
final_prediction = [classes[label.item()] for label in torch.argmax(mean_pred, 1)]

# Generating submission
print("Generating submission file...")
submission = list(zip(indexes, final_prediction))
submission.sort(key=lambda t: t[0])
lines = ["id,label\n"] + ["{},{}\n".format(idx, label) for idx, label in submission]
with open(args.save_path, "w") as f:
    f.writelines(lines)
print("Done")
