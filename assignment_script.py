# -*- coding: utf-8 -*-
import gzip
import numpy as np
import torch

def read_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(16)  # header skipping
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(-1, 28*28) / 255.0  # Normalize
        return torch.tensor(data)

def read_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return torch.tensor(labels, dtype=torch.long)

def get_fashion_mnist_data(path='.'):
    train_x = read_images(f'{path}/train-images-idx3-ubyte.gz')
    train_y = read_labels(f'{path}/train-labels-idx1-ubyte.gz')
    test_x = read_images(f'{path}/t10k-images-idx3-ubyte.gz')
    test_y = read_labels(f'{path}/t10k-labels-idx1-ubyte.gz')
    return train_x, train_y, test_x, test_y

# Load and verify the data
train_x, train_y, test_x, test_y = get_fashion_mnist_data()
print("Train:", train_x.shape, train_y.shape)
print("Test :", test_x.shape, test_y.shape)

import torch
import torch.nn as nn

class FashionClassifier(nn.Module):
    def __init__(self):
        super(FashionClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

import torch.nn.functional as F
from torch import optim

model = FashionClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

batch_size = 64
epochs = 50

for epoch in range(epochs):
    permutation = torch.randperm(train_x.size(0))
    epoch_loss = 0
    correct = 0
    total = 0

    for i in range(0, train_x.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        x_batch = train_x[indices]
        y_batch = train_y[indices]

        preds = model(x_batch)
        loss = F.cross_entropy(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += (preds.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    scheduler.step()
    train_acc = correct / total
    print(f"Epoch {epoch+1:2d} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f}")

torch.save(model.state_dict(), 'fashion_model_v2.pth')

model = FashionClassifier()
model.load_state_dict(torch.load('fashion_model_v2.pth'))
model.eval()

with torch.no_grad():
    outputs = model(test_x)
    predicted = outputs.argmax(dim=1)
    accuracy = (predicted == test_y).float().mean()

print(f" Final Test Accuracy: {accuracy:.4f}")

# Commented out IPython magic to ensure Python compatibility.
# %%writefile model.py
# import torch.nn as nn
# 
# class FashionClassifier(nn.Module):
#     def __init__(self):
#         super(FashionClassifier, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(784, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 10)
#         )
# 
#     def forward(self, x):
#         return self.net(x)
#
