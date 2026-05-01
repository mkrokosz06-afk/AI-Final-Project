"""
Bug 05: Excessively high learning rate.
Change: LEARNING_RATE = 0.001 -> LEARNING_RATE = 10.0
Effect: Loss becomes NaN after a few batches due to numerical overflow
        in softmax/cross-entropy computations with exploding weights.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import BoatCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
# BUG: learning rate of 10.0 causes exploding gradients and NaN loss
LEARNING_RATE = 10.0
NUM_CLASSES = 4
IMAGE_SIZE = 64

os.makedirs('results', exist_ok=True)

transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
class_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3, 8: 0, 9: 1}

class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mapping):
        self.data = [(img, mapping[lbl]) for img, lbl in dataset]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

train_loader = DataLoader(MappedDataset(train_dataset, class_mapping), batch_size=BATCH_SIZE, shuffle=True)

model = BoatCNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(3):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
