"""
Bug 04: Using MSELoss instead of CrossEntropyLoss for classification.
Change: nn.CrossEntropyLoss() -> nn.MSELoss()
Error: RuntimeError: Found dtype Long but expected Float
       MSELoss requires float tensors; class index targets are LongTensors.
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
LEARNING_RATE = 0.001
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
# BUG: MSELoss requires float targets; class indices are LongTensors -> dtype error
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(3):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # RuntimeError here
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
