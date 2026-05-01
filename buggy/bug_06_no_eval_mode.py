"""
Bug 06: model.eval() not called before test/evaluation loop.
Change: removed model.eval() line before the test loop
Effect: No crash, but Dropout remains active during inference,
        causing lower and non-deterministic test accuracy.
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
NUM_EPOCHS = 5
NUM_CLASSES = 4
IMAGE_SIZE = 64

os.makedirs('results', exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
class_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3, 8: 0, 9: 1}

class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mapping):
        self.data = [(img, mapping[lbl]) for img, lbl in dataset]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

train_loader = DataLoader(MappedDataset(train_dataset, class_mapping), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(MappedDataset(test_dataset, class_mapping), batch_size=BATCH_SIZE, shuffle=False)

model = BoatCNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # BUG: model.eval() is NOT called here — Dropout stays active during test
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Epoch {epoch+1}: Test Acc = {100*correct/total:.2f}%')
