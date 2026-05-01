"""
Training script for boat type classification using CNN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from model import BoatCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
NUM_CLASSES = 4
IMAGE_SIZE = 64

os.makedirs('results', exist_ok=True)

transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

print("Loading dataset...")
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

class_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3, 8: 0, 9: 1}

def map_classes(dataset, mapping):
    mapped_data = []
    for img, label in dataset:
        new_label = mapping[label]
        mapped_data.append((img, new_label))
    return mapped_data

train_mapped = map_classes(train_dataset, class_mapping)
test_mapped = map_classes(test_dataset, class_mapping)

class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, mapped_data):
        self.data = mapped_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_dataset_mapped = MappedDataset(train_mapped)
test_dataset_mapped = MappedDataset(test_mapped)

train_loader = DataLoader(train_dataset_mapped, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset_mapped, batch_size=BATCH_SIZE, shuffle=False)

model = BoatCNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

train_losses = []
train_accuracies = []
test_accuracies = []

print("Starting training...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()

    scheduler.step()

    train_accuracy = 100 * correct_train / total_train
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_accuracy)

    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_losses[-1]:.4f}, '
          f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

torch.save(model.state_dict(), 'results/boat_cnn_model.pth')
print('Model saved to results/boat_cnn_model.pth')
