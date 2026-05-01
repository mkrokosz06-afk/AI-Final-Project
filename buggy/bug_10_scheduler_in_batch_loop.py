"""
Bug 10: scheduler.step() called inside batch loop instead of epoch loop.
Change: scheduler.step() moved from after epoch loop to inside batch loop
Effect: No crash, but StepLR decays LR after every batch instead of every 5 epochs.
        With ~1563 batches/epoch, LR collapses to near-zero within the first epoch,
        causing the model to stop learning almost immediately (~25% accuracy).
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
NUM_EPOCHS = 15
NUM_CLASSES = 4
IMAGE_SIZE = 64

os.makedirs('results', exist_ok=True)

transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # BUG: scheduler.step() should be OUTSIDE the batch loop (called once per epoch)
        # Here it fires ~1563 times per epoch, collapsing LR to ~0 almost instantly
        scheduler.step()

        if batch_idx % 200 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.8f}')

    print(f'Epoch {epoch+1} complete. Final LR: {optimizer.param_groups[0]["lr"]:.10f}')
