"""
Bug 07: Normalization standard deviation set to zeros.
Change: Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
     -> Normalize((0.485,0.456,0.406),(0.0, 0.0, 0.0))
Error: ZeroDivisionError / tensor values become inf or NaN
       because Normalize computes (pixel - mean) / std, and std=0 is division by zero.
"""
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import BoatCNN
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 64

# BUG: std=(0.0, 0.0, 0.0) causes division by zero in Normalize transform
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.0, 0.0, 0.0))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
class_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3, 8: 0, 9: 1}

class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mapping):
        self.data = [(img, mapping[lbl]) for img, lbl in dataset]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

train_loader = DataLoader(MappedDataset(train_dataset, class_mapping), batch_size=32, shuffle=True)

model = BoatCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')  # Will print NaN or inf
    if batch_idx >= 5:
        break
