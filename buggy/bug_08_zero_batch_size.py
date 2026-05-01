"""
Bug 08: Batch size set to zero.
Change: BATCH_SIZE = 32 -> BATCH_SIZE = 0
Error: ValueError: batch_size should be a positive integer value, but got batch_size=0
       DataLoader cannot create batches of zero samples.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 64

# BUG: batch_size=0 is invalid — DataLoader requires a positive integer
BATCH_SIZE = 0

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
class_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3, 8: 0, 9: 1}

class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mapping):
        self.data = [(img, mapping[lbl]) for img, lbl in dataset]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# This line raises ValueError: batch_size should be a positive integer
train_loader = DataLoader(MappedDataset(train_dataset, class_mapping), batch_size=BATCH_SIZE, shuffle=True)
print("DataLoader created successfully")  # Never reached
