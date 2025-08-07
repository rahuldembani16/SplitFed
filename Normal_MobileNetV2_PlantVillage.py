#=====================================================
# Centralized (normal) learning: MobileNetV2 on PlantVillage Dataset
# Single program with Transfer Learning
# ====================================================
import os
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame

import math
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import random
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================    
program = "Normal Learning MobileNetV2 on PlantVillage Dataset"
print(f"---------{program}----------")              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#=============================================================================
#                         Data loading for PlantVillage Dataset
#=============================================================================

# Define the path to your PlantVillage dataset
# The dataset can have either:
# Option 1: Single directory with all classes
# plantvillage_dataset/
# ├── class1/
# ├── class2/
# └── ...
# 
# Option 2: Pre-split train/val directories
# plantvillage_dataset/
# ├── train/
# └── val/

dataset_path = './dataset/plantvillage'

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    print("Please ensure the PlantVillage dataset is available at this location")
    print("You can download the dataset from Kaggle: 'vipoooool/new-plant-diseases-dataset'")
    exit(1)

# Auto-detect dataset structure
train_dir = None
val_dir = None
auto_split = False

# Check if we have train/val directories
if os.path.exists(os.path.join(dataset_path, 'train')) and os.path.exists(os.path.join(dataset_path, 'val')):
    # Pre-split structure
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    print("Using pre-split train/val directories")
else:
    # Single directory - we'll split automatically
    auto_split = True
    print("Single directory detected - will automatically split into train/val")
    data_dir = dataset_path

# Get class names
def get_classes_from_directory(directory):
    return sorted([d for d in os.listdir(directory) 
                   if os.path.isdir(os.path.join(directory, d))])

if auto_split:
    classes = get_classes_from_directory(data_dir)
else:
    classes = get_classes_from_directory(os.path.join(dataset_path, 'train'))

num_classes = len(classes)
print(f"Number of classes: {num_classes}")
print(f"Classes: {classes}")

# Create class to index mapping
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

#=============================================================================
# Custom Dataset for PlantVillage
#=============================================================================
class PlantVillageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def collect_all_samples(directory, classes, class_to_idx):
    """Collect all image samples from a directory."""
    samples = []
    for class_name in classes:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_to_idx[class_name]))
    return samples

#=============================================================================
#                         Data preprocessing
#=============================================================================
# Data preprocessing: Transformation 
# Using ImageNet statistics as MobileNetV2 was trained on ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Image size for MobileNetV2 (224x224 is standard)
IMAGE_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Create datasets based on detected structure
if auto_split:
    # Automatically split the dataset
    all_samples = collect_all_samples(data_dir, classes, class_to_idx)
    
    # Split into train and validation (80-20 split)
    train_samples, val_samples = train_test_split(
        all_samples, test_size=0.2, random_state=SEED, stratify=[s[1] for s in all_samples]
    )
    
    train_dataset = PlantVillageDataset(train_samples, transform=train_transforms)
    val_dataset = PlantVillageDataset(val_samples, transform=val_transforms)
    
    print(f"Automatically split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
else:
    # Use pre-split directories
    train_samples = collect_all_samples(train_dir, classes, class_to_idx)
    val_samples = collect_all_samples(val_dir, classes, class_to_idx)
    
    train_dataset = PlantVillageDataset(train_samples, transform=train_transforms)
    val_dataset = PlantVillageDataset(val_samples, transform=val_transforms)
    
    print(f"Using pre-split directories: {len(train_dataset)} train, {len(val_dataset)} val")

# Create data loaders
train_iterator = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_iterator = DataLoader(val_dataset, batch_size=32, shuffle=False)

#=============================================================================
#                    Model definition: MobileNetV2 with Transfer Learning
#=============================================================================

class MobileNetV2Transfer(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(MobileNetV2Transfer, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Replace the classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.last_channel, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Initialize model
net_glob = MobileNetV2Transfer(num_classes=num_classes, freeze_backbone=True)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs")
    net_glob = nn.DataParallel(net_glob)

net_glob.to(device)

#=============================================================================
#                    Loss function and Optimizer
#=============================================================================

criterion = nn.CrossEntropyLoss()

# Use different learning rates for backbone and classifier
backbone_params = []
classifier_params = []

for name, param in net_glob.named_parameters():
    if 'classifier' in name:
        classifier_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': 0.0001},  # Lower LR for backbone
    {'params': classifier_params, 'lr': 0.001}  # Higher LR for classifier
])

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

#=============================================================================
#                    Training and Evaluation Functions
#=============================================================================

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch_idx, (data, target) in enumerate(iterator):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        acc = calculate_accuracy(output, target)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for data, target in iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            acc = calculate_accuracy(output, target)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def calculate_accuracy(y_pred, y_true):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y_true.view_as(top_pred)).sum()
    return correct.float() / y_pred.shape[0]

#=============================================================================
#                    Training Loop
#=============================================================================

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

EPOCHS = 20
best_val_acc = 0.0

train_losses = []
train_accs = []
val_losses = []
val_accs = []

print("Starting training...")

for epoch in range(EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(net_glob, train_iterator, optimizer, criterion)
    val_loss, val_acc = evaluate(net_glob, val_iterator, criterion)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(net_glob.state_dict(), 'model/best_mobilenetv2_plantvillage.pth')
        print(f'New best model saved with validation accuracy: {val_acc:.4f}')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.0f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:.2f}%')

# Save final model
torch.save(net_glob.state_dict(), 'model/final_mobilenetv2_plantvillage.pth')
print(f'Training completed! Best validation accuracy: {best_val_acc*100:.2f}%')

#=============================================================================
#                    Plotting Results
#=============================================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([acc*100 for acc in train_accs], label='Train Accuracy')
plt.plot([acc*100 for acc in val_accs], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('model/training_results.png')
plt.close()

#=============================================================================
#                    Save Training Results to Excel
#=============================================================================

df = DataFrame({
    'epoch': list(range(1, EPOCHS+1)),
    'train_loss': train_losses,
    'train_accuracy': [acc*100 for acc in train_accs],
    'val_loss': val_losses,
    'val_accuracy': [acc*100 for acc in val_accs]
})

df.to_excel('model/training_results.xlsx', index=False)
print("Training results saved to model/training_results.xlsx")