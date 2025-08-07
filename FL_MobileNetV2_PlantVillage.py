#===========================================================
# Federated learning: MobileNetV2 on PlantVillage
# PlantVillage dataset: Plant disease classification dataset
# This program is Version1: Single program simulation 
# ===========================================================
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob 
import math
import random
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from pathlib import Path

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================  
program = "FL MobileNetV2 on PlantVillage"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color during test/train 
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    

#===================================================================
# No. of users
num_users = 5
epochs = 100  # Reduced for faster training
frac = 1
lr = 0.0001

#==============================================================================================================
#                                  Client Side Program 
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class LocalUpdate(object):
    def __init__(self, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None, idxs_test=None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=64, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=64, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        
        epoch_acc = []
        epoch_loss = []
        for iter in range(self.local_ep):
            batch_acc = []
            batch_loss = []
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                
                # calculate loss
                loss = self.loss_func(fx, labels)
                # calculate accuracy
                acc = calculate_accuracy(fx, labels)
                
                #--------backward prop--------------
                loss.backward()
                optimizer.step()
                               
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            
            prRed('Client{} Train => Local Epoch: {}  \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx,
                        iter, acc.item(), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
    
    def evaluate(self, net):
        net.eval()
           
        epoch_acc = []
        epoch_loss = []
        with torch.no_grad():
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # calculate loss
                loss = self.loss_func(fx, labels)
                # calculate accuracy
                acc = calculate_accuracy(fx, labels)
                                 
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            
            prGreen('Client{} Test =>                     \tLoss: {:.4f} \tAcc: {:.3f}'.format(self.idx, loss.item(), acc.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
        return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)

#=============================================================================
#                         Data loading - PlantVillage
#=============================================================================

# Custom dataset for PlantVillage
def collect_all_samples(directory):
    """Collect all image paths and labels from directory."""
    samples = []
    classes = sorted([d for d in os.listdir(directory) 
                     if os.path.isdir(os.path.join(directory, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_dir = os.path.join(directory, cls_name)
        for filename in os.listdir(cls_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(cls_dir, filename)
                samples.append((path, class_to_idx[cls_name]))
    
    return samples, classes

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

#=============================================================================
#                         Data preprocessing
#=============================================================================

def setup_dataset():
    """Setup PlantVillage dataset with automatic splitting."""
    
    # Check for dataset directory
    dataset_path = './dataset/plantvillage'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory '{dataset_path}' not found!")
        print("Please download PlantVillage dataset from Kaggle: 'vipoooool/new-plant-diseases-dataset'")
        print("Extract to 'plantvillage_dataset/' directory")
        return None, None, None, None
    
    # Auto-detect dataset structure
    if os.path.exists(os.path.join(dataset_path, 'train')) and os.path.exists(os.path.join(dataset_path, 'val')):
        # Pre-split structure
        train_dir = os.path.join(dataset_path, 'train')
        val_dir = os.path.join(dataset_path, 'val')
        
        train_samples, classes = collect_all_samples(train_dir)
        val_samples, _ = collect_all_samples(val_dir)
        
        # Ensure classes match
        val_classes = sorted([d for d in os.listdir(val_dir) 
                           if os.path.isdir(os.path.join(val_dir, d))])
        if classes != val_classes:
            print("❌ Train and validation classes don't match!")
            return None, None, None, None
            
    else:
        # Single directory - perform automatic split
        all_samples, classes = collect_all_samples(dataset_path)
        
        # Split 80/20 train/val
        train_samples, val_samples = train_test_split(
            all_samples, test_size=0.2, random_state=SEED, stratify=[label for _, label in all_samples])
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Classes: {len(classes)} - {', '.join(classes[:3])}...")
    print(f"   Training samples: {len(train_samples)}")
    print(f"   Validation samples: {len(val_samples)}")
    
    return train_samples, val_samples, classes

#=============================================================================
#                         Data preprocessing
#=============================================================================
# Data preprocessing: Transformation 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])
    
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])

#=============================================================================
# Setup dataset
#=============================================================================
train_samples, val_samples, classes = setup_dataset()
if train_samples is None:
    exit(1)

# Create datasets
dataset_train = PlantVillageDataset(train_samples, transform=train_transforms)
dataset_test = PlantVillageDataset(val_samples, transform=test_transforms)

#=============================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID datasets will be created based on this
def dataset_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    

#---------------------------------------------
dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)

#====================================================================================================
#                               Server Side Program
#====================================================================================================
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc

#=============================================================================
#                    Model definition: MobileNetV2 with transfer learning
#=============================================================================
class MobileNetV2Transfer(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(MobileNetV2Transfer, self).__init__()
        # Load pre-trained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # Freeze backbone layers if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.last_channel, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Initialize global model
net_glob = MobileNetV2Transfer(num_classes=len(classes))

if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob = nn.DataParallel(net_glob)

net_glob.to(device)
print(net_glob)      

#===========================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

#====================================================
net_glob.train()
# copy weights
w_glob = net_glob.state_dict()

loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []

# Create model directory
os.makedirs('./model', exist_ok=True)

# Training loop
for iter in range(epochs):
    w_locals, loss_locals_train, acc_locals_train, loss_locals_test, acc_locals_test = [], [], [], [], []
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    
    # Training/Testing simulation
    for idx in idxs_users: # each client
        local = LocalUpdate(idx, lr, device, dataset_train=dataset_train, 
                          dataset_test=dataset_test, idxs=dict_users[idx], 
                          idxs_test=dict_users_test[idx])
        # Training ------------------
        w, loss_train, acc_train = local.train(net=copy.deepcopy(net_glob).to(device))
        w_locals.append(copy.deepcopy(w))
        loss_locals_train.append(copy.deepcopy(loss_train))
        acc_locals_train.append(copy.deepcopy(acc_train))
        # Testing -------------------
        loss_test, acc_test = local.evaluate(net=copy.deepcopy(net_glob).to(device))
        loss_locals_test.append(copy.deepcopy(loss_test))
        acc_locals_test.append(copy.deepcopy(acc_test))
        
    # Federation process
    w_glob = FedAvg(w_locals)
    print("------------------------------------------------")
    print("------ Federation process at Server-Side -------")
    print("------------------------------------------------")
    
    # update global model --- copy weight to net_glob -- distributed the model to all users
    net_glob.load_state_dict(w_glob)
    
    # Train/Test accuracy
    acc_avg_train = sum(acc_locals_train) / len(acc_locals_train)
    acc_train_collect.append(acc_avg_train)
    acc_avg_test = sum(acc_locals_test) / len(acc_locals_test)
    acc_test_collect.append(acc_avg_test)
    
    # Train/Test loss
    loss_avg_train = sum(loss_locals_train) / len(loss_locals_train)
    loss_train_collect.append(loss_avg_train)
    loss_avg_test = sum(loss_locals_test) / len(loss_locals_test)
    loss_test_collect.append(loss_avg_test)
    
    print('------------------- SERVER ----------------------------------------------')
    print('Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_train, loss_avg_train))
    print('Test:  Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_test, loss_avg_test))
    print('-------------------------------------------------------------------------')
    
    # Save best model
    if iter > 0 and acc_avg_test > max(acc_test_collect[:-1]):
        torch.save(net_glob.state_dict(), './model/FL_MobileNetV2_PlantVillage_best.pth')
        print(f"✅ Best model saved at round {iter+1}")

#===================================================================================     
print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process, 'acc_train': acc_train_collect, 'acc_test': acc_test_collect})     
file_name = program + ".xlsx"    
df.to_excel(file_name, sheet_name="v1_test", index=False)     

# Save final model
torch.save(net_glob.state_dict(), './model/FL_MobileNetV2_PlantVillage_final.pth')
print("✅ Final model saved to ./model/FL_MobileNetV2_PlantVillage_final.pth")

#=============================================================================
#                         Program Completed
#=============================================================================