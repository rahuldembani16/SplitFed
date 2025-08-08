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
epochs = 10  # Reduced for faster training
frac = 1
lr = 0.0001

# ================= Privacy Defense Settings (FL) =================
# Approximate DP-SGD on client updates: per-batch gradient clipping + Gaussian noise
DP_CLIP_NORM = 1.0      # L2 clip for per-batch grads (proxy for per-example)
DP_NOISE_STD = 0.01     # Gaussian noise std to add to model gradients

# Secure aggregation stub (simulation): we don't implement crypto; we simply note the step
ENABLE_SECURE_AGG = True

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
                # DP-like gradient defenses (per-batch proxy)
                if DP_CLIP_NORM is not None or (DP_NOISE_STD and DP_NOISE_STD > 0):
                    total_norm = torch.norm(torch.stack([p.grad.detach().data.norm(2) for p in net.parameters() if p.grad is not None]), 2)
                    clip_coef = DP_CLIP_NORM / (total_norm + 1e-12)
                    if DP_CLIP_NORM is not None and clip_coef < 1.0:
                        for p in net.parameters():
                            if p.grad is not None:
                                p.grad.data.mul_(clip_coef)
                    if DP_NOISE_STD and DP_NOISE_STD > 0:
                        for p in net.parameters():
                            if p.grad is not None:
                                p.grad.data.add_(DP_NOISE_STD * torch.randn_like(p.grad))
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
            all_samples, test_size=0.2, random_state=SEED, stratify=[x[1] for x in all_samples]
        )
    
    return train_samples, val_samples, classes, {cls_name: idx for idx, cls_name in enumerate(classes)}

#=============================================================================
#                         Data preprocessing
#=============================================================================

# Data preprocessing: Transformation 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# For 224x224 input size (MobileNetV2 standard)
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

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

#=============================================================================
#                         Data allocation functions
#=============================================================================

# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID datasets will be created based on this
def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# dataset_noniid() will create a dictionary to collect the indices of the data samples 
# using Dirichlet allocation for non-IID client splits
def dataset_noniid(dataset, num_users, alpha=0.3):
    """
    Sample non-I.I.D client data from dataset using Dirichlet allocation
    
    Args:
        dataset: The dataset to split
        num_users: Number of clients
        alpha: Dirichlet distribution parameter (lower = more non-IID)
    
    Returns:
        dict_users: Dictionary of user_id -> set of sample indices
    """
    labels = np.array([label for _, label in dataset])
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    idxs_by_class = {}
    for idx, label in enumerate(labels):
        if label not in idxs_by_class:
            idxs_by_class[label] = []
        idxs_by_class[label].append(idx)
    
    # Create dirichlet distribution for each client
    dict_users = {i: [] for i in range(num_users)}
    
    for cls in range(num_classes):
        # Get indices for this class
        cls_idxs = idxs_by_class[cls]
        np.random.shuffle(cls_idxs)
        
        # Sample proportions using Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        
        # Ensure we have enough samples for each client
        proportions = np.array([p * len(cls_idxs) for p in proportions])
        proportions = proportions.astype(int)
        
        # Handle any rounding issues
        diff = len(cls_idxs) - proportions.sum()
        if diff > 0:
            # Add remaining samples to random clients
            extra_clients = np.random.choice(num_users, diff, replace=False)
            proportions[extra_clients] += 1
        
        # Assign samples to clients
        start_idx = 0
        for client_id in range(num_users):
            end_idx = start_idx + proportions[client_id]
            dict_users[client_id].extend(cls_idxs[start_idx:end_idx])
            start_idx = end_idx
    
    # Convert lists to sets
    for client_id in dict_users:
        dict_users[client_id] = set(dict_users[client_id])
    
    return dict_users

#=============================================================================
#                         Main execution
#=============================================================================

# Setup dataset
train_samples, val_samples, classes, class_to_idx = setup_dataset()

if train_samples is None or val_samples is None:
    print("Failed to setup dataset. Exiting...")
    exit(1)

# Create datasets
dataset_train = PlantVillageDataset(train_samples, transform=train_transforms)
dataset_test = PlantVillageDataset(val_samples, transform=test_transforms)

#----------------------------------------------------------------
dict_users = dataset_noniid(dataset_train, num_users, alpha=0.3)
dict_users_test = dataset_iid(dataset_test, num_users)

#=============================================================================
#                         Model definition
#=============================================================================

# Using MobileNetV2 for plant disease classification
net_glob = models.mobilenet_v2(pretrained=True)

# Modify the classifier for our number of classes
num_classes = len(classes)
net_glob.classifier[1] = nn.Linear(net_glob.classifier[1].in_features, num_classes)

net_glob.to(device)
print(net_glob)

#=============================================================================
#                         Training functions
#=============================================================================

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

#=============================================================================
#                         Training Loop
#=============================================================================

net_glob.train()
w_glob = net_glob.state_dict()

loss_train = []
acc_train = []
loss_test = []
acc_test = []

for iter in range(epochs):
    w_locals, loss_locals, acc_locals = [], [], []
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)

    for idx in idxs_users:
        local = LocalUpdate(idx=idx, lr=lr, device=device, 
                          dataset_train=dataset_train, 
                          dataset_test=dataset_test, 
                          idxs=dict_users[idx], 
                          idxs_test=dict_users_test[idx])
        w, loss, acc = local.train(net=copy.deepcopy(net_glob).to(device))
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
        acc_locals.append(copy.deepcopy(acc))
        
        # Test on local test set
        local.evaluate(net=copy.deepcopy(net_glob).to(device))

    # update global weights
    # Secure aggregation simulated (no plaintext intermediate logging)
    if ENABLE_SECURE_AGG:
        # In practice use cryptographic secure aggregation; here we just avoid inspecting individual updates
        pass
    w_glob = FedAvg(w_locals)

    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)

    loss_avg = sum(loss_locals) / len(loss_locals)
    acc_avg = sum(acc_locals) / len(acc_locals)
    loss_train.append(loss_avg)
    acc_train.append(acc_avg)

    # print global training loss after each round
    print('Round {:3d}, Average loss {:.3f}, Average accuracy {:.3f}'.format(iter, loss_avg, acc_avg))

#=============================================================================
#                         Final evaluation
#=============================================================================

print("Training completed!")
print("Final training accuracy: {:.3f}%".format(acc_train[-1]))
print("Final training loss: {:.4f}".format(loss_train[-1]))

#=============================================================================
#                         Save final model
#=============================================================================

# Ensure model directory exists
import os
if not os.path.exists('./model'):
    os.makedirs('./model')

# Save final model
torch.save(net_glob.state_dict(), './model/FL_MobileNetV2_PlantVillage_final.pth')
print("✅ Final model saved to ./model/FL_MobileNetV2_PlantVillage_final.pth")

# Save training results
import pandas as pd
results_df = pd.DataFrame({
    'Round': range(1, len(loss_train) + 1),
    'Training_Loss': loss_train,
    'Training_Accuracy': acc_train
})
results_df.to_excel('./model/FL_MobileNetV2_PlantVillage_results.xlsx', index=False)
print("Training results saved to ./model/FL_MobileNetV2_PlantVillage_results.xlsx")