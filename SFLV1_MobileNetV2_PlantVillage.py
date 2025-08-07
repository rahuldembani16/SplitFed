#============================================================================
# SplitfedV1 (SFLV1) learning: MobileNetV2 on PlantVillage
# PlantVillage dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
# 
# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP
#
# This program is Version1: Single program simulation 
# ============================================================================
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
import random
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================
program = "SFLV1 MobileNetV2 on PlantVillage"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

#===================================================================
# No. of users
num_users = 5
epochs = 5
frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
lr = 0.0001

#=====================================================================================================
#                           Client-side Model definition (MobileNetV2)
#=====================================================================================================
# Model at client side - using MobileNetV2 features
class MobileNetV2_client_side(nn.Module):
    def __init__(self):
        super(MobileNetV2_client_side, self).__init__()
        # Load pretrained MobileNetV2 and use only the feature extraction layers
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = nn.Sequential(*list(mobilenet.features.children())[:14])  # Up to layer 14
        
        # Freeze early layers for transfer learning
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Add custom layers
        self.custom_layer = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.custom_layer(x)
        return x

net_glob_client = MobileNetV2_client_side()
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)    

net_glob_client.to(device)
print(net_glob_client)     

#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================
# Model at server side - using MobileNetV2 classifier
class MobileNetV2_server_side(nn.Module):
    def __init__(self, num_classes=15):
        super(MobileNetV2_server_side, self).__init__()
        
        # Continue from where client left off
        self.features = nn.Sequential(
            nn.Conv2d(128, 320, kernel_size=1, stride=1),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(320, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

net_glob_server = MobileNetV2_server_side(num_classes=15)  # PlantVillage has 15 classes
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

net_glob_server.to(device)
print(net_glob_server)      

#===================================================================================
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0
#====================================================================================================
#                                  Server Side Program
#====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server = net_glob_server.state_dict()
w_locals_server = []

#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_server and net_server (server-side model)
net_model_server = [net_glob_server for i in range(num_users)]
net_server = copy.deepcopy(net_model_server[0]).to(device)

# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr
    
    net_server = copy.deepcopy(net_model_server[idx]).to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

    
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device).long()
    
    #---------forward prop-------------
    fx_server = net_server(fx_client)
    
    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop-------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # Update the server-side model for the current client
    net_model_server[idx] = copy.deepcopy(net_server)
    
    return dfx_client

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test
    global acc_test_collect, loss_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_locals_server, w_glob_server, net_server
    
    net = copy.deepcopy(net_model_server[idx])
    net.eval()
    
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device).long()
        #---------forward prop-------------
        fx_server = net(fx_client)
        
        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
        count2 += 1
        if count2 == num_users:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            acc_test_collect.append(acc_avg_test)
            loss_test_collect.append(loss_avg_test)
            
            print('Global Test Round: {:3d}, Average Accuracy: {:.3f}%, Average Loss: {:.4f}'.format(ell, acc_avg_test, loss_avg_test))

#=====================================================================================================
#                                       Clients-side Program
#=====================================================================================================
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
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1 
        #self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 32, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 32, shuffle = True)

    def train(self, net_client):
        net_client.train()
        net_server = copy.deepcopy(net_glob_server).to(self.device)
        net_server.train()
        
        optimizer_client = torch.optim.Adam(net_client.parameters(), lr = lr) 
        optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr) 
        
        epoch_loss = []
        epoch_acc = []
        
        for iter in range(self.local_ep):
            batch_loss = []
            batch_acc = []
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device).long()
                
                # Client-side forward pass
                optimizer_client.zero_grad()
                optimizer_server.zero_grad()
                
                fx_client = net_client(images)
                fx_server = net_server(fx_client)
                
                # calculate loss
                loss = criterion(fx_server, labels)
                acc = calculate_accuracy(fx_server, labels)
                
                # Server-side backward pass
                loss.backward()
                optimizer_server.step()
                optimizer_client.step()
                               
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
                
            prRed('Client{} Train => Local Epoch: {}  \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx,
                        iter, acc.item(), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
            
        return net_client.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
    
    def evaluate(self, net_client, ell):
        net_client.eval()
        net_server = copy.deepcopy(net_glob_server).to(self.device)
        net_server.eval()
           
        epoch_loss = []
        epoch_acc = []
        
        with torch.no_grad():
            batch_loss = []
            batch_acc = []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device).long()
                
                # Client-side forward pass
                fx_client = net_client(images)
                fx_server = net_server(fx_client)
                
                # calculate loss
                loss = criterion(fx_server, labels)
                acc = calculate_accuracy(fx_server, labels)
                                 
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
                
            prGreen('Client{} Test =>                     \tLoss: {:.4f} \tAcc: {:.3f}'.format(self.idx, loss.item(), acc.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
            
        return net_client.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)

#=============================================================================
#                         Data loading and Setup Functions
#=============================================================================

def collect_all_samples(root_dir):
    """Collect all image paths and labels from a directory structure."""
    samples = []
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset directory '{root_dir}' not found. Please download the PlantVillage dataset.")
    
    # Check if this is a pre-split structure or single directory
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # Pre-split structure
        train_samples, classes, class_to_idx = collect_samples_from_dir(train_dir)
        val_samples, _, _ = collect_samples_from_dir(val_dir)
        return train_samples, val_samples, classes, class_to_idx, False  # False = not auto-split
    else:
        # Single directory structure - collect all samples for auto-split
        all_samples, classes, class_to_idx = collect_samples_from_dir(root_dir)
        return all_samples, None, classes, class_to_idx, True  # True = auto-split needed

def collect_samples_from_dir(directory):
    """Collect samples from a single directory."""
    samples = []
    
    if not os.path.exists(directory):
        return samples, [], {}
    
    classes = sorted([d for d in os.listdir(directory) 
                     if os.path.isdir(os.path.join(directory, d))])
    
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_dir = os.path.join(directory, cls_name)
        if not os.path.isdir(cls_dir):
            continue
            
        for filename in os.listdir(cls_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(cls_dir, filename)
                samples.append((path, cls_name))
    
    return samples, classes, class_to_idx

def setup_dataset(dataset_path):
    """Setup dataset with flexible structure handling."""
    print(f"Setting up dataset from: {dataset_path}")
    
    # Try different possible dataset paths
    possible_paths = [
        dataset_path,
        './plantvillage_dataset',
        './dataset/plantvillage',
        './PlantVillage'
    ]
    
    root_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            root_dir = path
            break
    
    if root_dir is None:
        raise FileNotFoundError(
            f"PlantVillage dataset not found. Please download from: "
            f"https://www.kaggle.com/datasets/emmarex/plantdisease and place in one of: {possible_paths}"
        )
    
    # Collect samples and handle dataset structure
    result = collect_all_samples(root_dir)
    
    if len(result) == 5:  # New format with classes and class_to_idx
        samples, val_samples, classes, class_to_idx, auto_split = result
    else:  # Fallback
        samples, val_samples, classes, class_to_idx, auto_split = result
    
    if auto_split:
        # Perform 80/20 stratified split
        from sklearn.model_selection import train_test_split
        
        paths, labels = zip(*samples) if samples else ([], [])
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, test_size=0.2, random_state=SEED, stratify=labels
        )
        
        train_samples = list(zip(train_paths, train_labels))
        val_samples = list(zip(val_paths, val_labels))
    else:
        # Use pre-split samples
        train_samples = samples
        
    return train_samples, val_samples, classes, class_to_idx

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
#                         Custom dataset for PlantVillage
#=============================================================================

class PlantVillageDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.class_to_idx[label]
        return image, label_idx

#=============================================================================
#                         Data preprocessing
#=============================================================================

# Setup dataset
try:
    train_samples, val_samples, classes, class_to_idx = setup_dataset('./plantvillage_dataset')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the PlantVillage dataset is properly downloaded.")
    exit(1)

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

# Create datasets
dataset_train = PlantVillageDataset(train_samples, class_to_idx, transform=train_transforms)
dataset_test = PlantVillageDataset(val_samples, class_to_idx, transform=test_transforms)

#----------------------------------------------------------------
dict_users = dataset_noniid(dataset_train, num_users, alpha=0.3)
dict_users_test = dataset_iid(dataset_test, num_users)

#=============================================================================
#                         Training And Testing
#=============================================================================

net_glob_client.train()
#copy weights
w_glob_client = net_glob_client.state_dict()
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    w_locals_client = []
    
    # Collect training metrics for this epoch
    epoch_train_losses = []
    epoch_train_accs = []
      
    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, 
                      dataset_train=dataset_train, 
                      dataset_test=dataset_test, 
                      idxs=dict_users[idx], 
                      idxs_test=dict_users_test[idx])
        
        # Training ------------------
        w_client, loss, acc = local.train(copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))
        
        # Collect training metrics
        epoch_train_losses.append(loss)
        epoch_train_accs.append(acc)
        
        # Testing -------------------
        _, loss, acc = local.evaluate(copy.deepcopy(net_glob_client).to(device), ell=iter)
            
    # Ater serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    w_glob_client = FedAvg(w_locals_client)   
    
    # copy weight to net_glob_client -- global model of client
    net_glob_client.load_state_dict(w_glob_client)
    
    # Server-side model to train
    net_server = copy.deepcopy(net_glob_server).to(device)
    
    # Train server-side model
    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, 
                      dataset_train=dataset_train, 
                      dataset_test=dataset_test, 
                      idxs=dict_users[idx], 
                      idxs_test=dict_users_test[idx])
        
        # Training ------------------
        w_client, loss, acc = local.train(copy.deepcopy(net_glob_client).to(device))
        
        # Testing -------------------
        _, loss, acc = local.evaluate(copy.deepcopy(net_glob_client).to(device), ell=iter)
    
    # Calculate average training metrics for this epoch
    avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    avg_train_acc = sum(epoch_train_accs) / len(epoch_train_accs)
    
    # Collect metrics
    loss_train_collect.append(avg_train_loss)
    acc_train_collect.append(avg_train_acc)
    
    print(f'Epoch {iter+1}/{epochs} - Avg Train Loss: {avg_train_loss:.4f}, Avg Train Acc: {avg_train_acc:.4f}')

#=============================================================================
#                         Final evaluation
#=============================================================================

print("Training completed!")

#=============================================================================
#                         Save trained models
#=============================================================================

# Ensure model directory exists
import os
if not os.path.exists('model'):
    os.makedirs('model')

# Save client model
torch.save(net_glob_client.state_dict(), 'model/SFL_MobileNetV2_client_PlantVillage_final.pth')
print("✅ Client model saved to model/SFL_MobileNetV2_client_PlantVillage_final.pth")

# Save server model
torch.save(net_glob_server.state_dict(), 'model/SFL_MobileNetV2_server_PlantVillage_final.pth')
print("✅ Server model saved to model/SFL_MobileNetV2_server_PlantVillage_final.pth")

# Save training results
import pandas as pd
results_df = pd.DataFrame({
    'Round': range(1, len(loss_train_collect) + 1),
    'Training_Loss': loss_train_collect,
    'Training_Accuracy': acc_train_collect
})
results_df.to_excel('model/SFL_MobileNetV2_PlantVillage_results.xlsx', index=False)
print("Training results saved to model/SFL_MobileNetV2_PlantVillage_results.xlsx")