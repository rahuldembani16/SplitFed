#=============================================================================
# Split learning: MobileNetV2 on PlantVillage
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
program = "SL MobileNetV2 on PlantVillage"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

#===================================================================  
# No. of users
num_users = 5
epochs = 200
frac = 1   # participation of clients; if 1 then 100% clients participate in SL
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
    net_glob_client = nn.DataParallel(net_glob_client)   # to use the multiple GPUs; later we can change this to CPUs only 

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


#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False

# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user
    
    net_glob_server.train()
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr = lr)

    
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_glob_server(fx_client)
    
    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # server-side model net_glob_server is global so it is updated automatically in each pass to this function
        # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        
                
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            l_epoch_check = True                # for evaluate_server function - to check local epoch has hitted 
                       
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is to check if all users are served for one round --------------------
        if len(idx_collect) == num_users:
            fed_check = True                                                  # for evaluate_server function  - to check fed check has hitted
            # all users served for one round ------------------------- output print and update is done in evaluate_server()
            # for nicer display 
                        
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
    # send gradients to the client               
    return dfx_client

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train
    
    net_glob_server.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        fx_server = net_glob_server(fx_client)
        
        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
               
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            
            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                
            # if all users are served for one round ----------                    
            if fed_check:
                fed_check = False
                                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                               
                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")
         
    return 

#==============================================================================================================
#                                       Clients Side Program
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
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1 
        #self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 32, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 32, shuffle = True)
        

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                
                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
                            
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict() 
    
    def evaluate(self, net, ell):
        net.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # Sending activations to server 
                evaluate_server(fx, labels, self.idx, len_batch, ell)
            
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            
        return          

#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID datasets will be created based on this
def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    

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
        train_samples = collect_samples_from_dir(train_dir)
        val_samples = collect_samples_from_dir(val_dir)
        return train_samples, val_samples, None, False  # False = not auto-split
    else:
        # Single directory structure - collect all samples for auto-split
        all_samples = collect_samples_from_dir(root_dir)
        return all_samples, None, None, True  # True = auto-split needed

def collect_samples_from_dir(directory):
    """Collect image samples from a directory with class subdirectories."""
    samples = []
    
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Find all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        for filename in os.listdir(class_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                file_path = os.path.join(class_dir, filename)
                samples.append((file_path, class_name))
    
    return samples

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
    
    if len(result) == 4:  # New format with auto-split flag
        samples, val_samples, _, auto_split = result
    else:  # Fallback
        samples, val_samples, _, auto_split = result
    
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
        train_samples = samples
        val_samples = val_samples
    
    # Get unique classes
    all_labels = [label for _, label in train_samples + val_samples]
    classes = sorted(list(set(all_labels)))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    print(f"Dataset setup complete:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Classes: {len(classes)}")
    print(f"  Class names: {classes}")
    
    return train_samples, val_samples, classes, class_to_idx

# Custom dataset for PlantVillage
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
#                         Main execution
#============================================================================= 

# Setup dataset
try:
    train_samples, val_samples, classes, class_to_idx = setup_dataset('./plantvillage_dataset')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the PlantVillage dataset is properly downloaded.")
    exit(1)

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

# Create datasets
dataset_train = PlantVillageDataset(train_samples, class_to_idx, transform=train_transforms)
dataset_test = PlantVillageDataset(val_samples, class_to_idx, transform=test_transforms)

#----------------------------------------------------------------
dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)

#=============================================================================
#                         Training Loop
#============================================================================= 

#net_glob_client.train()
# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)

    # Sequential training/testing among clients      
    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, 
                      dataset_train=dataset_train, 
                      dataset_test=dataset_test, 
                      idxs=dict_users[idx], 
                      idxs_test=dict_users_test[idx])
        
        # Training ------------------
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))
              
        # Testing -------------------
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)
        
        # copy weight to net_glob_client -- use to update the client-side model of the next client to be trained
        net_glob_client.load_state_dict(w_client)
   
#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
file_name = program+".xlsx"    
df.to_excel(file_name, sheet_name="v1_test", index=False)     

#=============================================================================
#                         Program Completed
#=============================================================================