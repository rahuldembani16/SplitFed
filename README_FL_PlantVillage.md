# Federated Learning MobileNetV2 on PlantVillage

This implementation replaces the FL_ResNet_HAM10000.py with a federated learning approach using MobileNetV2 architecture and PlantVillage dataset.

## Key Changes from FL_ResNet_HAM10000.py

### Architecture Changes
- **Model**: Replaced ResNet18 with MobileNetV2 (pre-trained on ImageNet)
- **Dataset**: Switched from HAM10000 (7 classes) to PlantVillage (38 plant disease classes)
- **Input Size**: Increased from 64x64 to 224x224 for better MobileNetV2 compatibility
- **Transfer Learning**: Utilizes pre-trained MobileNetV2 with frozen backbone

### Dataset Handling
- **Flexible Structure**: Supports both single directory (auto-split) and pre-split train/val directories
- **Automatic Splitting**: 80/20 train/validation split when single directory provided
- **Class Detection**: Automatically detects all 38 plant disease classes

## Setup Instructions

### 1. Download Dataset
Download the PlantVillage dataset from Kaggle:
```bash
# Dataset: "vipoooool/new-plant-diseases-dataset"
# File: "New Plant Diseases Dataset(Augmented).zip"
```

### 2. Dataset Structure Options

**Option A: Single Directory (Auto-split 80/20)**
```
plantvillage_dataset/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Tomato___healthy/
└── ... (38 total classes)
```

**Option B: Pre-split Structure**
```
plantvillage_dataset/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ...
└── val/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    └── ...
```

### 3. Verify Dataset
```bash
python check_dataset.py
```

### 4. Run Federated Learning
```bash
python FL_MobileNetV2_PlantVillage.py
```

## Configuration Parameters

```python
# Federated Learning Settings
num_users = 5          # Number of clients
epochs = 100           # Communication rounds
frac = 1               # Fraction of clients selected each round
lr = 0.0001            # Learning rate

# Model Settings
freeze_backbone = True # Freeze MobileNetV2 backbone
num_classes = 38     # Plant disease classes

# Training Settings
batch_size = 64        # Reduced from 256 for better stability
input_size = 224       # MobileNetV2 input size
```

## Expected Performance

- **Dataset**: ~88,517 total images (PlantVillage)
- **Classes**: 38 plant disease categories
- **Architecture**: MobileNetV2 (pre-trained)
- **Training**: Federated averaging across 5 clients
- **Expected Accuracy**: 90-95% validation accuracy

## Output Files

- **Model Files**: 
  - `./model/FL_MobileNetV2_PlantVillage_best.pth` - Best validation accuracy
  - `./model/FL_MobileNetV2_PlantVillage_final.pth` - Final trained model
- **Results**: `FL MobileNetV2 on PlantVillage.xlsx` - Training metrics per round

## Comparison with Original

| Feature | FL_ResNet_HAM10000.py | FL_MobileNetV2_PlantVillage.py |
|---------|----------------------|-------------------------------|
| Architecture | ResNet18 | MobileNetV2 (pre-trained) |
| Dataset | HAM10000 (7 classes) | PlantVillage (38 classes) |
| Input Size | 64x64 | 224x224 |
| Classes | Skin lesions | Plant diseases |
| Dataset Size | ~10,000 | ~88,500 |
| Training | Federated | Federated |

## Troubleshooting

### Common Issues
1. **Dataset not found**: Ensure `plantvillage_dataset/` directory exists
2. **Memory issues**: Reduce batch_size from 64 if needed
3. **GPU memory**: Use smaller batch size or enable gradient checkpointing

### Quick Test
```bash
# Create sample dataset for testing
python check_dataset.py --create-sample
python FL_MobileNetV2_PlantVillage.py
```

## Technical Details

### Federated Learning Process
1. **Initialization**: Global MobileNetV2 model with pre-trained weights
2. **Client Selection**: Random selection of clients each round
3. **Local Training**: Each client trains on their local data
4. **Aggregation**: FedAvg algorithm for weight averaging
5. **Evaluation**: Global model evaluation on test data

### MobileNetV2 Architecture
- **Backbone**: Pre-trained MobileNetV2 (frozen by default)
- **Classifier**: Custom classifier head with dropout
- **Optimization**: Adam optimizer with differential learning rates
- **Regularization**: Dropout (0.3) and data augmentation

## Next Steps

1. **Unfreeze backbone**: Set `freeze_backbone=False` for fine-tuning
2. **Adjust epochs**: Increase to 200+ for better convergence
3. **Client data**: Modify `dataset_iid()` for non-IID data distribution
4. **Hyperparameters**: Tune learning rate and batch size
5. **Advanced FL**: Implement FedProx, FedAvgM, or other FL algorithms