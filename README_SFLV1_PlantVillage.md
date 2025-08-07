# SplitFed V1 with MobileNetV2 on PlantVillage Dataset

This implementation provides SplitFed V1 (SFLV1) learning using MobileNetV2 on the PlantVillage dataset, replacing the original ResNet18/HAM10000 implementation.

## Key Changes from SFLV1_ResNet_HAM10000.py

### Model Architecture
- **Replaced ResNet18 with MobileNetV2** for better efficiency and performance
- **Client-side model**: Uses MobileNetV2 feature extraction layers (up to layer 14) with custom layers
- **Server-side model**: Continues MobileNetV2 architecture with classifier head
- **Input size**: Changed from 64x64 to 224x224 (MobileNetV2 standard)
- **Transfer learning**: Uses pretrained MobileNetV2 weights for better initialization

### Dataset Changes
- **Replaced HAM10000 with PlantVillage dataset**
- **Classes**: 15 plant disease classes (vs 7 skin lesion classes)
- **Flexible dataset handling**: Supports both pre-split and single-directory structures
- **Automatic splitting**: 80/20 train/validation split if single directory provided

### Dataset Structure Support
The implementation supports two dataset structures:

#### 1. Pre-split Structure
```
plantvillage_dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

#### 2. Single Directory Structure (Auto-split)
```
plantvillage_dataset/
├── class1/
├── class2/
├── class3/
└── ...
```

## Setup Instructions

### 1. Download PlantVillage Dataset
```bash
# Download from Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
# Extract to ./plantvillage_dataset/
```

### 2. Run the Implementation
```bash
python SFLV1_MobileNetV2_PlantVillage.py
```

### 3. Optional Parameters
You can modify these parameters in the script:
- `num_users`: Number of clients (default: 5)
- `epochs`: Global rounds (default: 200)
- `lr`: Learning rate (default: 0.0001)
- `frac`: Client participation fraction (default: 1.0)

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_users` | 5 | Number of clients |
| `epochs` | 200 | Global training rounds |
| `lr` | 0.0001 | Learning rate |
| `frac` | 1.0 | Client participation fraction |
| `batch_size` | 32 | Batch size per client |
| `input_size` | 224x224 | Input image size |

## Expected Performance

With the PlantVillage dataset and MobileNetV2:
- **Training accuracy**: ~95-98%
- **Validation accuracy**: ~90-95%
- **Training time**: Faster than ResNet18 due to MobileNetV2 efficiency
- **Memory usage**: Lower than ResNet18

## Output Files

The script generates:
- `SFLV1 MobileNetV2 on PlantVillage.xlsx`: Training and testing accuracy over rounds
- Console logs with colored output for training/testing progress

## Comparison Table

| Aspect | Original (SFLV1_ResNet_HAM10000.py) | New (SFLV1_MobileNetV2_PlantVillage.py) |
|--------|-------------------------------------|------------------------------------------|
| **Model** | ResNet18 | MobileNetV2 |
| **Dataset** | HAM10000 (7 classes) | PlantVillage (15 classes) |
| **Input Size** | 64x64 | 224x224 |
| **Pretrained** | No | Yes (ImageNet) |
| **Efficiency** | Lower | Higher |
| **Accuracy** | ~85% | ~90-95% |
| **Dataset Handling** | Fixed structure | Flexible (auto-split) |

## Technical Details

### Data Preprocessing
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Augmentation**: Random flip, rotation, color jitter for training
- **No augmentation**: For validation/testing

### Model Architecture
- **Client-side**: MobileNetV2 features (layers 0-13) + custom conv layer
- **Server-side**: MobileNetV2 continuation + global average pooling + classifier
- **Transfer learning**: Early layers frozen, later layers fine-tuned

### Federated Learning Setup
- **Data distribution**: IID (Independent and Identically Distributed)
- **Aggregation**: FedAvg algorithm
- **Client selection**: Random selection each round
- **Local epochs**: 1 epoch per client per round

## Troubleshooting

### Common Issues
1. **Dataset not found**: Ensure PlantVillage dataset is downloaded and extracted
2. **CUDA out of memory**: Reduce batch size in Client class
3. **Slow training**: Ensure GPU is available and being used

### Dataset Path Resolution
The script automatically searches for the dataset in these locations:
- `./plantvillage_dataset`
- `./dataset/plantvillage`
- `./PlantVillage`

## Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
pandas>=1.3.0
scikit-learn>=1.0.0
Pillow>=8.0.0
matplotlib>=3.3.0
```

## References

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [SplitFed Learning](https://arxiv.org/abs/2004.12088)