# Split Learning with MobileNetV2 on PlantVillage Dataset

This implementation provides split learning using MobileNetV2 on the PlantVillage dataset, adapted from the original SL_ResNet_HAM10000.py implementation.

## Key Changes from SL_ResNet_HAM10000.py

### Model Architecture
- **Replaced ResNet18 with MobileNetV2**: Uses a more efficient architecture with transfer learning
- **Split Architecture**: 
  - Client-side: First 14 layers of MobileNetV2 (feature extraction)
  - Server-side: Remaining layers + classifier
- **Input Size**: 224x224 (instead of 64x64) to leverage MobileNetV2's pretrained weights

### Dataset Changes
- **PlantVillage Dataset**: 15 classes of plant diseases
- **Flexible Dataset Structure**: Supports both pre-split and auto-split configurations
- **Data Augmentation**: Enhanced transforms for 224x224 images

## Dataset Setup

### Option 1: Single Directory (Auto-split 80/20)
```
plantvillage_dataset/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── ... (other class directories)
└── Tomato___Yellow_Leaf_Curl_Virus/
```

### Option 2: Pre-split Structure
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

## Installation and Setup

1. **Download PlantVillage Dataset**:
   ```bash
   # From Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
   # Extract to ./plantvillage_dataset/
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision pandas scikit-learn pillow matplotlib
   ```

3. **Verify Dataset Structure**:
   ```bash
   python check_dataset.py
   ```

## Usage

### Basic Usage
```bash
python SL_MobileNetV2_PlantVillage.py
```

### Configuration Parameters
You can modify these parameters at the top of the script:

```python
num_users = 5      # Number of clients
epochs = 200       # Training rounds
frac = 1          # Client participation fraction
lr = 0.0001       # Learning rate
```

## Expected Performance

Based on initial testing:
- **Training Accuracy**: ~85-95% (varies by number of clients)
- **Validation Accuracy**: ~80-90%
- **Training Time**: Faster than ResNet18 due to MobileNetV2 efficiency

## Output Files

- **Training Results**: `SL MobileNetV2 on PlantVillage.xlsx`
  - Contains training and validation accuracy per round
- **Console Output**: Real-time training progress with color-coded messages

## Model Architecture Details

### Client-Side Model (MobileNetV2_client_side)
- **Input**: 224x224 RGB images
- **Architecture**: First 14 layers of MobileNetV2
- **Output**: 128-channel feature maps
- **Frozen Layers**: Early layers use pretrained weights

### Server-Side Model (MobileNetV2_server_side)
- **Input**: 128-channel feature maps from client
- **Architecture**: Continuation of MobileNetV2 + classifier
- **Output**: 15 classes (PlantVillage diseases)
- **Dropout**: 0.2 for regularization

## Dataset Classes (15 classes)

The PlantVillage dataset includes:
- Apple diseases (3 classes)
- Blueberry healthy
- Cherry diseases (2 classes)
- Corn diseases (4 classes)
- Grape diseases (4 classes)
- Orange diseases (2 classes)
- Peach diseases (2 classes)
- Pepper diseases (2 classes)
- Potato diseases (2 classes)
- Raspberry healthy
- Soybean healthy
- Squash diseases
- Strawberry diseases (2 classes)
- Tomato diseases (10 classes)

## Troubleshooting

### Common Issues

1. **Dataset Not Found**:
   ```
   Error: Dataset directory './plantvillage_dataset' not found
   ```
   Solution: Download dataset from Kaggle and place in correct directory

2. **Memory Issues**:
   - Reduce batch size in Client class initialization
   - Use smaller input resolution (modify transforms)

3. **CUDA Out of Memory**:
   - Reduce batch size from 256*4 to smaller value
   - Use CPU by setting device='cpu'

### Performance Optimization

1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Batch Size**: Adjust based on GPU memory (256*4 is aggressive)
3. **Learning Rate**: Tune lr parameter for your specific setup
4. **Data Augmentation**: Modify transforms for better generalization

## Comparison with Original Implementation

| Aspect | SL_ResNet_HAM10000.py | SL_MobileNetV2_PlantVillage.py |
|--------|----------------------|--------------------------------|
| Model | ResNet18 | MobileNetV2 |
| Dataset | HAM10000 (7 classes) | PlantVillage (15 classes) |
| Input Size | 64x64 | 224x224 |
| Parameters | ~11M | ~3.5M (more efficient) |
| Pretrained | No | Yes (transfer learning) |
| Dataset Structure | Fixed | Flexible (auto-split) |

## Advanced Configuration

### Custom Dataset Path
Modify the dataset path in setup_dataset():
```python
def setup_dataset(dataset_path):
    # Add your custom path here
    possible_paths = [
        dataset_path,
        './your_custom_path',
        # ... other paths
    ]
```

### Model Modifications
- Adjust MobileNetV2 layers in client/server models
- Modify dropout rate in server model
- Change learning rate scheduling
- Add learning rate decay

## Notes

- The implementation uses transfer learning with ImageNet pretrained weights
- Early layers are frozen to prevent overfitting
- Stratified splitting ensures balanced class distribution
- Color-coded console output helps track training progress
- Results are automatically saved to Excel for analysis