# PlantVillage Dataset with MobileNetV2 - Setup Guide

This document explains how to set up and use the new MobileNetV2 model with the PlantVillage dataset.

## Dataset Setup

### Downloading the PlantVillage Dataset

1. **Download from Kaggle:**
   ```bash
   # Install kaggle if not already installed
   pip install kaggle
   
   # Download the dataset
   kaggle datasets download -d vipoooool/new-plant-diseases-dataset
   
   # Extract the dataset
   unzip new-plant-diseases-dataset.zip
   ```

2. **Manual Download:**
   - Go to [Kaggle New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   - Download the dataset
   - Extract to the project directory

### Dataset Structure

The script now supports **flexible dataset handling** with two options:

**Option A: Automatic Split (80/20)**
If you have a single directory with all classes:
```
plantvillage_dataset/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Tomato___Bacterial_spot/
└── ... (38 total classes)
```
The script will automatically split into train/val directories.

**Option B: Pre-split Structure**
If you already have train/val directories:
```
plantvillage_dataset/
├── train/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   ├── Tomato___healthy/
│   ├── Potato___Early_blight/
│   ├── Potato___healthy/
│   └── ... (38 total classes)
└── val/
    ├── Tomato___Bacterial_spot/
    ├── Tomato___Early_blight/
    ├── Tomato___healthy/
    ├── Potato___Early_blight/
    ├── Potato___healthy/
    └── ... (38 total classes)
```

### Quick Setup Commands

```bash
# Create dataset directory
mkdir -p plantvillage_dataset

# If you have the extracted dataset, move it to the correct location
mv "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train" plantvillage_dataset/
mv "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid" plantvillage_dataset/val

# Create model directory
mkdir -p model
```

## Running the Model

### Requirements

Install the required packages:

```bash
pip install torch torchvision pandas scikit-learn matplotlib pillow openpyxl
```

### Training the Model

```bash
python Normal_MobileNetV2_PlantVillage.py
```

### Model Outputs

After training, the following files will be created in the `model/` directory:

- `best_mobilenetv2_plantvillage.pth`: Best model based on validation accuracy
- `final_mobilenetv2_plantvillage.pth`: Final model after all epochs
- `training_results.png`: Training/validation loss and accuracy plots
- `training_results.xlsx`: Detailed training metrics in Excel format

## Model Architecture

The model uses **MobileNetV2** with transfer learning:

- **Backbone**: Pre-trained MobileNetV2 (ImageNet weights)
- **Classifier**: Custom classification head with dropout
- **Input Size**: 224x224 RGB images
- **Classes**: 38 plant disease classes from PlantVillage dataset
- **Transfer Learning**: Backbone frozen initially, then fine-tuning possible

## Key Features

- **Transfer Learning**: Leverages pre-trained ImageNet weights
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Early Stopping**: Built-in patience mechanism
- **Multi-GPU Support**: Automatic parallel processing if multiple GPUs available

## Performance Expectations

With the PlantVillage dataset, you can expect:
- **Training Accuracy**: ~95-98%
- **Validation Accuracy**: ~93-96%
- **Training Time**: ~30-60 minutes on GPU, ~2-4 hours on CPU
- **Model Size**: ~14MB for the saved weights

## Troubleshooting

### Verify Dataset Setup
Check if your dataset is correctly structured:
```bash
python check_dataset.py
```

**For Single Directory (Auto-split):**
```
✅ Single directory structure detected!
   Structure: single directory (auto-split 80/20)
   Classes: 38
   Classes: Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy, Blueberry___healthy...

📊 Dataset Statistics:
   Training images: 70,813
   Validation images: 17,704
   Total images: 88,517
```

**For Pre-split Structure:**
```
✅ Pre-split dataset structure verified!
   Structure: pre-split train/val
   Classes: 38
   Classes: Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy, Blueberry___healthy...

📊 Dataset Statistics:
   Training images: 70,295
   Validation images: 18,222
   Total images: 88,517

   Top 5 classes (training):
     Tomato___healthy: 16,153 images
     Potato___healthy: 6,000 images
     Tomato___Late_blight: 4,407 images
     Tomato___Septoria_leaf_spot: 4,177 images
     Tomato___Bacterial_spot: 3,765 images
```

### Troubleshooting

### Common Issues

1. **Dataset Not Found**:
   ```
   Dataset not found at plantvillage_dataset
   ```
   Solution: Ensure the dataset is properly extracted and the directory structure matches the expected format.

2. **Memory Issues**:
   - Reduce batch size in the script (line with `batch_size=32`)
   - Ensure you have at least 8GB RAM (16GB recommended)

3. **CUDA Issues**:
   - The script will automatically fall back to CPU if CUDA is not available
   - For GPU training, ensure PyTorch with CUDA support is installed

### Dataset Statistics

The PlantVillage dataset contains:
- **Total Classes**: 38 (including healthy plants)
- **Training Images**: ~70,295 images
- **Validation Images**: ~17,572 images
- **Image Types**: Various plant diseases and healthy plants
- **Crops**: Tomato, Potato, Pepper, Apple, Corn, Grape, etc.

## Comparison with Original ResNet18/HAM10000

| Feature | Original (ResNet18/HAM10000) | New (MobileNetV2/PlantVillage) |
|---------|------------------------------|---------------------------------|
| Dataset | HAM10000 (skin lesions) | PlantVillage (plant diseases) |
| Classes | 7 | 38 |
| Images | ~10,000 | ~88,000 |
| Model | ResNet18 | MobileNetV2 |
| Size | ~44MB | ~14MB |
| Input | 64x64 | 224x224 |
| Transfer Learning | No | Yes (ImageNet) |

## Next Steps

After training, you can:
1. **Fine-tune the model**: Unfreeze more layers for better performance
2. **Test on new images**: Use the saved model for inference
3. **Export to other formats**: Convert to ONNX or TensorFlow for deployment
4. **Compare with other models**: Try ResNet50, EfficientNet, etc.