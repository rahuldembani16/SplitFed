#!/usr/bin/env python3
"""
Dataset verification script for PlantVillage dataset.
This script checks if the dataset is properly set up and provides statistics.
"""

import os
import sys
from pathlib import Path

def check_dataset_structure(dataset_path='./dataset/plantvillage'):
    """Check if the PlantVillage dataset is properly structured."""
    
    print("=" * 60)
    print("PlantVillage Dataset Verification")
    print("=" * 60)
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory '{dataset_path}' not found!")
        print("\nTo set up the dataset:")
        print("1. Download from Kaggle: 'vipoooool/new-plant-diseases-dataset'")
        print("2. Extract to 'plantvillage_dataset/' directory")
        print("3. Choose structure:")
        print("   - Option 1: Single directory with all classes (auto-split)")
        print("   - Option 2: Pre-split train/val directories")
        return False
    
    # Count images in each class
    def count_images(directory, classes):
        counts = {}
        for cls in classes:
            cls_path = os.path.join(directory, cls)
            images = [f for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            counts[cls] = len(images)
        return counts
    
    # Auto-detect dataset structure
    auto_split = False
    
    # Check if we have train/val directories
    if os.path.exists(os.path.join(dataset_path, 'train')) and os.path.exists(os.path.join(dataset_path, 'val')):
        # Pre-split structure
        train_path = os.path.join(dataset_path, 'train')
        val_path = os.path.join(dataset_path, 'val')
        
        train_classes = sorted([d for d in os.listdir(train_path) 
                             if os.path.isdir(os.path.join(train_path, d))])
        val_classes = sorted([d for d in os.listdir(val_path) 
                           if os.path.isdir(os.path.join(val_path, d))])
        
        if train_classes != val_classes:
            print("❌ Training and validation classes don't match!")
            return False
        
        classes = train_classes
        structure_type = "pre-split train/val"
        
        # Count images
        train_counts = count_images(train_path, classes)
        val_counts = count_images(val_path, classes)
        
        total_train = sum(train_counts.values())
        total_val = sum(val_counts.values())
        
        print(f"✅ Pre-split dataset structure verified!")
        
    elif os.path.isdir(dataset_path):
        # Single directory - will auto-split
        auto_split = True
        classes = sorted([d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))])
        
        if not classes:
            print("❌ No classes found in dataset directory!")
            return False
        
        # Count all images
        all_counts = count_images(dataset_path, classes)
        total_images = sum(all_counts.values())
        
        # Calculate split
        train_size = int(total_images * 0.8)
        val_size = total_images - train_size
        
        structure_type = "single directory (auto-split 80/20)"
        total_train = train_size
        total_val = val_size
        
        print(f"✅ Single directory structure detected!")
        
    else:
        print(f"❌ No valid dataset structure found!")
        return False
    
    print(f"   Structure: {structure_type}")
    print(f"   Classes: {len(classes)}")
    print(f"   Classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training images: {total_train:,}")
    print(f"   Validation images: {total_val:,}")
    print(f"   Total images: {total_train + total_val:,}")
    
    if not auto_split:
        # Show class distribution for pre-split
        train_counts = count_images(os.path.join(dataset_path, 'train'), classes)
        val_counts = count_images(os.path.join(dataset_path, 'val'), classes)
        
        top_train = sorted(train_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n   Top 5 classes (training):")
        for cls, count in top_train:
            print(f"     {cls}: {count:,} images")
    
    return True

def create_sample_structure():
    """Create a sample dataset structure for testing."""
    
    print("\n" + "=" * 60)
    print("Creating Sample Dataset Structure (for testing)")
    print("=" * 60)
    
    sample_classes = ['Tomato___healthy', 'Tomato___Bacterial_spot', 'Potato___healthy']
    
    for split in ['train', 'val']:
        for cls in sample_classes:
            path = os.path.join('plantvillage_dataset', split, cls)
            os.makedirs(path, exist_ok=True)
            
            # Create dummy image files
            for i in range(5):
                dummy_file = os.path.join(path, f'dummy_{i}.jpg')
                with open(dummy_file, 'w') as f:
                    f.write('dummy image data')
    
    print("✅ Sample dataset structure created!")
    print("   This is just for testing. Replace with real dataset.")

if __name__ == "__main__":
    print("Checking PlantVillage dataset...")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--create-sample':
        create_sample_structure()
    else:
        success = check_dataset_structure()
        
        if not success:
            print("\n💡 To create a sample structure for testing:")
            print("   python check_dataset.py --create-sample")
        else:
            print("\n✅ Dataset ready! You can now run:")
            print("   python Normal_MobileNetV2_PlantVillage.py")