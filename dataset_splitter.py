#!/usr/bin/env python3
"""
Dataset Splitting Pipeline - Step 3
===================================

This script splits the generated overlay images and background images into
train/validation/test sets for YOLO training. It creates the proper directory
structure and generates the dataset.yaml configuration file.

Author: UAVS25_26 Team
"""

import os
import shutil
import random
from pathlib import Path

# Configuration constants
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SEED = 42
SUPPORTED_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg']


def create_yolo_dataset_structure():
    """
    Create the standard YOLO dataset directory structure.
    
    Returns:
        Path to the base dataset directory
    """
    base_dir = "dataset"
    
    # Create main directories
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(base_dir, split, subdir), exist_ok=True)
    
    return base_dir


def get_image_files(directory_path):
    """
    Get all image files from a directory.
    
    Args:
        directory_path: Path to directory containing images
    
    Returns:
        List of image file paths
    """
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(Path(directory_path).glob(ext))
    return sorted(image_files)


def split_dataset(raw_bg_dir, yolo_text_dir, overlays_dir, output_dir, 
                  train_ratio=DEFAULT_TRAIN_RATIO, val_ratio=DEFAULT_VAL_RATIO, 
                  test_ratio=DEFAULT_TEST_RATIO, seed=DEFAULT_SEED):
    """
    Split the dataset into train/val/test sets.
    
    Args:
        raw_bg_dir: Directory containing background images
        yolo_text_dir: Directory containing YOLO annotation files
        overlays_dir: Directory containing overlay images
        output_dir: Output directory for the dataset
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.2)
        test_ratio: Ratio of data for testing (default: 0.1)
        seed: Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all overlay files (these are our main dataset files with annotations)
    overlay_files = get_image_files(overlays_dir)
    
    # Get all raw background files (these will be negative samples)
    raw_bg_files = get_image_files(raw_bg_dir)
    
    # Shuffle the files
    random.shuffle(overlay_files)
    random.shuffle(raw_bg_files)
    
    # Calculate split indices for overlays
    total_overlay_files = len(overlay_files)
    train_end = int(total_overlay_files * train_ratio)
    val_end = train_end + int(total_overlay_files * val_ratio)
    
    # Split overlay files
    train_overlays = overlay_files[:train_end]
    val_overlays = overlay_files[train_end:val_end]
    test_overlays = overlay_files[val_end:]
    
    # Split raw background files proportionally
    total_bg_files = len(raw_bg_files)
    bg_train_end = int(total_bg_files * train_ratio)
    bg_val_end = bg_train_end + int(total_bg_files * val_ratio)
    
    train_bgs = raw_bg_files[:bg_train_end]
    val_bgs = raw_bg_files[bg_train_end:bg_val_end]
    test_bgs = raw_bg_files[bg_val_end:]
    
    # Print dataset statistics
    print(f"Dataset Statistics:")
    print(f"   Overlay files: {total_overlay_files}")
    print(f"   Raw background files: {total_bg_files}")
    print(f"   Total files: {total_overlay_files + total_bg_files}")
    print(f"\nSplit Distribution:")
    print(f"   Train: {len(train_overlays)} overlays + {len(train_bgs)} backgrounds")
    print(f"   Val: {len(val_overlays)} overlays + {len(val_bgs)} backgrounds")
    print(f"   Test: {len(test_overlays)} overlays + {len(test_bgs)} backgrounds")
    
    # Process each split
    splits = {
        'train': (train_overlays, train_bgs),
        'val': (val_overlays, val_bgs),
        'test': (test_overlays, test_bgs)
    }
    
    for split_name, (overlay_files, bg_files) in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Process overlay files (with annotations)
        for overlay_file in overlay_files:
            # Get base name without extension
            base_name = overlay_file.stem
            
            # Source files
            overlay_src = overlay_file
            label_src = Path(yolo_text_dir) / f"{base_name}.txt"
            
            # Destination files
            img_dst = Path(output_dir) / split_name / "images" / overlay_file.name
            label_dst = Path(output_dir) / split_name / "labels" / f"{base_name}.txt"
            
            # Copy files
            if overlay_src.exists():
                shutil.copy2(overlay_src, img_dst)
            
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
            else:
                # Create empty label file if no annotations exist
                label_dst.write_text("")
        
        # Process raw background files (negative samples - no objects)
        for bg_file in bg_files:
            # Get base name without extension
            base_name = bg_file.stem
            
            # Source file
            bg_src = bg_file
            
            # Destination files
            img_dst = Path(output_dir) / split_name / "images" / bg_file.name
            label_dst = Path(output_dir) / split_name / "labels" / f"{base_name}.txt"
            
            # Copy image file
            if bg_src.exists():
                shutil.copy2(bg_src, img_dst)
            
            # Create empty label file (no objects in background images)
            label_dst.write_text("")
    
    print(f"\nDataset created successfully in '{output_dir}' directory!")
    
    # Create dataset.yaml file for YOLOv11
    total_train = len(train_overlays) + len(train_bgs)
    total_val = len(val_overlays) + len(val_bgs)
    total_test = len(test_overlays) + len(test_bgs)
    create_dataset_yaml(output_dir, total_train, total_val, total_test)


def create_dataset_yaml(output_dir, train_count, val_count, test_count):
    """
    Create dataset.yaml file for YOLOv11 training.
    
    Args:
        output_dir: Output directory for the dataset
        train_count: Number of training images
        val_count: Number of validation images
        test_count: Number of test images
    """
    
    yaml_content = f"""# YOLOv11 Dataset Configuration
# Generated by dataset_splitter.py

# Dataset paths
path: {os.path.abspath(output_dir)}  # dataset root directory
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes
nc: 1
names:
  0: odlc  # Object Detection and Localization Challenge object

# Dataset statistics
# Train: {train_count} images
# Val: {val_count} images  
# Test: {test_count} images
# Total: {train_count + val_count + test_count} images
"""
    
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml at: {yaml_path}")


def main():
    """Main function to run the dataset splitting pipeline."""
    print("Starting Dataset Splitting Pipeline...")
    print("=" * 50)
    
    # Create dataset structure
    output_dir = create_yolo_dataset_structure()
    
    # Split the dataset
    split_dataset(
        raw_bg_dir="blended_images",
        yolo_text_dir="yolo_text", 
        overlays_dir="overlays",
        output_dir=output_dir,
        train_ratio=DEFAULT_TRAIN_RATIO,
        val_ratio=DEFAULT_VAL_RATIO,
        test_ratio=DEFAULT_TEST_RATIO,
        seed=DEFAULT_SEED
    )
    
    print("\n" + "=" * 50)
    print("DATASET SPLITTING COMPLETE!")
    print("=" * 50)
    print(f"Dataset location: {os.path.abspath(output_dir)}")
    print("\nDirectory structure:")
    print("dataset/")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── val/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── test/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("└── dataset.yaml")
    print("\nYou can now use this dataset for YOLOv11 training!")


if __name__ == "__main__":
    main() 