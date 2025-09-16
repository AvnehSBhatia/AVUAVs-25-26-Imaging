#!/usr/bin/env python3
"""
YOLO Training Pipeline - Step 4
===============================

This script trains a YOLOv11 model on the generated dataset. It uses optimized
parameters to ensure stable training and good performance.

Author: UAVS25_26 Team
"""

import subprocess
import sys
import os
from pathlib import Path

# Configuration constants
DEFAULT_EPOCHS = 20
DEFAULT_IMAGE_SIZE = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_WORKERS = 4
DEFAULT_PATIENCE = 3
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_MAX_DET = 5


def check_requirements():
    """
    Check if required packages are installed.
    
    Returns:
        True if requirements are met, False otherwise
    """
    try:
        import ultralytics
        print(f"ultralytics version: {ultralytics.__version__}")
        return True
    except ImportError:
        print("ultralytics not installed!")
        print("Install with: pip install ultralytics")
        return False


def check_dataset_exists():
    """
    Check if the dataset directory and configuration file exist.
    
    Returns:
        True if dataset exists, False otherwise
    """
    dataset_yaml = Path("dataset/dataset.yaml")
    if not dataset_yaml.exists():
        print("Error: dataset/dataset.yaml not found!")
        print("Please run dataset_splitter.py first to create the dataset.")
        return False
    
    if not Path("dataset").exists():
        print("Dataset directory not found!")
        print("Please run dataset_splitter.py first to create the dataset.")
        return False
    
    return True


def run_yolo_training():
    """
    Run YOLOv11 training with optimized parameters.
    
    Returns:
        True if training completed successfully, False otherwise
    """
    
    # Check if dataset.yaml exists
    dataset_yaml = Path("dataset/dataset.yaml")
    if not dataset_yaml.exists():
        print("Error: dataset/dataset.yaml not found!")
        print("Please run dataset_splitter.py first to create the dataset.")
        return False
    
    # YOLO training command with optimized parameters
    cmd = [
        "yolo",
        "task=detect",
        "mode=train", 
        "model=yolo11n.pt",
        f"data={dataset_yaml}",
        f"epochs={DEFAULT_EPOCHS}",
        f"imgsz={DEFAULT_IMAGE_SIZE}",
        "device=mps",  # Use MPS for Apple Silicon
        "project=runs",
        "name=detect",
        # Training optimization parameters
        f"batch={DEFAULT_BATCH_SIZE}",
        f"workers={DEFAULT_WORKERS}",
        f"patience={DEFAULT_PATIENCE}",
        f"conf={DEFAULT_CONFIDENCE}",
        f"iou={DEFAULT_IOU}",
        f"max_det={DEFAULT_MAX_DET}"
    ]
    
    print("Starting YOLOv11 Training with optimizations...")
    print(f"Command: {' '.join(cmd)}")
    print("\n" + "=" * 60)
    
    try:
        # Run the training command
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("\nError: 'yolo' command not found!")
        print("Please install ultralytics: pip install ultralytics")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False


def copy_best_model():
    """
    Copy the best model from training output to the root directory.
    
    Returns:
        True if copy was successful, False otherwise
    """
    best_model_path = Path("runs/detect/train/weights/best.pt")
    
    if best_model_path.exists():
        try:
            import shutil
            shutil.copy2(best_model_path, "best.pt")
            print("Copied best.pt to root directory")
            return True
        except Exception as e:
            print(f"Error copying best.pt: {e}")
            return False
    else:
        print("best.pt not found in training output")
        return False


def main():
    """Main function to run the YOLO training pipeline."""
    print("YOLOv11 Training Script (Optimized)")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check if dataset exists
    if not check_dataset_exists():
        return
    
    # Run training
    success = run_yolo_training()
    
    if success:
        print("\nTraining completed! Check the 'runs/detect' folder for results.")
        
        # Copy best model to root
        copy_best_model()
        
        print("\nTraining artifacts:")
        print("   - runs/detect/train/weights/best.pt (best model)")
        print("   - runs/detect/train/weights/last.pt (last checkpoint)")
        print("   - runs/detect/train/results.png (training curves)")
        print("   - best.pt (copied to root directory)")
    else:
        print("\nTraining failed. Check the error messages above.")


if __name__ == "__main__":
    main() 