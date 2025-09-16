#!/usr/bin/env python3
"""
Model Inference Testing Pipeline - Step 5
=========================================

This script tests the trained YOLO model on the test dataset to evaluate
performance and generate inference results.

Author: UAVS25_26 Team
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
import time
import json

# Configuration constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_TEST_DIR = "dataset/test"
DEFAULT_OUTPUT_DIR = "inference_results"
SUPPORTED_IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']


def load_model(model_path):
    """
    Load the YOLO model from the specified path.
    
    Args:
        model_path: Path to the trained model file
    
    Returns:
        Loaded YOLO model or None if failed
    """
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_image_files(test_dir):
    """
    Get all image files from the test directory.
    
    Args:
        test_dir: Path to test directory
    
    Returns:
        List of image file paths
    """
    test_path = Path(test_dir)
    images_dir = test_path / "images"
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return []
    
    # Get all image files
    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(list(images_dir.glob(ext)))
    
    return sorted(image_files)


def run_inference_on_dataset(model, test_dir, output_dir=None, conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Run inference on all images in the test dataset.
    
    Args:
        model: Loaded YOLO model
        test_dir: Path to test dataset directory
        output_dir: Output directory for results (optional)
        conf_threshold: Confidence threshold for detections
    
    Returns:
        Dictionary containing inference results summary
    """
    
    # Get image files
    image_files = get_image_files(test_dir)
    
    print(f"Found {len(image_files)} images in test dataset")
    
    if len(image_files) == 0:
        print("No images found!")
        return None
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        results_dir = output_path / "results"
        results_dir.mkdir(exist_ok=True)
    
    # Initialize results summary
    results_summary = {
        'total_images': len(image_files),
        'processed_images': 0,
        'total_detections': 0,
        'detections_per_image': [],
        'processing_time': 0,
        'average_confidence': 0,
        'class_counts': {}
    }
    
    print("Starting inference...")
    start_time = time.time()
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
        
        try:
            # Run inference
            results = model(image_path, conf=conf_threshold, verbose=False)
            
            # Process results
            result = results[0]  # Get first result
            detections = len(result.boxes) if result.boxes is not None else 0
            
            results_summary['total_detections'] += detections
            results_summary['detections_per_image'].append({
                'image': image_path.name,
                'detections': detections,
                'confidences': [],
                'classes': []
            })
            
            # Collect confidence scores and classes
            if result.boxes is not None:
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for conf, cls in zip(confidences, classes):
                    results_summary['detections_per_image'][-1]['confidences'].append(float(conf))
                    results_summary['detections_per_image'][-1]['classes'].append(int(cls))
                    
                    # Update class counts
                    cls_name = f"class_{int(cls)}"
                    results_summary['class_counts'][cls_name] = results_summary['class_counts'].get(cls_name, 0) + 1
            
            results_summary['processed_images'] += 1
            
            # Save annotated image if output directory specified
            if output_dir:
                annotated_img = result.plot()
                output_file = results_dir / f"result_{image_path.name}"
                cv2.imwrite(str(output_file), annotated_img)
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
    
    # Calculate summary statistics
    end_time = time.time()
    results_summary['processing_time'] = end_time - start_time
    
    if results_summary['total_detections'] > 0:
        all_confidences = []
        for det in results_summary['detections_per_image']:
            all_confidences.extend(det['confidences'])
        results_summary['average_confidence'] = np.mean(all_confidences)
    
    return results_summary


def print_results_summary(results_summary):
    """
    Print a formatted summary of inference results.
    
    Args:
        results_summary: Dictionary containing inference results
    """
    print("\n" + "=" * 50)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {results_summary['processed_images']}")
    print(f"Total detections: {results_summary['total_detections']}")
    print(f"Average detections per image: {results_summary['total_detections']/results_summary['processed_images']:.2f}")
    print(f"Average confidence: {results_summary['average_confidence']:.3f}")
    print(f"Processing time: {results_summary['processing_time']:.2f} seconds")
    print(f"Images per second: {results_summary['processed_images']/results_summary['processing_time']:.2f}")
    
    print("\nClass distribution:")
    for cls, count in results_summary['class_counts'].items():
        print(f"   {cls}: {count} detections")


def save_results_to_json(results_summary, output_dir):
    """
    Save detailed results to JSON file.
    
    Args:
        results_summary: Dictionary containing inference results
        output_dir: Output directory for results
    """
    if output_dir:
        results_file = Path(output_dir) / "inference_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")


def main():
    """Main function to run the inference testing pipeline."""
    print("YOLO Model Inference Testing")
    print("=" * 30)
    
    # Configuration
    model_path = "best.pt"  # Path to your trained model
    test_dir = DEFAULT_TEST_DIR  # Path to test dataset
    output_dir = DEFAULT_OUTPUT_DIR  # Output directory for results
    conf_threshold = DEFAULT_CONFIDENCE_THRESHOLD  # Confidence threshold
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        print("Please ensure 'best.pt' is in the current directory")
        return
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Run inference
    print(f"\nRunning inference on test dataset: {test_dir}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Output directory: {output_dir}")
    
    results = run_inference_on_dataset(model, test_dir, output_dir, conf_threshold)
    
    if results:
        # Print summary
        print_results_summary(results)
        
        # Save detailed results
        save_results_to_json(results, output_dir)
        
        print("\nInference completed successfully!")
        print(f"Results saved to: {output_dir}")
    else:
        print("\nInference failed!")


if __name__ == "__main__":
    main() 