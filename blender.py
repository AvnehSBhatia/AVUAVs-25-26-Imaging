#!/usr/bin/env python3
"""
Image Blending Pipeline - Step 1
================================

This script blends runway images with background textures to create
varied training data for the YOLO model. It processes images in parallel
for efficiency and creates quartered versions for data augmentation.

Author: UAVS25_26 Team
"""

import os
import cv2
import numpy as np
from pathlib import Path
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration constants
BLEND_PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
MAX_WORKERS = 12
SUPPORTED_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.webp']


def blend_images(img1, img2, blend_percentage):
    """
    Blend two images with random patterns for realistic training data.
    
    Args:
        img1: First image (runway)
        img2: Second image (background texture)
        blend_percentage: Blend ratio (0-100, where 0 = 100% img1, 100 = 100% img2)
    
    Returns:
        Blended image as numpy array
    """
    # Ensure both images have the same dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    height, width = img1.shape[:2]
    base_alpha = blend_percentage / 100.0
    
    # Create alpha mask with random patterns
    alpha_mask = np.random.random((height, width)) * 0.3
    
    # Add random blobs for natural blending
    num_blobs = np.random.randint(2, 5)
    for _ in range(num_blobs):
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        radius = np.random.randint(min(width, height) // 6, min(width, height) // 3)
        intensity = np.random.uniform(0.4, 0.8)
        
        # Create blob using Gaussian distribution
        y, x = np.ogrid[:height, :width]
        distance_sq = (x - center_x)**2 + (y - center_y)**2
        blob = intensity * np.exp(-distance_sq / (2 * radius**2))
        alpha_mask += blob
    
    # Add random streaks for texture variation
    num_streaks = np.random.randint(1, 4)
    for _ in range(num_streaks):
        start_x = np.random.randint(0, width)
        start_y = np.random.randint(0, height)
        end_x = np.random.randint(0, width)
        end_y = np.random.randint(0, height)
        thickness = np.random.randint(3, 10)
        intensity = np.random.uniform(0.3, 0.7)
        
        line_mask = np.zeros((height, width))
        cv2.line(line_mask, (start_x, start_y), (end_x, end_y), intensity, thickness)
        alpha_mask += line_mask
    
    # Normalize and apply final blend
    alpha_mask = np.clip(alpha_mask, 0, 1)
    alpha_mask = alpha_mask * base_alpha
    
    # Vectorized blending for performance
    alpha_mask_3d = alpha_mask[:, :, np.newaxis]
    blended = img1 * (1 - alpha_mask_3d) + img2 * alpha_mask_3d
    
    return blended.astype(np.uint8)


def load_image(image_path):
    """
    Load an image and handle different formats including WebP.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Loaded image as numpy array or None if failed
    """
    # Handle WebP files with PIL fallback
    if str(image_path).lower().endswith('.webp'):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                from PIL import Image
                pil_img = Image.open(image_path).convert('RGB')
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    else:
        img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    return img


def cut_image_into_quarters(img):
    """
    Cut an image into 4 equal quarters for data augmentation.
    
    Args:
        img: Input image
    
    Returns:
        Tuple of 4 quarter images (top_left, top_right, bottom_left, bottom_right)
    """
    height, width = img.shape[:2]
    mid_h, mid_w = height // 2, width // 2
    
    # Create 4 quarters
    top_left = img[:mid_h, :mid_w]
    top_right = img[:mid_h, mid_w:]
    bottom_left = img[mid_h:, :mid_w]
    bottom_right = img[mid_h:, mid_w:]
    
    return top_left, top_right, bottom_left, bottom_right


def process_combination(args):
    """
    Process a single combination of runway image, background image, and blend percentage.
    
    Args:
        args: Tuple containing (runway_img, bg_img, runway_name, bg_name, 
              blend_pct, output_dir, yolo_dir)
    
    Returns:
        List of saved filenames
    """
    runway_img, bg_img, runway_name, bg_name, blend_pct, output_dir, yolo_dir = args
    
    # Blend images
    blended_img = blend_images(runway_img, bg_img, blend_pct)
    
    # Cut the blended image into 4 quarters
    quarters = cut_image_into_quarters(blended_img)
    quarter_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    saved_files = []
    
    # Save each quarter
    for i, (quarter_img, quarter_name) in enumerate(zip(quarters, quarter_names)):
        # Create output filename for quarter
        output_filename = f"{runway_name}_blended_{bg_name}_{blend_pct}pct_{quarter_name}.png"
        output_path = output_dir / output_filename
        
        # Save quarter image
        cv2.imwrite(str(output_path), quarter_img)
        
        # Create blank YOLO annotation file for quarter
        yolo_filename = f"{runway_name}_blended_{bg_name}_{blend_pct}pct_{quarter_name}.txt"
        yolo_path = yolo_dir / yolo_filename
        yolo_path.write_text("")  # Create empty file
        
        saved_files.append(output_filename)
    
    return saved_files


def load_all_images(directory_path):
    """
    Load all images from a directory with progress tracking.
    
    Args:
        directory_path: Path to directory containing images
    
    Returns:
        List of tuples (image, filename_stem)
    """
    loaded_images = []
    
    for ext in SUPPORTED_EXTENSIONS:
        for img_path in directory_path.glob(ext):
            img = load_image(img_path)
            if img is not None:
                loaded_images.append((img, img_path.stem))
    
    return loaded_images


def main():
    """Main function to run the image blending pipeline."""
    print("Starting Image Blending Pipeline...")
    print("=" * 50)
    
    # Setup directories
    runway_dir = Path("runway")
    bg_mixer_dir = Path("bg_mixer")
    output_dir = Path("blended_images")
    yolo_dir = Path("yolo_text")
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    yolo_dir.mkdir(exist_ok=True)
    
    # Load all images
    print("Loading runway images...")
    loaded_runway_images = load_all_images(runway_dir)
    print(f"Loaded {len(loaded_runway_images)} runway images")
    
    print("Loading background images...")
    loaded_bg_images = load_all_images(bg_mixer_dir)
    print(f"Loaded {len(loaded_bg_images)} background images")
    
    # Create all combinations
    print("Creating image combinations...")
    combinations = []
    for runway_img, runway_name in loaded_runway_images:
        for bg_img, bg_name in loaded_bg_images:
            for blend_pct in BLEND_PERCENTAGES:
                combinations.append((runway_img, bg_img, runway_name, bg_name, 
                                   blend_pct, output_dir, yolo_dir))
    
    total_combinations = len(combinations)
    print(f"Total combinations to process: {total_combinations}")
    
    # Process with multi-threading
    print("Starting parallel processing...")
    completed_count = 0
    total_files_created = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_combination = {
            executor.submit(process_combination, combo): combo 
            for combo in combinations
        }
        
        # Process completed tasks
        for future in as_completed(future_to_combination):
            completed_count += 1
            result = future.result()
            if result:
                total_files_created += len(result)  # Each combination creates 4 files
            
            # Progress reporting
            if completed_count % 10 == 0:
                elapsed_time = time.time() - start_time
                rate = completed_count / elapsed_time if elapsed_time > 0 else 0
                eta = (total_combinations - completed_count) / rate if rate > 0 else 0
                print(f"Progress: {completed_count}/{total_combinations} "
                      f"({completed_count/total_combinations*100:.1f}%) - "
                      f"Files: {total_files_created} - Rate: {rate:.1f}/s - ETA: {eta:.1f}s")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("BLENDING PIPELINE COMPLETE!")
    print("=" * 50)
    print(f"Created {total_files_created} image files and annotation files")
    print(f"Images saved to: '{output_dir}'")
    print(f"YOLO annotations saved to: '{yolo_dir}'")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average rate: {completed_count/total_time:.1f} combinations/second")


if __name__ == "__main__":
    main() 