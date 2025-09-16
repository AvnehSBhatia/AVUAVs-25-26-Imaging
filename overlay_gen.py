#!/usr/bin/env python3
"""
Overlay Generation Pipeline - Step 2
====================================

This script overlays ODLC objects onto blended background images to create
training data with realistic object placements. It includes random positioning,
scaling, rotation, and decoy objects for robust model training.

Author: UAVS25_26 Team
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration constants
MAX_WORKERS = 12
BATCH_SIZE = 100
SUPPORTED_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.webp']
MAX_BG_IMAGES = None  # Set to a number for testing, None for all images


def load_image(image_path):
    """
    Load an image and handle different formats including WebP with alpha channel support.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Loaded image as numpy array or None if failed
    """
    if str(image_path).lower().endswith('.webp'):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                from PIL import Image
                pil_img = Image.open(image_path).convert('RGBA')
                img = np.array(pil_img)
                if img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                else:  # RGB
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    else:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    return img


def create_decoy_shape(width, height, shape_type='random'):
    """
    Create a decoy shape (gray or green) to add noise to training data.
    
    Args:
        width: Width of the decoy shape
        height: Height of the decoy shape
        shape_type: Type of shape ('circle', 'rectangle', 'triangle', 'random')
    
    Returns:
        Decoy shape as numpy array with alpha channel
    """
    if shape_type == 'random':
        shape_type = random.choice(['circle', 'rectangle', 'triangle'])
    
    # Create transparent canvas
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    
    if shape_type == 'circle':
        center = (width // 2, height // 2)
        radius = min(width, height) // 3
        color = random.choice([(128, 128, 128, 180), (0, 128, 0, 180)])  # Gray or green
        cv2.circle(canvas, center, radius, color, -1)
    
    elif shape_type == 'rectangle':
        x1, y1 = width // 4, height // 4
        x2, y2 = 3 * width // 4, 3 * height // 4
        color = random.choice([(128, 128, 128, 180), (0, 128, 0, 180)])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
    
    elif shape_type == 'triangle':
        pts = np.array([
            [width // 2, height // 4],
            [width // 4, 3 * height // 4],
            [3 * width // 4, 3 * height // 4]
        ], np.int32)
        color = random.choice([(128, 128, 128, 180), (0, 128, 0, 180)])
        cv2.fillPoly(canvas, [pts], color)
    
    return canvas


def rotate_image(image, angle):
    """
    Rotate image by angle degrees with transparent background support.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
    
    Returns:
        Rotated image or None if failed
    """
    if image is None or abs(angle) < 1:  # Skip rotation if angle is very small
        return image
    
    # Ensure image has alpha channel
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image dimensions
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))
    
    # Adjust rotation matrix
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Perform rotation with better interpolation and border handling
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    
    # Clean up rotation artifacts by applying alpha threshold
    alpha = rotated[:, :, 3]
    alpha_mask = alpha > 30  # Higher threshold to remove artifacts
    
    # Apply mask to all channels
    for i in range(4):
        rotated[:, :, i] = rotated[:, :, i] * alpha_mask
    
    # Crop out transparent borders to get tight bounding box
    coords = np.where(alpha > 30)  # Use same threshold for cropping
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        rotated = rotated[y_min:y_max+1, x_min:x_max+1]
    
    return rotated


def overlay_object(background, obj_img, x, y, scale_factor, rotation_angle, blur=False, pseudo_blend=False):
    """
    Overlay an object onto background with scaling, rotation, and optional effects.
    
    Args:
        background: Background image
        obj_img: Object image to overlay
        x, y: Position coordinates
        scale_factor: Scale factor for the object
        rotation_angle: Rotation angle in degrees
        blur: Whether to apply blur effect
        pseudo_blend: Whether to apply pseudo-blending effect
    
    Returns:
        Tuple of (modified_background, bounding_box)
    """
    if obj_img is None:
        return background, None
    
    # Ensure object has alpha channel
    if obj_img.shape[2] == 3:
        obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2BGRA)
    
    # Store original dimensions for bounding box calculation
    original_height, original_width = obj_img.shape[:2]
    
    # Scale the object
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    scaled_obj = cv2.resize(obj_img, (new_width, new_height))
    
    # Rotate the object
    if rotation_angle != 0:
        scaled_obj = rotate_image(scaled_obj, rotation_angle)
        if scaled_obj is None:
            return background, None
    
    # Apply blur if requested
    if blur:
        # Blur only RGB channels, keep alpha intact
        rgb = scaled_obj[:, :, :3]
        blurred_rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
        scaled_obj[:, :, :3] = blurred_rgb
    
    # Apply pseudo-blending if requested
    if pseudo_blend:
        # Reduce alpha channel for blending effect
        alpha = scaled_obj[:, :, 3].astype(np.float32) / 255.0
        alpha = alpha * 0.7  # Make it more transparent
        scaled_obj[:, :, 3] = (alpha * 255).astype(np.uint8)
    
    # Get final object dimensions after all transformations
    obj_height, obj_width = scaled_obj.shape[:2]
    
    # Check if object fits within background
    bg_height, bg_width = background.shape[:2]
    if x < 0 or y < 0 or x + obj_width > bg_width or y + obj_height > bg_height:
        return background, None
    
    # Optimized alpha blending for transparent backgrounds
    alpha = scaled_obj[:, :, 3].astype(np.float32) / 255.0
    
    # Apply threshold to prevent artifacts from very low alpha values
    alpha = np.where(alpha < 0.1, 0, alpha)  # Remove very transparent pixels
    
    alpha = np.stack([alpha] * 3, axis=2)
    
    # Extract RGB channels
    obj_rgb = scaled_obj[:, :, :3]
    
    # Blend with background using alpha
    roi = background[y:y+obj_height, x:x+obj_width]
    blended = roi * (1 - alpha) + obj_rgb * alpha
    background[y:y+obj_height, x:x+obj_width] = blended.astype(np.uint8)
    
    # Calculate tight bounding box based on actual visible content
    alpha_channel = scaled_obj[:, :, 3]
    coords = np.where(alpha_channel > 15)  # Higher threshold for cleaner bounds
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        # Adjust bounding box to actual visible content
        bbox = [x + x_min, y + y_min, x + x_max + 1, y + y_max + 1]
    else:
        # Fallback to full object bounds if no visible content found
        bbox = [x, y, x + obj_width, y + obj_height]
    
    return background, bbox


def check_overlap(bbox1, bbox2, min_distance=50):
    """
    Check if two bounding boxes overlap or are too close.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        min_distance: Minimum distance between object centers
    
    Returns:
        True if objects overlap or are too close, False otherwise
    """
    if bbox1 is None or bbox2 is None:
        return False
    
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # Check for overlap
    if not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1):
        return True
    
    # Check for minimum distance
    center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
    center2 = ((x3 + x4) // 2, (y3 + y4) // 2)
    distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    return distance < min_distance


def generate_yolo_annotation(bboxes, image_width, image_height):
    """
    Generate YOLO format annotation string.
    
    Args:
        bboxes: List of bounding boxes [x1, y1, x2, y2]
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        YOLO format annotation string
    """
    yolo_lines = []
    
    for bbox in bboxes:
        if bbox is None:
            continue
        
        x1, y1, x2, y2 = bbox
        
        # Convert to YOLO format (center_x, center_y, width, height) - normalized
        center_x = (x1 + x2) / 2 / image_width
        center_y = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        # Class 0 for ODLC objects
        yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    return "\n".join(yolo_lines)


def process_overlay(args):
    """
    Process a single overlay operation.
    
    Args:
        args: Tuple containing (bg_img, bg_name, odlc_objects, output_dir, yolo_dir)
    
    Returns:
        Output filename
    """
    bg_img, bg_name, odlc_objects, output_dir, yolo_dir = args
    
    # Create a copy of the background
    result_img = bg_img.copy()
    bg_height, bg_width = bg_img.shape[:2]
    total_pixels = bg_width * bg_height
    
    # Random number of objects (1-4)
    num_objects = random.randint(1, 4)
    
    # Random number of decoys (1-2)
    num_decoys = random.randint(1, 2)
    
    bboxes = []
    
    # Place ODLC objects
    placed_objects = []
    for i in range(num_objects):
        # Select random object
        obj_img, obj_name = random.choice(odlc_objects)
        
        # Calculate target area (0.7% to 4% of total pixels)
        target_area_ratio = random.uniform(0.007, 0.04)
        target_area = total_pixels * target_area_ratio
        
        # Calculate scale factor based on object's original area
        obj_height, obj_width = obj_img.shape[:2]
        original_area = obj_width * obj_height
        scale_factor = math.sqrt(target_area / original_area)
        
        # Random rotation
        rotation = random.uniform(0, 360)
        
        # Random blur
        blur = random.choice([True, False])
        
        # Random pseudo-blend
        pseudo_blend = random.choice([True, False])
        
        # Try to find a position without overlap
        max_attempts = 30
        placed = False
        
        for attempt in range(max_attempts):
            # Random position
            x = random.randint(0, bg_width - int(obj_width * scale_factor))
            y = random.randint(0, bg_height - int(obj_height * scale_factor))
            
            # Calculate bbox for overlap check
            scaled_width = int(obj_width * scale_factor)
            scaled_height = int(obj_height * scale_factor)
            bbox = [x, y, x + scaled_width, y + scaled_height]
            
            # Check overlap with existing objects
            overlap = False
            for existing_bbox in placed_objects:
                if check_overlap(bbox, existing_bbox):
                    overlap = True
                    break
            
            if not overlap:
                # Place the object
                result_img, final_bbox = overlay_object(result_img, obj_img, x, y, 
                                                      scale_factor, rotation, blur, pseudo_blend)
                if final_bbox is not None:
                    bboxes.append(final_bbox)
                    placed_objects.append(final_bbox)
                    placed = True
                    break
        
        if not placed:
            # Just place it anywhere if we can't find a good spot
            x = random.randint(0, bg_width - int(obj_width * scale_factor))
            y = random.randint(0, bg_height - int(obj_height * scale_factor))
            result_img, final_bbox = overlay_object(result_img, obj_img, x, y, 
                                                  scale_factor, rotation, blur, pseudo_blend)
            if final_bbox is not None:
                bboxes.append(final_bbox)
    
    # Place decoys
    for i in range(num_decoys):
        # Random decoy size
        decoy_width = random.randint(20, 60)
        decoy_height = random.randint(20, 60)
        
        # Create decoy
        decoy = create_decoy_shape(decoy_width, decoy_height)
        
        # Just place it randomly without complex overlap checking
        x = random.randint(0, bg_width - decoy_width)
        y = random.randint(0, bg_height - decoy_height)
        result_img, _ = overlay_object(result_img, decoy, x, y, 1.0, 0, False, False)
    
    # Save the result image
    output_filename = f"overlay_{bg_name}.png"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), result_img)
    
    # Generate and save YOLO annotation
    yolo_content = generate_yolo_annotation(bboxes, bg_width, bg_height)
    yolo_filename = f"overlay_{bg_name}.txt"
    yolo_path = yolo_dir / yolo_filename
    yolo_path.write_text(yolo_content)
    
    return output_filename


def load_images_in_batches(directory_path, max_images=None):
    """
    Load images in batches with progress tracking.
    
    Args:
        directory_path: Path to directory containing images
        max_images: Maximum number of images to load (None for all)
    
    Returns:
        List of tuples (image, filename_stem)
    """
    # Get all image files first
    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(list(directory_path.glob(ext)))
    
    # Limit number of files if specified
    if max_images is not None:
        all_files = all_files[:max_images]
        print(f"Limited to {len(all_files)} image files for testing")
    else:
        print(f"Found {len(all_files)} image files")
    
    # Load images in batches
    loaded_images = []
    for i in range(0, len(all_files), BATCH_SIZE):
        batch = all_files[i:i+BATCH_SIZE]
        print(f"Loading batch {i//BATCH_SIZE + 1}/{(len(all_files) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} images)")
        
        for img_path in batch:
            img = load_image(img_path)
            if img is not None:
                loaded_images.append((img, img_path.stem))
    
    return loaded_images


def main():
    """Main function to run the overlay generation pipeline."""
    print("Starting Overlay Generation Pipeline...")
    print("=" * 50)
    
    # Setup directories
    odlc_dir = Path("odlc")
    blended_dir = Path("blended_images")
    output_dir = Path("overlays")
    yolo_dir = Path("yolo_text")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load ODLC objects
    print("Loading ODLC objects...")
    odlc_objects = []
    for ext in SUPPORTED_EXTENSIONS:
        for obj_path in odlc_dir.glob(ext):
            obj_img = load_image(obj_path)
            if obj_img is not None:
                odlc_objects.append((obj_img, obj_path.stem))
    
    print(f"Loaded {len(odlc_objects)} ODLC objects")
    
    # Load blended background images
    print("Loading blended background images...")
    bg_images = load_images_in_batches(blended_dir, MAX_BG_IMAGES)
    print(f"Successfully loaded {len(bg_images)} background images")
    
    # Create all combinations
    print("Creating overlay combinations...")
    combinations = []
    for bg_img, bg_name in bg_images:
        combinations.append((bg_img, bg_name, odlc_objects, output_dir, yolo_dir))
    
    total_combinations = len(combinations)
    print(f"Total combinations to process: {total_combinations}")
    
    # Process with multi-threading
    print("Starting parallel processing...")
    completed_count = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_combination = {
            executor.submit(process_overlay, combo): combo 
            for combo in combinations
        }
        
        # Process completed tasks
        for future in as_completed(future_to_combination):
            completed_count += 1
            if completed_count % 10 == 0:
                elapsed_time = time.time() - start_time
                rate = completed_count / elapsed_time if elapsed_time > 0 else 0
                eta = (total_combinations - completed_count) / rate if rate > 0 else 0
                print(f"Progress: {completed_count}/{total_combinations} "
                      f"({completed_count/total_combinations*100:.1f}%) - "
                      f"Rate: {rate:.1f}/s - ETA: {eta:.1f}s")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("OVERLAY GENERATION COMPLETE!")
    print("=" * 50)
    print(f"Created {completed_count} overlay images")
    print(f"Images saved to: '{output_dir}'")
    print(f"YOLO annotations saved to: '{yolo_dir}'")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average rate: {completed_count/total_time:.1f} images/second")


if __name__ == "__main__":
    main() 