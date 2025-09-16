#!/bin/bash

# UAVS25_26 YOLO Training Pipeline
# Complete pipeline execution script
# Author: UAVS25_26 Team

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a directory exists
directory_exists() {
    [ -d "$1" ]
}

# Function to check if a file exists
file_exists() {
    [ -f "$1" ]
}

# Function to run a step with error handling
run_step() {
    local step_name="$1"
    local command="$2"
    
    print_status "Starting $step_name..."
    echo "=================================================="
    
    if eval "$command"; then
        print_success "$step_name completed successfully!"
        echo ""
    else
        print_error "$step_name failed!"
        exit 1
    fi
}

# Function to copy best model to root
copy_best_model() {
    local source_path="runs/detect/train/weights/best.pt"
    local dest_path="best.pt"
    
    if file_exists "$source_path"; then
        print_status "Copying best model to root directory..."
        cp "$source_path" "$dest_path"
        print_success "best.pt copied to root directory"
    else
        print_warning "best.pt not found in training output"
        print_warning "Training may have failed or model path is different"
    fi
}

# Main pipeline execution
main() {
    echo "UAVS25_26 YOLO Training Pipeline"
    echo "=================================================="
    echo ""
    
    # Check Python installation
    if ! command_exists python3; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    print_success "Python 3 found: $(python3 --version)"
    
    # Check if required directories exist
    print_status "Checking required input directories..."
    
    if ! directory_exists "runway"; then
        print_error "runway/ directory not found"
        exit 1
    fi
    
    if ! directory_exists "bg_mixer"; then
        print_error "bg_mixer/ directory not found"
        exit 1
    fi
    
    if ! directory_exists "odlc"; then
        print_error "odlc/ directory not found"
        exit 1
    fi
    
    print_success "All required input directories found"
    echo ""
    
    # Step 1: Image Blending
    run_step "Image Blending Pipeline" "python3 blender.py"
    
    # Step 2: Object Overlay Generation
    run_step "Object Overlay Generation" "python3 overlay_gen.py"
    
    # Step 3: Dataset Splitting
    run_step "Dataset Splitting" "python3 dataset_splitter.py"
    
    # Step 4: YOLO Training
    run_step "YOLO Model Training" "python3 train_yolo.py"
    
    # Copy best model to root directory
    copy_best_model
    
    # Step 5: Inference Testing
    run_step "Model Inference Testing" "python3 test_inference.py"
    
    echo ""
    echo "PIPELINE COMPLETED SUCCESSFULLY!"
    echo "=================================================="
    print_success "All steps completed successfully!"
    echo ""
    print_status "Generated files and directories:"
    echo "   blended_images/     - Blended background images"
    echo "   overlays/          - Object overlay images"
    echo "   yolo_text/         - YOLO annotation files"
    echo "   dataset/           - Organized dataset (train/val/test)"
    echo "   runs/              - Training outputs and logs"
    echo "   inference_results/ - Inference test results"
    echo "  best.pt            - Best trained model (in root)"
    echo ""
    print_status "Next steps:"
    echo "  1. Review training results in runs/detect/train/"
    echo "  2. Check inference results in inference_results/"
    echo "  3. Use best.pt for deployment or further testing"
    echo ""
}

# Run the main function
main "$@" 