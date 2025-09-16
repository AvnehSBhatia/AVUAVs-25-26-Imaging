# UAVS25_26 YOLO Training Pipeline

A complete pipeline for training YOLO models on custom datasets with automated data generation, augmentation, and training.

## 🎯 Overview

This project implements a comprehensive pipeline for training YOLOv11 models on custom object detection datasets. The pipeline includes:

1. **Image Blending** - Creates varied background textures
2. **Object Overlay** - Places objects on backgrounds with realistic positioning
3. **Dataset Splitting** - Organizes data into train/val/test sets
4. **Model Training** - Trains YOLOv11 with optimized parameters
5. **Inference Testing** - Evaluates model performance

## 📁 Project Structure

```
UAVS25_26/
├── blender.py              # Step 1: Image blending pipeline
├── overlay_gen.py          # Step 2: Object overlay generation
├── dataset_splitter.py     # Step 3: Dataset splitting and organization
├── train_yolo.py          # Step 4: YOLO model training
├── test_inference.py      # Step 5: Model inference testing
├── run_pipeline.sh        # Complete pipeline execution script
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── runway/                # Input runway images
├── bg_mixer/              # Background texture images
├── odlc/                  # ODLC object images (transparent PNGs)
│
├── blended_images/        # Generated blended backgrounds
├── overlays/              # Generated overlay images
├── yolo_text/            # YOLO annotation files
├── dataset/               # Organized dataset (train/val/test)
├── runs/                  # Training outputs
├── inference_results/     # Inference test results
└── best.pt               # Best trained model (copied to root)
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **Required packages** (install via `pip install -r requirements.txt`):
   - `opencv-python`
   - `numpy`
   - `Pillow`
   - `ultralytics`
   - `torch`

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd UAVS25_26

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
# Make the script executable
chmod +x run_pipeline.sh

# Run the complete pipeline
./run_pipeline.sh
```

## 📋 Execution Order

The pipeline must be executed in the following order:

### Step 1: Image Blending (`blender.py`)
```bash
python blender.py
```
- **Purpose**: Blends runway images with background textures
- **Input**: `runway/` + `bg_mixer/` directories
- **Output**: `blended_images/` + `yolo_text/` directories
- **Process**: Creates quartered versions for data augmentation

### Step 2: Object Overlay (`overlay_gen.py`)
```bash
python overlay_gen.py
```
- **Purpose**: Overlays ODLC objects onto blended backgrounds
- **Input**: `blended_images/` + `odlc/` directories
- **Output**: `overlays/` + `yolo_text/` directories
- **Process**: Random positioning, scaling, rotation, and decoy objects

### Step 3: Dataset Splitting (`dataset_splitter.py`)
```bash
python dataset_splitter.py
```
- **Purpose**: Organizes data into train/validation/test sets
- **Input**: `overlays/` + `blended_images/` + `yolo_text/` directories
- **Output**: `dataset/` directory with proper YOLO structure
- **Process**: Creates `dataset.yaml` configuration file

### Step 4: Model Training (`train_yolo.py`)
```bash
python train_yolo.py
```
- **Purpose**: Trains YOLOv11 model on the dataset
- **Input**: `dataset/` directory
- **Output**: `runs/detect/` + `best.pt` (copied to root)
- **Process**: Optimized training with early stopping

### Step 5: Inference Testing (`test_inference.py`)
```bash
python test_inference.py
```
- **Purpose**: Tests the trained model on test data
- **Input**: `best.pt` + `dataset/test/` directory
- **Output**: `inference_results/` directory
- **Process**: Generates performance metrics and annotated images

## ⚙️ Configuration

### Training Parameters (in `train_yolo.py`)
```python
DEFAULT_EPOCHS = 20
DEFAULT_IMAGE_SIZE = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_WORKERS = 4
DEFAULT_PATIENCE = 3
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_MAX_DET = 5
```

### Dataset Splitting (in `dataset_splitter.py`)
```python
DEFAULT_TRAIN_RATIO = 0.7  # 70% training
DEFAULT_VAL_RATIO = 0.2    # 20% validation
DEFAULT_TEST_RATIO = 0.1   # 10% testing
```

### Blending Parameters (in `blender.py`)
```python
BLEND_PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
MAX_WORKERS = 12
```

## 📊 Expected Outputs

### Training Results
- **Model files**: `best.pt`, `last.pt`
- **Training curves**: `runs/detect/train/results.png`
- **Logs**: `runs/detect/train/logs/`

### Inference Results
- **Annotated images**: `inference_results/results/`
- **Performance metrics**: `inference_results/inference_results.json`
- **Summary statistics**: Console output

## 🔧 Customization

### Adding New Objects
1. Place transparent PNG images in the `odlc/` directory
2. Ensure images have alpha channels for proper overlay
3. Re-run the pipeline from Step 2

### Modifying Backgrounds
1. Add new background textures to `bg_mixer/` directory
2. Re-run the pipeline from Step 1

### Adjusting Training Parameters
1. Modify constants in `train_yolo.py`
2. Re-run the pipeline from Step 4

## 🐛 Troubleshooting

### Common Issues

1. **"ultralytics not found"**
   ```bash
   pip install ultralytics
   ```

2. **"dataset/dataset.yaml not found"**
   - Ensure you've run `dataset_splitter.py` first
   - Check that `overlays/` and `blended_images/` directories exist

3. **Memory issues during training**
   - Reduce `DEFAULT_BATCH_SIZE` in `train_yolo.py`
   - Reduce `DEFAULT_WORKERS` in `train_yolo.py`

4. **Slow processing**
   - Reduce `MAX_WORKERS` in `blender.py` and `overlay_gen.py`
   - Set `MAX_BG_IMAGES` in `overlay_gen.py` for testing

### Performance Optimization

- **For testing**: Set `MAX_BG_IMAGES = 100` in `overlay_gen.py`
- **For production**: Set `MAX_BG_IMAGES = None` for all images
- **GPU training**: Change `device=mps` to `device=0` for CUDA

## 📈 Performance Metrics

The pipeline generates comprehensive performance metrics:

- **Training metrics**: Loss curves, mAP scores
- **Inference metrics**: Detection counts, confidence scores
- **Processing speed**: Images per second, total time
- **Dataset statistics**: File counts, split distributions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the pipeline
5. Submit a pull request

---

**Note**: This pipeline is designed for SUAS 2025 - 2026 competition requirements. Adjust parameters and configurations based on your specific use case. 