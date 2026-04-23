---
title: Fish Density Monitoring System
emoji: 🐟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: apache-2.0
python_version: "3.11"
---


# Fish Density Monitoring System


A real-time fish instance segmentation and zonal density monitoring system powered by a **Physics-Informed Attention U-Net (PIAU-Net)**, with a Gradio web interface, Grad-CAM explainability, and temporal video analysis.

---

## Problem Statement

Monitoring fish populations in underwater environments is critical for marine ecology research, aquaculture management, and conservation efforts. Manual counting is slow, subjective, and impractical at scale. Existing computer vision pipelines applied to underwater footage suffer from:

- **Massive false-positive noise** — coral, water turbulence, and sunlight scatter trigger incorrect detections
- **Uniform mask flooding** — poorly calibrated models assign high fish-class probability to entire frames including open water and sky
- **Ghost bounding boxes** — tiny noise artifacts produce hundreds of phantom detections with no corresponding fish
- **Incorrect class identification** — inverted class-channel mapping causes the model to segment water instead of fish
- **Fragmented instance masks** — edge-slicing post-processing splits a single fish blob into dozens of phantom instances
- **Missing small fish** — single-scale inference at 256×256 resolution loses spatial detail for small or distant fish
- **Visualization failures** — heatmaps rendered on binary masks produce uniform color floods that obscure actual detections

---

## Solution

The system uses **PIAU-Net** — a Physics-Informed Attention U-Net trained on the AquaOV255 underwater dataset — combined with a robust post-processing and visualization pipeline designed specifically for complex reef environments.

### Key Fixes Applied

| Problem | Root Cause | Fix |
|---|---|---|
| Entire frame masked as fish | Fish class channel was `1` (water), not `0` (fish) | Corrected `fish_class = 0` based on observed model output |
| Uniform yellow flood | Fixed threshold (0.15) too low for poorly calibrated model | **Adaptive confidence-margin threshold**: `fish_prob − bg_prob > 0.02` |
| Ghost boxes everywhere | `MIN_AREA = 5` kept pixel-level noise | Raised to `MIN_AREA = 150` pixels |
| Fish fragmented into tiny pieces | Canny edge-slicing divided every blob | Removed entirely |
| Green/teal fish missed | HSV "green water" filter deleted valid fish | Removed colour-based filter |
| Small fish not detected | Single-scale 256×256 inference misses fine details | **MAX-based sliding window**: 4 overlapping 60%-patches, `max()` not `mean()` |
| Masks patchy / no fill visible | MORPH_OPEN left interior holes in blobs | Added `MORPH_CLOSE` (5×5) to fill within-fish holes |
| Heatmap floods entire frame | Heatmap applied to binary mask (0/1 → uniform JET color) | Switched to `prob_map` (continuous gradient) |
| Input distribution mismatch | Inference used ImageNet stats; model trained with `mean=0.5, std=0.5` | Corrected normalization constants |
| `WinError 10054` console noise | Windows asyncio fires on browser tab close | Custom `logging.Filter` on the `asyncio` logger |
| Temporal video saved to `results/` | Hardcoded output path | Writes to OS `tempfile`, auto-cleaned by OS |

---

## Architecture

```
Fish Density Monitoring System
│
├── app.py                          # Gradio web interface & orchestration
├── config.py                       # Global hyperparameters
│
├── pipeline/
│   ├── inference.py                # Single-frame & sliding-window inference
│   ├── mask_processing.py          # Confidence-margin thresholding + morphology
│   ├── visualization.py            # Instance mask overlay, heatmap, zone grid
│   ├── density.py                  # Per-zone fish density computation
│   ├── instance.py                 # Instance ID assignment
│   ├── post_processing.py          # Bounding box NMS, tracking
│   ├── gradcam.py                  # Grad-CAM engine (SegGradCAM)
│   ├── explain.py                  # Image-level XAI explanation
│   ├── temporal_gradcam.py         # Frame-by-frame temporal CAM for video
│   ├── xai_visualization.py        # CAM overlay rendering
│   └── integration.py              # Full pipeline integration
│
└── PIAUNet/
    └── model/                      # PhysicsInformedAttentionUNet definition
```

### Model: PhysicsInformedAttentionUNet (PIAU-Net)

- **Input:** RGB image → resized to 256×256, normalized with `mean=0.5, std=0.5`
- **Output:** 2-channel logit map `[C, H, W]`
  - Channel 0 → P(fish)
  - Channel 1 → P(background/water)
- **Training dataset:** AquaOV255 (binary masks: pixel=0 → background, pixel=1 → fish)
- **Weights:** `weights/best_model.pth`

---

## Detection Pipeline (Post-Processing)

```
Raw logits (2×H×W)
    │
    ▼ softmax
fish_prob = probs[0],  bg_prob = probs[1]
    │
    ▼ confidence margin
confidence = fish_prob − bg_prob          # pixel votes fish over background
mask = (confidence > 0.02)               # small positive margin gate
    │
    ▼ resize to original resolution
    │
    ▼ temporal voting (3-frame window, majority > 0.5)
    │
    ▼ morphological cleanup
MORPH_OPEN  (3×3)   → remove noise specks
MORPH_CLOSE (5×5)   → fill holes within fish blobs
    │
    ▼ connected-component filtering
keep if  150 ≤ area ≤ 25% of image       # not noise, not open-water flood
keep if  aspect ratio ≤ 5:1              # not a horizontal sky/water band
    │
    ▼ instance mask (clean binary)
```

### Sliding Window Inference (High-Res Mode)

For small or distant fish, single-scale 256×256 inference loses spatial detail.  
**Sliding window** divides the frame into 4 overlapping 60%-sized patches, runs inference on each, then aggregates with **MAX** (not mean):

```
pixel_logit = max(patch1_logit, patch2_logit, patch3_logit, patch4_logit)
```

MAX aggregation ensures a small fish strongly detected in one patch is **never diluted** by the other patches where that pixel maps to open water.

---

## Features

| Feature | Description |
|---|---|
| **Instance Segmentation** | Color-coded per-fish masks with bounding boxes |
| **Zonal Density Grid** | Configurable N×M grid with per-zone fish counts and density levels |
| **Density Alerts** | Highlights zones exceeding configurable alert threshold (LOW / MEDIUM / HIGH) |
| **Grad-CAM XAI** | Gradient-based attention map showing which pixels drove the fish prediction |
| **Temporal Video CAM** | Frame-by-frame model attention visualization for uploaded videos |
| **Summary Panel** | Total fish count, density statistics overlay |
| **Density Heatmap** | Continuous probability gradient overlay (NOT binary mask) |
| **High-Res Sliding Window** | Optional 4-patch overlapping inference for small fish detection |

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `DEVICE` | `"cuda"` | Inference device (`"cuda"` or `"cpu"`) |
| `IMG_SIZE` | `256` | Model input resolution |
| `THRESHOLD` | `0.50` | Hard minimum confidence floor |
| `MIN_AREA` | `150` | Minimum blob area in pixels (filters noise) |
| `IOU_THRESHOLD` | `0.6` | IoU threshold for NMS |
| `ZONAL_GRID_ROWS` | `5` | Number of zone grid rows |
| `ZONAL_GRID_COLS` | `5` | Number of zone grid columns |
| `ZONAL_ALERT_THRESHOLD` | `"MEDIUM-HIGH"` | Zone density level that triggers alerts |

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- Anaconda / Miniconda

### 1. Create environment

```bash
conda create -n pytorch_env python=3.10
conda activate pytorch_env
```

### 2. Install PyTorch (CUDA)

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies

```bash
pip install opencv-python>=4.6.0 Pillow>=9.0.0 numpy>=1.23.0 pandas>=1.5.0
pip install matplotlib>=3.6.0 albumentations>=1.3.0 tqdm>=4.64.0
pip install gradio
```

### 4. Place model weights

```
weights/
└── best_model.pth      ← PIAU-Net trained weights
```

---

## Running the Application

```bash
conda activate pytorch_env
python app.py
```

Then open your browser at: **http://localhost:7860**

---

## Usage

1. **Upload** an image or video using the _Media Input_ panel
2. **Configure** zone grid rows/columns and alert threshold
3. Toggle **High-Res Sliding Window** for better small-fish detection (slower)
4. Toggle **Show Density Heatmap** and **Show Zone Grid** as needed
5. Enable **Image Grad-CAM** for explainability visualization
6. Enable **Video Temporal CAM** for frame-by-frame attention on videos
7. Click **Analyze Data**

### Outputs

| Output Panel | Contents |
|---|---|
| Segmentation Output | Instance-masked image with color-coded bounding boxes |
| Detailed Report | Fish count, zone density breakdown, alert summary |
| Zone Summary Legend | Per-zone density level and count table |
| Explainable AI (Image) | Grad-CAM attention heatmap with interpretation text |
| Explainable AI (Video) | Temporal attention video (3-panel: original / heatmap / overlay) |

---

##  Project Structure

```
Monitoring System/
├── app.py                    # Main Gradio application
├── config.py                 # Hyperparameters and thresholds
├── weights/
│   └── best_model.pth        # PIAU-Net model weights
├── pipeline/
│   ├── inference.py          # Inference engine (single + sliding window)
│   ├── mask_processing.py    # Adaptive thresholding + morphology
│   ├── visualization.py      # Instance masks, heatmaps, zone grid
│   ├── density.py            # Zonal density computation
│   ├── gradcam.py            # Grad-CAM implementation
│   ├── explain.py            # Image XAI
│   ├── temporal_gradcam.py   # Video temporal CAM
│   └── integration.py        # End-to-end pipeline
├── PIAUNet/
│   ├── model/                # PIAU-Net architecture
│   ├── dataset/              # AquaOV255 dataset loader
│   └── main.py               # Training entry point
├── logs/                     # Daily application logs
├── zonal_logs/               # Per-session zone density CSV logs
└── results/                  # Output files (processed videos)
```

---

## Known Limitations

| Limitation | Description |
|---|---|
| Model resolution | PIAU-Net operates at 256×256; very small fish (<5px at model scale) may not be detected even with sliding window |
| Semantic segmentation | The model performs semantic (not instance) segmentation; instance separation is done via connected components — touching fish may be merged |
| Camouflaged fish | Fish with colours/textures similar to background coral may not be detected at the model's confidence margin |
| CPU inference | Sliding window runs 4× inference passes; CPU-only mode is significantly slower |

---

##  License

This project is intended for academic and research use in marine biology and fish population monitoring.

---

## Acknowledgements

- **PIAU-Net** — Physics-Informed Attention U-Net for underwater semantic segmentation
- **AquaOV255** — Underwater fish segmentation dataset
- **Gradio** — Web interface framework
- **OpenCV** — Image processing and morphological operations
