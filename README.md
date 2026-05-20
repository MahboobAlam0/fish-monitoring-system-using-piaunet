---
title: Fish Density Monitoring System
emoji: 🐟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.7.1"
app_file: app.py
pinned: false
license: apache-2.0
python_version: "3.11"
---

# Fish Density Monitoring System

<p align="center">
  <img src="./PIAUNet/piaunet_overall.png" alt="System Overview" width="80%"/>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/mahboobalam0/fish-density-monitoring-system">
    <img src="https://img.shields.io/badge/Live_Demo-HuggingFace_Spaces-yellow?logo=huggingface" alt="Live Demo"/>
  </a>
  <a href="https://huggingface.co/mahboobalam0/piaunet">
    <img src="https://img.shields.io/badge/Model-HF_Hub-orange?logo=huggingface" alt="HF Model"/>
  </a>
  <a href="./PIAUNet">
    <img src="https://img.shields.io/badge/PIAU--Net-Published_Paper-blueviolet" alt="Paper"/>
  </a>
  <img src="https://img.shields.io/badge/mIoU-97.38%25-brightgreen" alt="mIoU"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Gradio-5.7-blue?logo=gradio" alt="Gradio"/>
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python"/>
</p>

---

## What This System Does

Underwater fish populations are critical to monitor for aquaculture management, marine ecology, and conservation — but manual counting is slow, subjective, and impossible at scale.

This system solves it end-to-end:

1. Upload an image or video of an underwater scene
2. **PIAU-Net** (a peer-reviewed, physics-informed segmentation model) detects and masks every fish
3. The frame is divided into a configurable **zonal grid** — each zone gets a live fish count and density level (LOW / MEDIUM-HIGH / HIGH)
4. **Density alerts** fire on zones that breach a configurable threshold
5. **Seg-Grad-CAM** explains which pixels drove the prediction
6. For videos, a **Temporal CAM** video shows model attention evolving frame-by-frame

Everything runs in a dark-themed Gradio dashboard deployable locally or on Hugging Face Spaces.

---

## Live Demo

**[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/mahboobalam0/fish-density-monitoring-system)**

---

## Key Features

| Feature | Description |
|---|---|
| **Physics-Informed Segmentation** | PIAU-Net embeds turbidity and backscatter physics into the attention mechanism for underwater-robust detection |
| **Zonal Density Grid** | Configurable N×M grid with per-zone fish count, density score, and color-coded alert level |
| **Adaptive Confidence Thresholding** | `fish_prob − bg_prob > margin` instead of a fixed threshold — eliminates open-water false positives |
| **Sliding Window Inference** | 4 overlapping 60%-patches with MAX aggregation for small or distant fish at full resolution |
| **Temporal Voting** | 3-frame majority vote filter removes flickering detections in video |
| **Morphological Cleanup** | MORPH_OPEN (noise removal) + MORPH_CLOSE (hole fill) for clean, solid fish masks |
| **Seg-Grad-CAM XAI** | Gradient-based attention heatmap on `dec1` showing which regions drove the prediction |
| **Temporal Video CAM** | Frame-by-frame model attention video (3-panel: original / heatmap / overlay) |
| **Session Audit Trail** | Per-session JSON metadata + CSV zonal logs with timestamps |
| **HF Spaces Deployment** | Weights auto-downloaded from HF Hub at startup — zero manual setup |

---

## Model: PIAU-Net

The segmentation backbone is **PIAU-Net** — a published, peer-reviewed architecture designed specifically for underwater image segmentation.

> **Alam, M., Dhavale, S. V., and Srikanth, D.**
> *Physics-Informed Attention U-Net (PIAUNet): An Enhanced U-Net Framework for Underwater Segmentation in Aquaculture.*
> Indian Journal of Technical Education, Vol. 48, No. 2, December 2025.

### Why It Beats Standard U-Nets Underwater

Standard segmentation models rely purely on appearance. Underwater, light scattering, turbidity, and backscatter shift pixel statistics in ways that confuse appearance-based features. PIAU-Net addresses this directly:

- A **Physics Branch** at the bottleneck learns turbidity (`t`) and backscatter (`b`) maps from scene features alone
- **Physics-Informed Attention Gates (PAGs)** use `t` to gate every skip connection — suppressing features from optically unreliable (murky) regions before decoder fusion
- **Deep Supervision** via two auxiliary heads stabilizes convergence

```
alpha = Sigmoid( ReLU( W_g(decoder) + W_x(skip) + W_phys(turbidity) ) )
output = skip * alpha
```

### Benchmark Results

**Large-Scale Fish Dataset**

| Model | mIoU (%) | Dice (%) | Precision (%) | Recall (%) | Pixel Acc. (%) |
|---|---|---|---|---|---|
| U-Net | 93.48 | 94.66 | 96.50 | 96.83 | 95.70 |
| Attention U-Net | 95.23 | 96.53 | 97.60 | 97.46 | 98.06 |
| DeepLab V3+ | 95.01 | 96.04 | 96.42 | 97.67 | 96.85 |
| **PIAU-Net (Ours)** | **97.38** | **98.18** | **98.83** | **98.53** | **99.54** |

**AquaOV255 — After Stage 2 Fine-Tuning**

| Model | mIoU (%) | Dice (%) | Precision (%) | Recall (%) | Pixel Acc. (%) |
|---|---|---|---|---|---|
| U-Net (fine-tuned) | 87.79 | 90.67 | 88.96 | 92.65 | 95.10 |
| Attention U-Net (fine-tuned) | 88.05 | 90.92 | 90.29 | 91.59 | 95.38 |
| DeepLab V3+ (fine-tuned) | 90.54 | 94.91 | 95.83 | 94.04 | 97.50 |
| **PIAU-Net (Ours)** | **93.98** | **96.85** | **96.56** | **97.13** | **98.41** |

### Visual Results

**Large-Scale Fish Dataset**

<p align="center">
  <img src="./PIAUNet/TestResults/LargeFishDataset/image1.png" width="32%"/>
  <img src="./PIAUNet/TestResults/LargeFishDataset/image2.png" width="32%"/>
  <img src="./PIAUNet/TestResults/LargeFishDataset/image3.png" width="32%"/>
</p>

**AquaOV255 Dataset**

<p align="center">
  <img src="./PIAUNet/TestResults/AquaOV255Dataset/img_5.png" width="32%"/>
  <img src="./PIAUNet/TestResults/AquaOV255Dataset/img_40.png" width="32%"/>
  <img src="./PIAUNet/TestResults/AquaOV255Dataset/img_63.png" width="32%"/>
</p>
<p align="center">
  <img src="./PIAUNet/TestResults/AquaOV255Dataset/img_70.png" width="32%"/>
  <img src="./PIAUNet/TestResults/AquaOV255Dataset/img_8.png" width="32%"/>
  <img src="./PIAUNet/TestResults/AquaOV255Dataset/img_135.png" width="32%"/>
</p>

*Each panel: Input | Ground Truth | PIAU-Net Prediction*

---

## System Architecture

```
Fish Density Monitoring System
│
├── app.py                          # Gradio dashboard & session orchestration
├── config.py                       # Thresholds, grid config, device
│
├── pipeline/
│   ├── inference.py                # Single-frame + 4-patch sliding window (MAX aggregation)
│   ├── mask_processing.py          # Adaptive confidence-margin thresholding + morphology
│   ├── density.py                  # Per-zone fish density scoring & alert logic
│   ├── visualization.py            # Instance masks, heatmap, zone grid overlay
│   ├── post_processing.py          # NMS, temporal voting (3-frame), frame recovery
│   ├── gradcam.py                  # Seg-Grad-CAM on dec1 layer
│   ├── explain.py                  # Image-level XAI text + figure generation
│   ├── temporal_gradcam.py         # Frame-by-frame temporal CAM for video
│   ├── xai_visualization.py        # 4-panel CAM overlay figure renderer
│   └── integration.py              # Session context manager + health monitoring
│
└── PIAUNet/
    ├── model/model.py              # PhysicsInformedAttentionUNet + PAG
    ├── physics/physicsComponents.py # PhysicsGuidedModule (turbidity + backscatter)
    ├── lossfunction/               # Physics-guided loss with smoothness regularizer
    ├── dataset/                    # AquaOV255 loader, CLAHE, mask validation
    └── main.py                     # Training / testing / tuning CLI
```

### Detection Pipeline

```
Raw logits [2 × H × W]
    │ softmax
    ▼
fish_prob = probs[0],  bg_prob = probs[1]
    │ confidence margin
    ▼
confidence = fish_prob − bg_prob  →  mask = (confidence > 0.02)
    │ resize to original resolution
    ▼
temporal voting  (3-frame window, majority > 0.5)
    │
    ▼  MORPH_OPEN  3×3   → remove noise
       MORPH_CLOSE 5×5   → fill holes within fish blobs
    │
    ▼  connected-component filter
       keep: 150px ≤ area ≤ 25% of frame   (not noise, not flood)
       keep: aspect ratio ≤ 5:1             (not sky/water band)
    │
    ▼
clean instance mask  →  zonal density analysis  →  alerts + visualization
```

---

## Quickstart

### Local Setup

```bash
# 1. Clone
git clone https://github.com/MahboobAlam0/fish-monitoring-system-using-piaunet.git
cd fish-monitoring-system-using-piaunet

# 2. Create environment
conda create -n fishmon python=3.11
conda activate fishmon

# 3. Install PyTorch (CUDA 11.8)
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Place weights  (auto-downloaded from HF Hub if not found)
mkdir weights
# copy best_model.pth → weights/best_model.pth

# 6. Launch
python app.py
```

Open **http://localhost:7860** in your browser.

> **No weights?** The app auto-downloads `best_model.pth` from [huggingface.co/mahboobalam0/piaunet](https://huggingface.co/mahboobalam0/piaunet) at startup if the `weights/` folder is empty.

### Usage

1. Upload an image (`.jpg`, `.png`) or video (`.mp4`, `.avi`, `.mov`)
2. Set zone grid rows/columns and alert threshold
3. Toggle **High-Res Sliding Window** for better small-fish detection
4. Toggle **Show Density Heatmap** / **Show Zone Grid** as needed
5. Enable **Image Grad-CAM** or **Video Temporal CAM** for XAI
6. Click **Analyze Data**

### Outputs

| Panel | Contents |
|---|---|
| Segmentation Output | Color-coded instance masks with bounding boxes |
| Detailed Report | Fish count, zone-by-zone density, alert summary, processing time |
| Zone Summary Legend | Per-zone density level breakdown (LOW → HIGH) |
| Explainable AI (Image) | Seg-Grad-CAM heatmap + interpretation text |
| Explainable AI (Video) | 3-panel temporal CAM video + frame-by-frame stats |

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` / `cpu` | Auto-detected at startup |
| `IMG_SIZE` | `256` | Model input resolution |
| `THRESHOLD` | `0.50` | Hard minimum confidence floor |
| `MIN_AREA` | `150` | Minimum blob area (px) — filters noise |
| `IOU_THRESHOLD` | `0.6` | NMS overlap threshold |
| `ZONAL_GRID_ROWS` | `5` | Zone grid rows |
| `ZONAL_GRID_COLS` | `5` | Zone grid columns |
| `ZONAL_ALERT_THRESHOLD` | `MEDIUM-HIGH` | Density level that triggers alerts |

---

## Training PIAU-Net

Full training code, datasets, and instructions are in the [PIAUNet/](./PIAUNet) subdirectory.

```bash
# Stage 1: Train on Large-Scale Fish Dataset
python PIAUNet/main.py --mode train --dataset_root "./Fish Dataset" --epochs 30

# Stage 2: Fine-tune on AquaOV255
python PIAUNet/main.py --mode train --dataset_root ./AquaOV255 \
    --checkpoint checkpoints/best.pth --epochs 30

# Test
python PIAUNet/main.py --mode test --checkpoint checkpoints/best.pth \
    --dataset_root ./AquaOV255
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Segmentation Model | PyTorch 2.0, custom PIAU-Net |
| Inference | CUDA / CPU, AMP, sliding window |
| Explainability | Seg-Grad-CAM (gradient-based, `dec1` layer) |
| Image Processing | OpenCV, Pillow, Albumentations |
| Web Interface | Gradio 5.7 |
| Deployment | Hugging Face Spaces (auto HF Hub weight download) |
| Logging | Python logging, per-session JSON + CSV audit trail |

---

## Limitations

| Limitation | Detail |
|---|---|
| Resolution | PIAU-Net operates at 256×256; very small fish (<5 px at model scale) may be missed even with sliding window |
| Instance separation | Touching fish may be merged — instance splitting is connected-component based, not learned |
| Camouflaged fish | Fish blending into coral or sand at the model's confidence margin may be missed |
| CPU inference | Sliding window runs 4× passes — CPU-only is significantly slower than GPU |

---

## Citation

If you use this work, please cite the paper:

```bibtex
@article{alam2025piaunet,
  title   = {Physics-Informed Attention U-Net (PIAUNet): An Enhanced U-Net Framework
             for Underwater Segmentation in Aquaculture},
  author  = {Alam, Mahboob and Dhavale, Sunita Vikram and Srikanth, D.},
  journal = {Indian Journal of Technical Education},
  volume  = {48},
  number  = {2},
  month   = {December},
  year    = {2025}
}
```

---

## Links

- Live Demo: [huggingface.co/spaces/mahboobalam0/fish-density-monitoring-system](https://huggingface.co/spaces/mahboobalam0/fish-density-monitoring-system)
- Model Weights: [huggingface.co/mahboobalam0/piaunet](https://huggingface.co/mahboobalam0/piaunet)
- Model Repo: [github.com/MahboobAlam0/piaunet](https://github.com/MahboobAlam0/piaunet)
- This Repo: [github.com/MahboobAlam0/fish-monitoring-system-using-piaunet](https://github.com/MahboobAlam0/fish-monitoring-system-using-piaunet)
