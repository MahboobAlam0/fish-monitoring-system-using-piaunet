"""
Visualization utilities for XAI explanations.

Creates a comprehensive figure showing:
- Input image
- Segmentation prediction
- Grad-CAM heatmap
- Text explanation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range"""
    if img.max() <= 1.0:
        return img
    return img / 255.0


def create_xai_figure(
    input_image: np.ndarray,
    seg_pred: np.ndarray,
    cam_heatmap: np.ndarray,
    explanation_text: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    colormap: str = 'jet'
) -> Figure:
    """
    Create a comprehensive visualization figure for XAI explanation.
    
    Args:
        input_image: Input image (H, W, 3) in [0, 1] or [0, 255]
        seg_pred: Segmentation prediction, shape (H, W) or (H, W, 2)
        cam_heatmap: Grad-CAM heatmap, shape (H, W) in [0, 1]
        explanation_text: Text explanation to display (optional)
        figsize: Figure size (width, height)
        colormap: Colormap for heatmaps
    
    Returns:
        matplotlib Figure object
    """
    # Normalize inputs
    input_image = normalize_image(input_image)
    
    # Extract foreground class from segmentation if 2-channel
    if seg_pred.ndim == 3 and seg_pred.shape[2] == 2:
        seg_pred = seg_pred[:, :, 1]  # Fish class
    
    # Create figure with grid
    fig = plt.figure(figsize=figsize)
    
    # Adjust layout for text
    n_rows = 3 if explanation_text else 2
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Input image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(input_image)
    ax1.set_title("Input Image", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Segmentation prediction
    ax2 = fig.add_subplot(gs[0, 1])
    seg_colored = ax2.imshow(seg_pred, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax2.set_title("Segmentation Prediction (Fish Probability)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(seg_colored, ax=ax2, label='Probability')
    
    # 3. Grad-CAM heatmap overlay on input
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(input_image)
    # Overlay CAM with transparency
    im_cam = ax3.imshow(cam_heatmap, cmap=colormap, alpha=0.6, vmin=0, vmax=1)
    ax3.set_title("Grad-CAM Heatmap (where model focuses)", fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im_cam, ax=ax3, label='Activation')
    
    # 4. Additional visualization
    ax4 = fig.add_subplot(gs[1, 1])
    # Show CAM as standalone
    im_cam_standalone = ax4.imshow(cam_heatmap, cmap=colormap, vmin=0, vmax=1)
    ax4.set_title("Grad-CAM Heatmap (standalone)", fontsize=12, fontweight='bold')
    plt.colorbar(im_cam_standalone, ax=ax4, label='Activation')
    ax4.axis('off')
    
    # 5. Text explanation
    if explanation_text:
        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis('off')
        
        # Format text with better spacing
        formatted_text = "\n".join(explanation_text.split("\n"))
        
        ax_text.text(
            0.05, 0.95,
            formatted_text,
            transform=ax_text.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1)
        )
    
    return fig


def save_xai_figure(
    fig: Figure,
    save_path: str,
    dpi: int = 150
) -> None:
    """Save figure to disk"""
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {save_path}")


def create_comparison_figure(
    input_image: np.ndarray,
    seg_pred: np.ndarray,
    cam_heatmap: np.ndarray,
    save_path: Optional[str] = None
) -> Figure:
    """
    Create simpler 2x2 comparison figure (for quick checks).
    
    Args:
        input_image: Input image
        seg_pred: Segmentation prediction
        cam_heatmap: Grad-CAM heatmap
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    input_image = normalize_image(input_image)
    
    # Extract foreground class if needed
    if seg_pred.ndim == 3 and seg_pred.shape[2] == 2:
        seg_pred = seg_pred[:, :, 1]
    
    # Row 1
    axes[0, 0].imshow(input_image)
    axes[0, 0].set_title("Input")
    axes[0, 0].axis('off')
    
    seg_im = axes[0, 1].imshow(seg_pred, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[0, 1].set_title("Segmentation")
    axes[0, 1].axis('off')
    plt.colorbar(seg_im, ax=axes[0, 1])
    
    # Row 2
    axes[1, 0].imshow(input_image)
    cam_im = axes[1, 0].imshow(cam_heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[1, 0].set_title("Grad-CAM Overlay")
    axes[1, 0].axis('off')
    plt.colorbar(cam_im, ax=axes[1, 0])
    
    cam_standalone = axes[1, 1].imshow(cam_heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title("Grad-CAM (standalone)")
    axes[1, 1].axis('off')
    plt.colorbar(cam_standalone, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig
