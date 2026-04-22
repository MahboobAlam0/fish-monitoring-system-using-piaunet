"""
Temporal Grad-CAM for Video Analysis
Generates CAM heatmaps for each frame, showing how model attention changes over time.
Outputs animated visualization of attention patterns.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TemporalGradCAM:
    """
    Generate temporal Grad-CAM visualizations for video.
    Shows how model attention changes across frames.
    """
    
    def __init__(self, gradcam_engine, model, device: str = "cuda"):
        """
        Args:
            gradcam_engine: SegGradCAM instance for frame analysis
            model: The segmentation model
            device: torch device
        """
        self.gradcam = gradcam_engine
        self.model = model
        self.device = device
    
    def extract_frames(
        self,
        video_path: str,
        sample_rate: int = 1,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame (1=all, 2=every 2nd, etc.)
            max_frames: Maximum frames to extract (None=all)
        
        Returns:
            List of frame arrays (RGB, 0-255)
        """
        logger.info(f"[TEMPORAL CAM] Extracting frames from: {video_path}")
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[TEMPORAL CAM] Video: {total_frames} frames @ {fps} FPS")
        
        frame_idx = 0
        while len(frames) < (max_frames or total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                logger.info(f"[TEMPORAL CAM] Extracted frame {frame_idx}: shape {frame_rgb.shape}")
            
            frame_idx += 1
            
            if max_frames and len(frames) >= max_frames:
                break
        
        cap.release()
        logger.info(f"[TEMPORAL CAM] Extracted {len(frames)} frames")
        return frames
    
    def generate_temporal_cam(
        self,
        frames: List[np.ndarray],
        target_class: int = 1
    ) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Generate Grad-CAM for each frame.
        
        Args:
            frames: List of frame arrays
            target_class: Target class for CAM (1 for fish)
        
        Returns:
            Tuple of (heatmaps, confidences, activations)
            - heatmaps: List of CAM heatmaps (0-1 normalized)
            - confidences: Model confidence per frame
            - activations: Mean CAM activation per frame
        """
        logger.info(f"[TEMPORAL CAM] Generating CAM for {len(frames)} frames...")
        
        heatmaps = []
        confidences = []
        activations = []
        
        self.model.eval()
        
        for idx, frame in enumerate(frames):
            # Initialize dimensions before try block
            h, w = frame.shape[:2]
            
            try:
                # Prepare frame
                frame_resized = cv2.resize(frame, (512, 512))
                frame_norm = frame_resized.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1).unsqueeze(0)
                frame_tensor = frame_tensor.to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(frame_tensor)
                    seg_main = outputs[0]
                
                # Calculate confidence
                seg_prob = torch.softmax(seg_main, dim=1)[0, 1]
                confidence = seg_prob.mean().item()
                confidences.append(confidence)
                
                # Generate CAM
                heatmap, _, _ = self.gradcam(frame_tensor, target_class=target_class)
                heatmap_resized = cv2.resize(heatmap, (w, h))
                heatmaps.append(heatmap_resized)
                activations.append(heatmap_resized.mean())
                
                logger.info(f"[TEMPORAL CAM] Frame {idx+1}/{len(frames)}: confidence={confidence:.2%}, CAM mean={heatmap_resized.mean():.4f}")
                
            except Exception as e:
                logger.error(f"[TEMPORAL CAM] Error processing frame {idx}: {e}")
                # Use zero heatmap for failed frame
                heatmaps.append(np.zeros((h, w)))
                confidences.append(0.0)
                activations.append(0.0)
        
        return heatmaps, confidences, activations
    
    def create_temporal_visualization(
        self,
        frames: List[np.ndarray],
        heatmaps: List[np.ndarray],
        confidences: List[float],
        output_path: str,
        fps: float = 10.0,
        cmap: str = 'jet'
    ) -> str:
        """
        Create animated temporal CAM visualization (MP4 video).
        Shows: Original Frame | CAM Heatmap | CAM Overlay
        
        Args:
            frames: List of original frames
            heatmaps: List of CAM heatmaps
            confidences: List of confidence scores
            output_path: Path to save output video
            fps: Frames per second for output video
            cmap: Colormap for heatmap visualization
        
        Returns:
            Path to saved video file
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        logger.info("[TEMPORAL CAM] Creating animated visualization...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare colormap
        cmap_func = cm.get_cmap(cmap)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
        video_writer = None
        
        for idx, (frame, heatmap, confidence) in enumerate(zip(frames, heatmaps, confidences)):
            h, w = frame.shape[:2]
            
            # Create 3-panel figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Temporal Grad-CAM - Frame {idx+1}/{len(frames)} (Confidence: {confidence:.2%})', 
                        fontsize=14, fontweight='bold')
            
            # Panel 1: Original frame
            axes[0].imshow(frame)
            axes[0].set_title('Original Frame')
            axes[0].axis('off')
            
            # Panel 2: CAM heatmap
            heatmap_colored = cmap_func(heatmap)
            axes[1].imshow(heatmap_colored)
            axes[1].set_title(f'Grad-CAM (Mean: {heatmap.mean():.3f})')
            axes[1].axis('off')
            
            # Panel 3: CAM overlay on frame
            heatmap_3channel = cmap_func(heatmap)[:, :, :3]  # RGB only (0-1)
            # Blend: 70% original frame + 30% heatmap (keep values in 0-255 range)
            overlay = (0.7 * frame + 0.3 * (heatmap_3channel * 255)).astype(np.uint8)
            axes[2].imshow(overlay)
            axes[2].set_title('CAM Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Convert matplotlib figure to numpy array (modern matplotlib compatible)
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            # Use buffer_rgba() for modern matplotlib compatibility
            rgba_buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
            img = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(height, width, 4)
            # Convert RGBA to RGB then to BGR for OpenCV
            img_rgb = img[:, :, :3]  # Drop alpha channel
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Initialize video writer on first frame
            if video_writer is None:
                frame_h, frame_w = img_bgr.shape[:2]
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
            
            video_writer.write(img_bgr)
            plt.close(fig)
            
            logger.info(f"[TEMPORAL CAM] Processed frame {idx+1}/{len(frames)}")
        
        if video_writer:
            video_writer.release()
        
        logger.info(f"[TEMPORAL CAM] Temporal CAM video saved: {output_path}")
        return output_path
    
    def analyze_video_stats_only(
        self,
        video_path: str,
        sample_rate: int = 2,
        max_frames: int = 30,
    ) -> dict:
        """
        Run temporal CAM analysis and return statistics only.
        No video file is created or saved anywhere on disk.

        Returns:
            statistics_dict
        """
        logger.info(f"[TEMPORAL CAM] Stats-only analysis: {video_path}")

        frames = self.extract_frames(video_path, sample_rate=sample_rate, max_frames=max_frames)
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")

        heatmaps, confidences, activations = self.generate_temporal_cam(frames)

        stats = {
            "total_frames": len(frames),
            "mean_confidence": float(np.mean(confidences)),
            "max_confidence":  float(np.max(confidences)),
            "min_confidence":  float(np.min(confidences)),
            "mean_activation": float(np.mean(activations)),
            "max_activation":  float(np.max(activations)),
            "min_activation":  float(np.min(activations)),
            "confidences":  confidences,
            "activations":  activations,
        }

        logger.info("[TEMPORAL CAM] Stats-only analysis complete.")
        logger.info(f"  Frames: {stats['total_frames']}")
        logger.info(f"  Avg Confidence: {stats['mean_confidence']:.2%}")
        logger.info(f"  Avg CAM Activation: {stats['mean_activation']:.4f}")
        return stats
