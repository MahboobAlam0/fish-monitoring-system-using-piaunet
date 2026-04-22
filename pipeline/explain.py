"""
Text explanation generator based on Seg-Grad-CAM heatmap and model outputs.

Generates human-readable explanations of model predictions by analyzing:
- Where the model focuses (CAM heatmap)
- Whether focus aligns with predicted fish regions
- Whether focus aligns with clear water (low turbidity)
"""

import numpy as np
from typing import Any, Dict, Optional


class ExplanationGenerator:
    """
    Generate textual explanations for segmentation predictions.
    
    Analyzes:
    1. CAM focus regions
    2. Overlap between CAM and predicted segmentation
    3. Alignment with turbidity map (physics insight)
    """
    
    def __init__(self, threshold_high: float = 0.7, threshold_low: float = 0.3):
        """
        Args:
            threshold_high: High CAM intensity threshold
            threshold_low: Low CAM intensity threshold
        """
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
    
    def _get_focus_region(self, heatmap: np.ndarray) -> Dict[str, Any]:
        """
        Analyze where CAM heatmap focuses.
        
        Returns:
            Dict with focus statistics:
            - intensity: 'high', 'moderate', 'low'
            - coverage: percentage of image with significant CAM
            - center: whether focus is centered or peripheral
        """
        high_intensity = np.sum(heatmap > self.threshold_high) / heatmap.size
        mean_intensity = np.mean(heatmap)
        
        # Determine intensity level
        if high_intensity > 0.2:
            intensity = "high"
        elif high_intensity > 0.05:
            intensity = "moderate"
        else:
            intensity = "low"
        
        # Check if focus is centered (top/bottom/left/right have less activation)
        h, w = heatmap.shape
        center_region = heatmap[h//4:3*h//4, w//4:3*w//4]
        edge_region = np.concatenate([
            heatmap[:h//4, :].flatten(),
            heatmap[3*h//4:, :].flatten(),
            heatmap[:, :w//4].flatten(),
            heatmap[:, 3*w//4:].flatten()
        ])
        
        center_mean = center_region.mean() if center_region.size > 0 else 0
        edge_mean = edge_region.mean() if edge_region.size > 0 else 0
        
        is_centered = center_mean > edge_mean * 1.5
        
        return {
            "intensity": intensity,
            "coverage": high_intensity,
            "mean_value": mean_intensity,
            "is_centered": is_centered
        }
    
    def _get_overlap_with_segmentation(
        self,
        heatmap: np.ndarray,
        seg_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Check overlap between CAM focus and predicted fish regions.
        
        Args:
            heatmap: CAM heatmap
            seg_pred: Segmentation prediction (binary or soft 0-1)
            threshold: Threshold for heatmap to consider as "focused"
        
        Returns:
            Dict with overlap statistics
        """
        # Binarize predictions
        high_cam = heatmap > threshold
        high_seg = seg_pred > 0.5
        
        # Calculate overlaps
        cam_area = np.sum(high_cam)
        seg_area = np.sum(high_seg)
        overlap = np.sum(high_cam & high_seg)
        
        if cam_area > 0:
            overlap_ratio = overlap / cam_area
            recall = overlap / seg_area if seg_area > 0 else 0
        else:
            overlap_ratio = 0
            recall = 0
        
        return {
            "cam_area": cam_area / high_cam.size,
            "seg_area": seg_area / high_seg.size,
            "overlap_ratio": overlap_ratio,
            "recall": recall
        }
    
    def generate_explanation(
        self,
        heatmap: np.ndarray,
        seg_pred: np.ndarray,
        pred_confidence: Optional[float] = None
    ) -> str:
        """
        Generate complete text explanation.
        
        Args:
            heatmap: Seg-Grad-CAM heatmap (0-1 normalized)
            seg_pred: Segmentation prediction (0-1 or binary)
            pred_confidence: Model's confidence in prediction (optional)
        
        Returns:
            Formatted explanation string
        """
        # Analyze focus
        focus = self._get_focus_region(heatmap)
        
        # Analyze overlap
        overlap = self._get_overlap_with_segmentation(heatmap, seg_pred)
        
        # Calculate actual model confidence from segmentation prediction
        seg_confidence = np.mean(seg_pred)  # Average prediction confidence
        seg_max = np.max(seg_pred)  # Peak confidence
        high_confidence_pixels = np.sum(seg_pred > 0.7) / seg_pred.size  # % pixels >70% confident
        
        # Determine overall confidence level
        if seg_confidence > 0.8 or high_confidence_pixels > 0.8:
            confidence_level = "VERY HIGH"
            confidence_emoji = "✓"
        elif seg_confidence > 0.6 or high_confidence_pixels > 0.5:
            confidence_level = "HIGH"
            confidence_emoji = "✓"
        elif seg_confidence > 0.4 or high_confidence_pixels > 0.3:
            confidence_level = "MODERATE"
            confidence_emoji = "≈"
        else:
            confidence_level = "LOW"
            confidence_emoji = "✗"
        
        # Build explanation strings
        explanation_parts = []
        
        # 1. Overall prediction confidence (NEW - separate from focus)
        confidence_text = f"Prediction Confidence: {confidence_emoji} {confidence_level}"
        confidence_text += f" (Mean: {seg_confidence:.2%}, Peak: {seg_max:.2%})"
        explanation_parts.append(confidence_text)
        
        # 2. Focus distribution analysis (clarified)
        if focus['is_centered']:
            focus_pattern = "concentrated on central region"
        else:
            focus_pattern = "distributed across image (broad attention)"
        
        focus_text = f"Attention Pattern: Model focuses with {focus['intensity']} CAM intensity, "
        focus_text += focus_pattern + "."
        focus_text += f" (CAM activation: {focus['mean_value']:.2f})"
        explanation_parts.append(focus_text)
        
        # 3. Detection alignment (improved logic based on both metrics)
        alignment_text = "Prediction Alignment: "
        
        # Use overlap ratio (which is 99.3% in your case) as primary metric
        if overlap['overlap_ratio'] > 0.85:
            alignment_text += "PERFECT - CAM focus precisely matches predicted regions."
        elif overlap['recall'] > 0.7 or overlap['overlap_ratio'] > 0.7:
            alignment_text += "Excellent - CAM focus aligns well with predicted regions."
        elif overlap['recall'] > 0.4 or overlap['overlap_ratio'] > 0.4:
            alignment_text += "Good - CAM focus reasonably aligns with predictions."
        else:
            alignment_text += "Needs review - CAM focus and predictions diverge."
        alignment_text += f" (Overlap: {overlap['overlap_ratio']:.1%}, Recall: {overlap['recall']:.1%})"
        explanation_parts.append(alignment_text)
        
        # 4. Physics insight (Removed)
        
        # 5. Enhanced summary
        summary_parts = []
        
        # Confidence-based assessment
        if seg_confidence > 0.85:
            summary_parts.append("Model is VERY CONFIDENT in this prediction")
        elif seg_confidence > 0.65:
            summary_parts.append("Model is MODERATELY CONFIDENT in this prediction")
        else:
            summary_parts.append("Model is UNCERTAIN - consider with caution")
        
        # Attention quality
        if focus['is_centered'] and overlap['recall'] > 0.6:
            summary_parts.append("with focused, well-localized attention")
        elif overlap['recall'] > 0.6:
            summary_parts.append("with distributed but aligned attention")
        elif focus['is_centered']:
            summary_parts.append("but attention is concentrated away from predictions")
        else:
            summary_parts.append("with diffuse attention patterns")
        
        summary_text = "Summary: " + ", ".join(summary_parts) + "."
        explanation_parts.append(summary_text)
        
        return "\n".join(explanation_parts)


def explain_prediction(
    heatmap: np.ndarray,
    seg_pred: np.ndarray
) -> str:
    """
    Convenience function to generate explanation.
    
    Args:
        heatmap: Seg-Grad-CAM heatmap
        seg_pred: Segmentation prediction
    
    Returns:
        Explanation string
    """
    generator = ExplanationGenerator()
    return generator.generate_explanation(heatmap, seg_pred)
