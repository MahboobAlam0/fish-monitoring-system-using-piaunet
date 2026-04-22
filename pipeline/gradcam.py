"""
Seg-Grad-CAM (Segmentation Gradient-weighted Class Activation Mapping)
for Physics-Informed Attention U-Net model.

Generates visual explanations by highlighting regions most important
for the model's segmentation decision.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class SegGradCAM:
    """
    Seg-Grad-CAM for segmentation models.
    
    Hooks into the last decoder layer and computes class-specific activation maps
    based on gradients of the target class with respect to feature maps.
    
    Args:
        model: PyTorch model (should be in eval mode)
        target_layer_name: Name of the layer to hook into (e.g., 'dec1')
        device: torch device
    """
    
    def __init__(self, model, target_layer_name: str = "dec1", device: str = "cpu"):
        self.model = model
        # Ensure device is a torch.device object
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.target_layer_name = target_layer_name
        
        # Get the target layer
        self.target_layer = self._get_layer_by_name(model, target_layer_name)
        if self.target_layer is None:
            raise ValueError(f"Layer {target_layer_name} not found in model")
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _get_layer_by_name(self, model, layer_name: str):
        """Retrieve a layer by its attribute name"""
        for name, module in model.named_modules():
            if name == layer_name or name.endswith('.' + layer_name):
                return module
        return None
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            # Capture activations from target layer
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            # Capture gradients flowing back through target layer
            self.gradients = grad_output[0].detach()
        
        if self.target_layer is not None:
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,  # Fish class (foreground)
        return_heatmap: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Generate Grad-CAM heatmap for a given input.
        
        Args:
            input_tensor: Input image tensor of shape (B, C, H, W)
            target_class: Target class index (1 for fish)
            return_heatmap: If True, return normalized heatmap; if False, raw weights
        
        Returns:
            heatmap: CAM heatmap (2D array, normalized to [0, 1])
            weights: Channel-wise importance weights
            input_size: Original input spatial dimensions (H, W)
        """
        B, C, H, W = input_tensor.shape
        input_size = (H, W)
        
        # Ensure input is on the model's device
        input_tensor = input_tensor.to(self.device)
        
        # CRITICAL: Use no_grad() false context + explicit enable_grad for Grad-CAM
        # This is required because model may have been set to eval mode with no_grad
        torch.set_grad_enabled(True)
        
        with torch.enable_grad():
            # Set model to eval mode (no dropout/batchnorm stochasticity)
            self.model.eval()
            
            # Clear any existing gradients
            for param in self.model.parameters():
                param.grad = None
            
            # Create input that requires gradients
            input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
            
            # Forward pass - model returns (seg_main, seg_aux, j, t, b)
            outputs = self.model(input_tensor_grad)
            seg_main = outputs[0]  # Segmentation logits: (B, 2, H, W)
            
            # Get target class logits
            target_output = seg_main[:, target_class, :, :]  # (B, H, W)
            
            # Compute loss as sum over spatial dimensions
            loss = target_output.sum()
            
            # Backward pass to compute gradients
            loss.backward(retain_graph=True)
        
        # Compute weights: average gradients over spatial dimensions
        # gradients shape: (B, C_dec1, H_feat, W_feat)
        # weights shape: (C_dec1,)
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured - hooks may not have been triggered. "
                             "Check target_layer_name and model architecture.")
        
        # Move gradients and activations to CPU for computation
        gradients = self.gradients.cpu().detach()
        activations = self.activations.cpu().detach()
        
        # Debug: Check if gradients have meaningful values
        grad_norm = gradients.abs().mean().item()
        
        if grad_norm < 1e-6:
            raise RuntimeError(f"Gradients are near zero (norm={grad_norm:.2e}). "
                             "Model may not be properly connected for gradient flow. "
                             "Verify model forward pass returns differentiable outputs.")
        
        # Compute channel importance weights (use absolute value to capture magnitude)
        weights = gradients.mean(dim=(0, 2, 3))  # Average over batch and spatial dims
        weights = torch.abs(weights)  # Use absolute value to capture gradient magnitude
        
        # Compute CAM by weighted sum of activations
        # activations shape: (B, C_dec1, H_feat, W_feat)
        cam = torch.zeros(B, activations.shape[2], activations.shape[3])
        
        for i in range(len(weights)):
            cam += weights[i] * activations[:, i, :, :]
        
        cam = F.relu(cam)  # Apply ReLU to keep only positive activations
        cam = cam[0].numpy()  # Take first sample, convert to numpy
        
        # Normalize CAM to [0, 1]
        if cam.max() > 0:
            cam_normalized = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam_normalized = cam
        
        # Resize CAM to input size using bilinear interpolation
        cam_tensor = torch.from_numpy(cam_normalized).unsqueeze(0).unsqueeze(0).float()
        cam_resized = F.interpolate(
            cam_tensor,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )
        heatmap = cam_resized[0, 0].cpu().numpy()
        
        # Final normalization
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap, weights.cpu().numpy(), input_size
    
    def __call__(self, input_tensor: torch.Tensor, target_class: int = 1):
        """Make the instance callable"""
        return self.generate_cam(input_tensor, target_class)
