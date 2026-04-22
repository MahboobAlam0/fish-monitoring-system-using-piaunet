import cv2
import torch
import numpy as np
from torchvision.transforms import functional as TF
from config import IMG_SIZE
import torch.nn.functional as F

# Normalization must match training: GenericSegmentationDataset uses mean=0.5, std=0.5
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

def run_inference(model, frame, device, use_sliding_window=False):
    if use_sliding_window:
        return run_sliding_window_inference(model, frame, device)
        
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    img = TF.to_tensor(resized)
    img = TF.normalize(img, mean=MEAN, std=STD).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)

    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out

    return logits


def run_sliding_window_inference(model, frame, device):
    """
    Run inference using a 2×2 overlapping grid to preserve high-res details for small fish.

    Aggregation strategy: MAX instead of MEAN.
    - MEAN dilutes the signal from small fish that only appear in one patch.
    - MAX preserves the strongest detection from ANY patch that covers a pixel,
      so a small fish strongly detected in one patch is not washed out by the
      other three patches where that pixel maps to open water.
    """
    h, w = frame.shape[:2]

    patch_w = int(w * 0.6)
    patch_h = int(h * 0.6)

    step_x = w - patch_w
    step_y = h - patch_h

    positions = [
        (0,      0),
        (step_x, 0),
        (0,      step_y),
        (step_x, step_y),
    ]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialise accumulator with -inf so the first real value always wins
    accum_logits = None

    for x, y in positions:
        patch = rgb[y:y+patch_h, x:x+patch_w]
        resized_patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))

        img = TF.to_tensor(resized_patch)
        img = TF.normalize(img, mean=MEAN, std=STD).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)

        if isinstance(out, tuple):
            patch_logits = out[0]
        else:
            patch_logits = out

        # Upsample patch logits back to the patch's original pixel dimensions
        patch_logits_resized = F.interpolate(
            patch_logits, size=(patch_h, patch_w), mode='bilinear', align_corners=False
        )

        C = patch_logits_resized.shape[1]

        if accum_logits is None:
            # -inf baseline: any real logit beats this
            accum_logits = torch.full(
                (1, C, h, w), float('-inf'), device=device, dtype=torch.float32
            )

        # Embed this patch into a full-frame tensor (rest stays -inf)
        full_frame = torch.full(
            (1, C, h, w), float('-inf'), device=device, dtype=torch.float32
        )
        full_frame[:, :, y:y+patch_h, x:x+patch_w] = patch_logits_resized

        # Keep the MAX logit seen so far at each pixel
        accum_logits = torch.maximum(accum_logits, full_frame)

    # Any pixel that was never covered (shouldn't happen with 4×60% patches) → 0
    accum_logits = torch.nan_to_num(accum_logits, neginf=0.0)

    return accum_logits