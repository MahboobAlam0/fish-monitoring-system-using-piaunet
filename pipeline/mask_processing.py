import cv2
import torch
import numpy as np
from collections import deque


class MaskProcessor:
    def __init__(self, window_size=3):
        self.masks = deque(maxlen=window_size)

    def process(self, logits, h, w, threshold, min_area, frame=None):

        # Handle tuple output safely
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        # Softmax over class dimension → [C, H, W]
        probs = torch.softmax(logits, dim=1)[0]

        # ── Class assignment ──────────────────────────────────────────────────
        # The saved weights (best_model.pth) treat channel 0 = fish, channel 1 = water/bg.
        # This is confirmed by observation: class 1 consistently detected open water & sky,
        # while class 0 (original code) detected fish locations.
        fish_class = 0
        bg_class   = 1

        fish_prob = probs[fish_class].cpu().numpy()   # float32, model-space (256×256)
        bg_prob   = probs[bg_class].cpu().numpy()

        # ── Confidence-margin thresholding ─────────────────────────────────────
        # confidence = fish_prob - bg_prob  → range [-1, +1]
        #   > 0  means fish is more likely than background at that pixel
        #   < 0  means background is more likely
        #
        # The 90th-percentile approach was too strict (kept only top ~2.5% of all
        # pixels → very few boxes). Instead, we use a simple positive-margin gate:
        # any pixel where the model thinks "fish" beats "background" qualifies.
        # Area/aspect filters below then remove noise and false large regions.
        confidence = fish_prob - bg_prob   # range [-1, +1]

        # Require a tiny positive margin to avoid 50/50 uncertainty pixels.
        # 0.02 is intentionally low to catch small/distant fish; the area and
        # aspect ratio filters below are the real gatekeepers against false positives.
        mask = (confidence > 0.02).astype("uint8")

        # ── Resize to original frame dimensions ───────────────────────────────
        mask        = cv2.resize(mask,      (w, h), interpolation=cv2.INTER_NEAREST)
        fish_prob_r = cv2.resize(fish_prob, (w, h))
        confidence_r = cv2.resize(confidence, (w, h))

        # ── Multi-frame temporal voting (images: single frame, videos: smoothed) ──
        self.masks.append(mask)
        if len(self.masks) > 1:
            avg_mask = np.mean(np.array(self.masks), axis=0)
            mask = (avg_mask > 0.5).astype("uint8")

        # ── Morphological cleanup ─────────────────────────────────────────────
        # OPEN  (3×3): removes isolated noise specks / salt-pepper artifacts.
        # CLOSE (5×5): fills small interior holes within a single fish's mask blob,
        #   giving solid filled contours for visualization. Kernel intentionally kept
        #   at 5×5 (not 7×7) so fish that are >2px apart at mask resolution are
        #   NOT merged together.
        kernel_open  = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # ── Connected-component filtering ─────────────────────────────────────
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)

        # Max blob area = 25% of the image. Anything larger (sky, water regions)
        # is almost certainly a false positive, not a single fish or animal.
        max_area = int(0.25 * h * w)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw   = stats[i, cv2.CC_STAT_WIDTH]
            bh_  = stats[i, cv2.CC_STAT_HEIGHT]

            # Skip blobs that are too small (noise) — min_area=150 keeps small fish
            if area < min_area:
                continue

            # Skip blobs that are implausibly large (water, sky, full-frame artifacts)
            if area > max_area:
                continue

            # Skip blobs with extreme aspect ratios (very wide strips = water/sky bands)
            aspect = max(bw, bh_) / (min(bw, bh_) + 1e-6)
            if aspect > 5.0:
                continue

            clean[labels == i] = 1

        return clean, fish_prob_r