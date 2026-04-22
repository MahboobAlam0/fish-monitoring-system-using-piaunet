import cv2
import numpy as np
from config import MIN_AREA

def extract_instances(mask, prob_map):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 10 or h < 10:
            continue

        if w * h < MIN_AREA:
            continue

        region = prob_map[y:y+h, x:x+w]

        # prevent empty slice bug
        if region.size == 0:
            continue

        conf = float(region.mean())

        detections.append([x, y, x+w, y+h, conf])

    return np.array(detections) if len(detections) > 0 else np.empty((0,5))