"""
Divides frames into zones and monitors density per zone
"""

import numpy as np
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class Zone:
    """Represents a single zone in the grid"""
    def __init__(self, zone_id, x1, y1, x2, y2):
        self.zone_id = zone_id
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.width = x2 - x1
        self.height = y2 - y1
        self.area = self.width * self.height
        
        self.density_score = 0.0
        self.density_level = "LOW"
        self.pixel_count = 0
        
    def get_roi(self):
        """Return zone coordinates as (x1, y1, x2, y2)"""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def get_center(self):
        """Return zone center coordinates"""
        cx = (self.x1 + self.x2) // 2
        cy = (self.y1 + self.y2) // 2
        return (cx, cy)


class ZonalDensityMonitor:
    """
    Monitors density across multiple zones in a frame.
    Supports grid-based zone division with configurable thresholds.
    """
    
    def __init__(self, 
        grid_rows=3, 
        grid_cols=3,
        low_threshold=0.01,
        medium_threshold=0.05,
        high_threshold=0.15,
        enable_logging=True,
        log_dir="zonal_logs"):
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_zones = grid_rows * grid_cols
        
        # Density thresholds
        self.low_th = low_threshold
        self.medium_th = medium_threshold
        self.high_th = high_threshold
        
        # Logging
        self.enable_logging = enable_logging
        self.log_dir = Path(log_dir)
        if enable_logging:
            self.log_dir.mkdir(exist_ok=True)
        
        self.zones = {}
        self.frame_history = defaultdict(list)
        self.lock = threading.Lock()
        
    def initialize_zones(self, frame_height, frame_width):
        """Create zone grid based on frame dimensions"""
        self.zones = {}
        
        zone_height = frame_height // self.grid_rows
        zone_width = frame_width // self.grid_cols
        
        zone_id = 0
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * zone_width
                y1 = row * zone_height
                x2 = x1 + zone_width if col < self.grid_cols - 1 else frame_width
                y2 = y1 + zone_height if row < self.grid_rows - 1 else frame_height
                
                zone = Zone(zone_id, x1, y1, x2, y2)
                self.zones[zone_id] = zone
                zone_id += 1
    
    def analyze_frame(self, mask):
        """
        Analyze density for each zone in the frame.
        
        Args:
            mask: Binary mask (H x W) where 1 indicates foreground
        
        Returns:
            dict: Zone densities with format {zone_id: {'score': float, 'level': str}}
        """
        if not self.zones:
            h, w = mask.shape[:2]
            self.initialize_zones(h, w)
        
        results = {}
        frame_timestamp = datetime.now()
        
        with self.lock:
            for zone_id, zone in self.zones.items():
                # Extract zone ROI
                zone_mask = mask[zone.y1:zone.y2, zone.x1:zone.x2]
                
                # Calculate density
                pixel_count = zone_mask.sum()
                density_score = pixel_count / zone.area if zone.area > 0 else 0.0
                
                # Classify density level
                if density_score < self.low_th:
                    density_level = "LOW"
                elif density_score < self.medium_th:
                    density_level = "LOW-MEDIUM"
                elif density_score < self.high_th:
                    density_level = "MEDIUM-HIGH"
                else:
                    density_level = "HIGH"
                
                # Update zone
                zone.density_score = density_score
                zone.density_level = density_level
                zone.pixel_count = int(pixel_count)
                
                results[zone_id] = {
                    'score': density_score,
                    'level': density_level,
                    'pixels': zone.pixel_count,
                    'threshold': self._get_threshold_for_level(density_level)
                }
                
                # Store history
                self.frame_history[zone_id].append({
                    'timestamp': frame_timestamp,
                    'score': density_score,
                    'level': density_level
                })
        
        return results
    
    def _get_threshold_for_level(self, level):
        """Get the density threshold for a given level"""
        thresholds = {
            'LOW': self.low_th,
            'LOW-MEDIUM': self.medium_th,
            'MEDIUM-HIGH': self.high_th,
            'HIGH': self.high_th
        }
        return thresholds.get(level, self.low_th)
    
    def get_zone_stats(self, zone_id):
        """Get statistics for a specific zone"""
        if zone_id not in self.frame_history:
            return None
        
        history = self.frame_history[zone_id]
        scores = [h['score'] for h in history]
        
        return {
            'zone_id': zone_id,
            'mean_density': np.mean(scores) if scores else 0,
            'max_density': np.max(scores) if scores else 0,
            'min_density': np.min(scores) if scores else 0,
            'std_density': np.std(scores) if scores else 0,
            'frame_count': len(history)
        }
    
    def log_analysis(self, results, frame_num=None):
        """Log analysis results to file"""
        if not self.enable_logging:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_file = self.log_dir / "density_log.txt"
        
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {timestamp}\n")
            if frame_num is not None:
                f.write(f"Frame: {frame_num}\n")
            f.write(f"{'='*60}\n")
            
            for zone_id, data in sorted(results.items()):
                f.write(f"Zone {zone_id}: {data['level']} ")
                f.write(f"(Score: {data['score']:.4f}, Pixels: {data['pixels']})\n")
    
    def get_alerts(self, results, threshold_level="HIGH"):
        """
        Get alerts for zones exceeding a threshold level.
        
        Args:
            results: Dict of zone analysis results
            threshold_level: Alert if zone reaches this level or higher
        
        Returns:
            list: Zone IDs that exceeded threshold
        """
        level_hierarchy = {
            'LOW': 0,
            'LOW-MEDIUM': 1,
            'MEDIUM-HIGH': 2,
            'HIGH': 3
        }
        
        threshold_value = level_hierarchy.get(threshold_level, 3)
        alerts = []
        
        for zone_id, data in results.items():
            zone_level_value = level_hierarchy.get(data['level'], 0)
            if zone_level_value >= threshold_value:
                alerts.append(zone_id)
        
        return alerts
    
    def reset_history(self):
        """Clear frame history"""
        with self.lock:
            self.frame_history.clear()
