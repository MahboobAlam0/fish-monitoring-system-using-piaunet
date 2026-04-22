"""
Zonal visualization module - Draw zones, heatmaps, and density overlays
"""

import cv2
import numpy as np


class ZonalVisualizer:
    """Handles visualization of zonal density analysis"""
    
    # Color mapping for density levels
    DENSITY_COLORS = {
        'LOW': (0, 255, 0),           # Green
        'LOW-MEDIUM': (0, 255, 255),  # Cyan
        'MEDIUM-HIGH': (0, 165, 255), # Orange
        'HIGH': (0, 0, 255)           # Red
    }
    
    def __init__(self, monitor):
        """
        Initialize visualizer with a ZonalDensityMonitor instance
        
        Args:
            monitor: ZonalDensityMonitor instance
        """
        self.monitor = monitor
    
    def draw_zones(self, frame, show_labels=True, alpha=0.3, label_position='outside'):
        """
        Draw zone grid on frame with density color coding
        
        Args:
            frame: Input frame (H x W x 3)
            show_labels: Whether to show zone ID and density info
            alpha: Transparency for zone overlays
            label_position: 'inside' or 'outside' for label placement
        
        Returns:
            Annotated frame
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw zone rectangles with color coding
        for zone_id, zone in self.monitor.zones.items():
            x1, y1, x2, y2 = zone.get_roi()
            
            # Get color based on density level
            color = self.DENSITY_COLORS.get(zone.density_level, (255, 255, 255))
            
            # Draw filled rectangle with transparency
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Draw border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Blend overlay with original frame (without zone labels - clean and simple)
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def draw_heatmap(self, frame, mask, alpha=0.5):
        """
        Draw density heatmap based on mask
        
        Args:
            frame: Input frame (H x W x 3)
            mask: Density mask (H x W)
            alpha: Transparency for heatmap
        
        Returns:
            Frame with heatmap overlay
        """
        # Normalize mask to 0-255
        mask_normalized = cv2.normalize(mask, np.zeros_like(mask), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap (COLORMAP_JET: cool colors for low density, hot colors for high)
        heatmap = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return result
    
    def draw_zone_alerts(self, frame, alerts):
        """
        Highlight zones with alerts
        
        Args:
            frame: Input frame (H x W x 3)
            alerts: List of zone IDs with alerts
        
        Returns:
            Frame with alert highlights
        """
        result = frame.copy()
        
        for zone_id in alerts:
            if zone_id in self.monitor.zones:
                zone = self.monitor.zones[zone_id]
                x1, y1, x2, y2 = zone.get_roi()
                
                # Draw thick red border for alert
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
        return result
    
    def get_legend_data(self, results):
        """
        Extract legend data from results for display in separate UI box
        
        Args:
            results: Zone analysis results dictionary
        
        Returns:
            Dictionary with legend information
        """
        level_counts = {
            'LOW': 0,
            'LOW-MEDIUM': 0,
            'MEDIUM-HIGH': 0,
            'HIGH': 0
        }
        
        for data in results.values():
            level = data.get('level', 'UNKNOWN')
            if level in level_counts:
                level_counts[level] += 1
        
        return {
            'total': len(results),
            'LOW': level_counts['LOW'],
            'LOW-MEDIUM': level_counts['LOW-MEDIUM'],
            'MEDIUM-HIGH': level_counts['MEDIUM-HIGH'],
            'HIGH': level_counts['HIGH']
        }
    
    def draw_summary(self, frame, results):
        """
        Draw summary statistics panel on frame (right side)
        
        Args:
            frame: Input frame
            results: Zone analysis results dictionary
        
        Returns:
            Frame with summary panel overlay
        """
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Count zones by density level
        level_counts = {
            'LOW': 0,
            'LOW-MEDIUM': 0,
            'MEDIUM-HIGH': 0,
            'HIGH': 0
        }
        
        for data in results.values():
            level = data.get('level', 'UNKNOWN')
            if level in level_counts:
                level_counts[level] += 1
        
        # Create summary panel on right side
        panel_width = 140
        panel_height = 130
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Draw semi-transparent dark background
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1
        )
        result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
        
        # Draw border
        cv2.rectangle(
            result,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (100, 100, 100),
            2
        )
        
        # Draw title
        cv2.putText(
            result,
            "ZONE SUMMARY",
            (panel_x + 8, panel_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )
        
        # Draw statistics
        stats_lines = [
            (f"Total: {len(results)}", (255, 255, 255)),
            (f"LOW: {level_counts['LOW']}", (0, 255, 0)),
            (f"LOW-MED: {level_counts['LOW-MEDIUM']}", (0, 255, 255)),
            (f"MED-HIGH: {level_counts['MEDIUM-HIGH']}", (0, 165, 255)),
            (f"HIGH: {level_counts['HIGH']}", (0, 0, 255))
        ]
        
        y = panel_y + 35
        for text, color in stats_lines:
            cv2.putText(
                result,
                text,
                (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1
            )
            y += 18
        
        return result
    
    def draw_mask_overlay(self, frame, mask, alpha=0.6, style='instance', min_area=50):
        """
        Draw bounding box with mask only on the species.
        
        Args:
            frame: Input frame (H x W x 3)
            mask: Binary segmentation mask (H x W)
            alpha: Transparency level (0-1), higher = more visible
            style: Ignored, kept for compatibility
            min_area: Minimum area for an object to be considered
        
        Returns:
            Frame with instance masks and bounding boxes overlay
        """
        result = frame.copy()
        overlay = frame.copy()
        
        if mask is not None:
            # Find contours in the mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Distinct glowing neon palette for underwater subjects
            target_colors = [
                (0, 255, 255),    # Neon Yellow
                (255, 0, 255),    # Neon Magenta
                (50, 255, 50),    # Neon Green
                (255, 50, 50),    # Bright Blue
                (50, 50, 255),    # Bright Red
                (0, 165, 255),    # Deep Orange
                (200, 100, 255),  # Violet
                (0, 255, 150)     # Aquamarine
            ]
            
            object_id = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip small contours
                if area < min_area:
                    continue
                
                color = target_colors[object_id % len(target_colors)]
                object_id += 1
                
                # Draw filled mask for this contour on the overlay
                cv2.drawContours(overlay, [contour], 0, color, -1)
            
            # Blend overlay with original frame
            result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
            
            # Draw bounding boxes ON TOP of the blended result so they are 100% opaque
            object_id = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                color = target_colors[object_id % len(target_colors)]
                object_id += 1
                
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        return result
    
    def draw_object_contours(self, frame, mask, prob_map=None, min_area=50, thickness=2):
        """
        Draw contours of detected underwater objects with confidence levels
        
        Args:
            frame: Input frame (H x W x 3)
            mask: Binary segmentation mask (H x W)
            prob_map: Confidence/probability map (H x W) - optional
            min_area: Minimum contour area to display
            thickness: Contour line thickness
        
        Returns:
            Frame with object contours drawn
        """
        result = frame.copy()
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Color for contours (bright cyan)
        contour_color = (0, 255, 255)
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Skip small contours
            if area < min_area:
                continue
            
            # Draw contour
            cv2.drawContours(result, [contour], 0, contour_color, thickness)
            
            # Calculate confidence if prob_map is provided
            confidence = None
            if prob_map is not None:
                # Get mask of this contour
                contour_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], 0, 1, -1)
                # Calculate average probability within contour
                contour_pixels = prob_map[contour_mask == 1]
                if len(contour_pixels) > 0:
                    confidence = float(np.mean(contour_pixels))
            
            # Get bounding rectangle for label placement
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw confidence label if available
            if confidence is not None:
                conf_text = f"Conf: {confidence:.2f}"
                cv2.putText(
                    result,
                    conf_text,
                    (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1
                )
        
        return result
    
    def draw_object_labels(self, frame, mask, prob_map=None, min_area=50):
        """
        Label individual detected underwater objects with ID, size, and confidence
        
        Args:
            frame: Input frame (H x W x 3)
            mask: Binary segmentation mask (H x W)
            prob_map: Confidence/probability map (H x W) - optional
            min_area: Minimum object area to label
        
        Returns:
            Frame with labeled objects
        """
        result = frame.copy()
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        object_id = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip small contours
            if area < min_area:
                continue
            
            object_id += 1
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            
            # Get centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Calculate confidence if prob_map provided
            confidence = None
            if prob_map is not None:
                contour_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], 0, 1, -1)
                contour_pixels = prob_map[contour_mask == 1]
                if len(contour_pixels) > 0:
                    confidence = float(np.mean(contour_pixels))
            
            # Draw bounding box
            box_color = (0, 255, 0) if confidence is None or confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(result, (x, y), (x + w, y + h), box_color, 2)
            
            # Draw centroid
            cv2.circle(result, (cx, cy), 3, (0, 255, 255), -1)
            
            # Build label text
            label_text = f"Obj#{object_id}"
            size_text = f"Area:{int(area)}"
            conf_text = f"Conf:{confidence:.2f}" if confidence is not None else ""
            
            # Draw label background box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Draw background rectangle for label
            label_bg_color = (50, 50, 50)
            cv2.rectangle(
                result,
                (x, y - text_h - baseline - 10),
                (x + text_w + 10, y - baseline),
                label_bg_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                result,
                label_text,
                (x + 5, y - baseline - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
            
            # Draw size info (below bounding box)
            cv2.putText(
                result,
                size_text,
                (x + 5, y + h + 15),
                font,
                0.4,
                (0, 255, 255),
                1
            )
            
            # Draw confidence info (below size if available)
            if confidence is not None:
                cv2.putText(
                    result,
                    conf_text,
                    (x + 5, y + h + 30),
                    font,
                    0.4,
                    box_color,
                    1
                )
        return result
    
    def draw_full_analysis(self, frame, mask, results, alerts=None, 
                          show_zones=True, show_heatmap=True, 
                          show_summary=True, show_alerts=True,
                          show_contours=False, show_labels=False,
                          show_mask=True, mask_style='instance', prob_map=None,
                          show_zone_labels=True, show_summary_panel=True):
        """
        Draw complete zonal density analysis visualization
        
        Args:
            frame: Input frame
            mask: Density mask
            results: Zone analysis results
            alerts: List of zone IDs with alerts (optional)
            show_zones: Whether to draw zone grid
            show_heatmap: Whether to draw heatmap
            show_summary: Whether to draw summary stats
            show_alerts: Whether to highlight alerts
            show_contours: Whether to draw object contours with confidence
            show_labels: Whether to draw individual object labels
            show_mask: Whether to draw raw mask overlay
            mask_style: Mask color style ('dark', 'green', 'red', 'blue', 'white', 'inverse')
            prob_map: Probability/confidence map (optional, for contours and labels)
            show_zone_labels: Whether to show zone ID and density score labels
            show_summary_panel: Whether to show the summary statistics panel
        
        Returns:
            Fully annotated frame
        """
        result = frame.copy()
        
        # Step 1: Heatmap first (background layer) — use prob_map for smooth gradient.
        # Passing the binary mask to JET colormap creates a flat color flood;
        # prob_map gives a continuous confidence gradient that is visually informative.
        if show_heatmap:
            heatmap_data = prob_map if prob_map is not None else mask.astype(np.float32)
            result = self.draw_heatmap(result, heatmap_data, alpha=0.15)
        
        # Step 2: Instance mask overlay on top of heatmap (alpha=0.55 ensures fills are clearly visible)
        if show_mask:
            result = self.draw_mask_overlay(result, mask, alpha=0.55, style=mask_style)
        
        # Step 3: Zone grid
        if show_zones:
            label_pos = 'outside' if show_zone_labels else 'inside'
            result = self.draw_zones(result, show_labels=show_zone_labels, alpha=0.15, label_position=label_pos)
        
        # Step 4: Object contours with confidence
        if show_contours:
            result = self.draw_object_contours(result, mask, prob_map=prob_map)
        
        # Step 5: Individual object labels
        if show_labels:
            result = self.draw_object_labels(result, mask, prob_map=prob_map)
        
        # Step 6: Alert highlights (drawn last so borders are always visible)
        if show_alerts and alerts:
            result = self.draw_zone_alerts(result, alerts)
        
        # Step 7: Summary statistics panel
        if show_summary and show_summary_panel:
            result = self.draw_summary(result, results)
        
        return result
