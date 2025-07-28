#!/usr/bin/env python3
"""
Region Merging Module
Handles merging of nearby text regions into coherent text blocks
"""

import cv2
import numpy as np
import logging
from config import MERGE_CONFIG
from image_utils import calculate_contour_properties

logger = logging.getLogger(__name__)


class RegionMerger:
    """
    Merges nearby text regions
    """
    
    def __init__(self, config=None):
        """
        Initialize region merger with configuration
        
        Args:
            config: Dictionary of merging parameters
        """
        self.config = config or MERGE_CONFIG
        
    def extract_potential_boxes(self, morphology_mask):
        """
        Extract potential text boxes from morphology mask
        
        Args:
            morphology_mask: Binary mask after morphological operations
            
        Returns:
            list: List of potential box dictionaries
        """
        # Find contours
        contours, _ = cv2.findContours(morphology_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_boxes = []
        
        for contour in contours:
            # Calculate properties
            properties = calculate_contour_properties(contour)
            area = properties['area']
            x, y, w, h = properties['bbox']
            
            # Initial filtering
            if area < self.config['min_initial_area']:
                continue
            
            aspect_ratio = properties['aspect_ratio']
            min_ar, max_ar = 0.15, 15  # Broad initial range
            if aspect_ratio < min_ar or aspect_ratio > max_ar:
                continue
            
            if h < self.config['min_box_height'] or w < self.config['min_box_width']:
                continue
            
            if properties['solidity'] < self.config['min_box_solidity']:
                continue
            
            potential_boxes.append({
                'box': (x, y, w, h),
                'area': area,
                'contour': contour,
                'solidity': properties['solidity'],
                'properties': properties
            })
        
        return potential_boxes
    
    def check_merge_criteria(self, box1, box2):
        """
        Check if two boxes should be merged
        
        Args:
            box1: First box dictionary
            box2: Second box dictionary
            
        Returns:
            bool: True if boxes should be merged
        """
        x1, y1, w1, h1 = box1['box']
        x2, y2, w2, h2 = box2['box']
        
        # Calculate distances between boxes
        x_dist = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        y_dist = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
        
        max_height = max(h1, h2)
        
        # Check horizontal merging
        if (x_dist < max_height * self.config['horizontal_merge_factor'] and 
            y_dist < max_height * self.config['vertical_merge_factor']):
            # Check vertical alignment
            y_center1 = y1 + h1 / 2
            y_center2 = y2 + h2 / 2
            if abs(y_center1 - y_center2) < max_height * self.config['alignment_threshold']:
                return True
        
        # Check vertical merging
        if (y_dist < max_height * 0.8 and x_dist < max_height * 0.2):
            # Check horizontal alignment
            x_center1 = x1 + w1 / 2
            x_center2 = x2 + w2 / 2
            if abs(x_center1 - x_center2) < max(w1, w2) * self.config['alignment_threshold']:
                return True
        
        return False
    
    def merge_boxes(self, potential_boxes):
        """
        Merge nearby boxes
        
        Args:
            potential_boxes: List of potential box dictionaries
            
        Returns:
            list: List of merged boxes (x, y, w, h)
        """
        if not potential_boxes:
            return []
        
        merged_groups = []
        used = [False] * len(potential_boxes)
        
        # Find groups of boxes to merge
        for i, box1 in enumerate(potential_boxes):
            if used[i]:
                continue
            
            # Start new merge group
            merge_group = [i]
            used[i] = True
            
            # Find all boxes that should merge with this group
            changed = True
            while changed:
                changed = False
                for j, box2 in enumerate(potential_boxes):
                    if used[j]:
                        continue
                    
                    # Check if box2 should merge with any box in current group
                    for idx in merge_group:
                        if self.check_merge_criteria(potential_boxes[idx], box2):
                            merge_group.append(j)
                            used[j] = True
                            changed = True
                            break
            
            merged_groups.append(merge_group)
        
        # Create merged boxes from groups
        merged_boxes = []
        
        for group in merged_groups:
            # Calculate bounding box of all boxes in group
            min_x = min(potential_boxes[idx]['box'][0] for idx in group)
            min_y = min(potential_boxes[idx]['box'][1] for idx in group)
            max_x = max(potential_boxes[idx]['box'][0] + potential_boxes[idx]['box'][2] for idx in group)
            max_y = max(potential_boxes[idx]['box'][1] + potential_boxes[idx]['box'][3] for idx in group)
            
            merged_w = max_x - min_x
            merged_h = max_y - min_y
            merged_area = sum(potential_boxes[idx]['area'] for idx in group)
            avg_solidity = np.mean([potential_boxes[idx]['solidity'] for idx in group])
            
            # Apply final filtering
            if self.apply_final_filtering(min_x, min_y, merged_w, merged_h, 
                                        merged_area, avg_solidity):
                merged_boxes.append((min_x, min_y, merged_w, merged_h))
        
        return merged_boxes
    
    def apply_final_filtering(self, x, y, w, h, area, avg_solidity):
        """
        Apply final filtering to merged box
        
        Args:
            x, y, w, h: Box coordinates and dimensions
            area: Total area of merged regions
            avg_solidity: Average solidity of merged regions
            
        Returns:
            bool: True if box passes filtering
        """
        # Check area
        if area < self.config['min_area_threshold']:
            if area >= self.config['min_merge_area']:
                if w > self.config['min_merge_width'] and h > self.config['min_merge_height']:
                    aspect_ratio = w / h
                    min_ar, max_ar = self.config['merge_aspect_ratio_range']
                    if not (min_ar <= aspect_ratio <= max_ar):
                        return False
                else:
                    return False
            else:
                return False
        
        # Check aspect ratio for all boxes
        aspect_ratio = w / h
        if aspect_ratio < 0.2 or aspect_ratio > 12:
            return False
        
        # Check dimensions
        if h < self.config['min_box_height'] or w < self.config['min_merge_width']:
            return False
        
        # Check solidity
        if avg_solidity < self.config['min_avg_solidity']:
            return False
        
        return True
    
    def remove_overlapping_boxes(self, boxes):
        """
        Remove overlapping boxes, keeping larger ones
        
        Args:
            boxes: List of boxes (x, y, w, h)
            
        Returns:
            list: List of non-overlapping boxes
        """
        if not boxes:
            return []
        
        # Sort by area (largest first)
        sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        
        final_boxes = []
        
        for box in sorted_boxes:
            x1, y1, w1, h1 = box
            overlap = False
            
            for final_box in final_boxes:
                x2, y2, w2, h2 = final_box
                
                # Calculate intersection
                ix1 = max(x1, x2)
                iy1 = max(y1, y2)
                ix2 = min(x1 + w1, x2 + w2)
                iy2 = min(y1 + h1, y2 + h2)
                
                if ix2 > ix1 and iy2 > iy1:  # There is overlap
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    
                    # Check overlap threshold
                    if intersection > min(area1, area2) * self.config['overlap_threshold']:
                        overlap = True
                        break
            
            if not overlap:
                final_boxes.append(box)
        
        return final_boxes
    
    def merge_text_regions(self, morphology_mask, img_shape=None):
        """
        Complete region merging pipeline
        
        Args:
            morphology_mask: Binary mask after morphological operations
            img_shape: Image dimensions for border checking
            
        Returns:
            list: List of final bounding boxes (x, y, w, h)
        """
        logger.info("Starting region merging")
        
        # Extract potential boxes
        potential_boxes = self.extract_potential_boxes(morphology_mask)
        logger.info(f"Found {len(potential_boxes)} potential boxes")
        
        # Apply border filtering if image shape provided
        if img_shape is not None:
            potential_boxes = self.filter_border_boxes(potential_boxes, img_shape)
        
        # Merge nearby boxes
        merged_boxes = self.merge_boxes(potential_boxes)
        logger.info(f"Created {len(merged_boxes)} merged boxes")
        
        # Remove overlapping boxes
        final_boxes = self.remove_overlapping_boxes(merged_boxes)
        logger.info(f"Final number of text regions: {len(final_boxes)}")
        
        return final_boxes
    
    def filter_border_boxes(self, boxes, img_shape):
        """
        Apply special filtering for boxes near image borders
        
        Args:
            boxes: List of box dictionaries
            img_shape: Image dimensions (height, width)
            
        Returns:
            list: Filtered boxes
        """
        img_height, img_width = img_shape
        filtered = []
        
        for box_dict in boxes:
            x, y, w, h = box_dict['box']
            
            # Check if near border
            if (x < self.config['border_threshold'] or 
                y < self.config['border_threshold'] or 
                x + w > img_width - self.config['border_threshold'] or 
                y + h > img_height - self.config['border_threshold']):
                
                # Apply stricter criteria for border boxes
                if (box_dict['area'] >= self.config['border_min_area'] and 
                    h >= self.config['border_min_height']):
                    filtered.append(box_dict)
            else:
                filtered.append(box_dict)
        
        return filtered