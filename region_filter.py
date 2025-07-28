#!/usr/bin/env python3
"""
Region Filtering Module
Filters out non-text regions using various criteria
"""

import cv2
import numpy as np
import logging
from config import FILTER_CONFIG
from image_utils import get_connected_components, calculate_contour_properties

logger = logging.getLogger(__name__)


class RegionFilter:
    """
    Filters regions to identify text areas
    """
    
    def __init__(self, config=None):
        """
        Initialize region filter with configuration
        
        Args:
            config: Dictionary of filtering parameters
        """
        self.config = config or FILTER_CONFIG
        
    def check_size_constraints(self, area, width, height, total_image_area):
        """
        Check if region meets size constraints
        
        Args:
            area: Region area
            width: Region width
            height: Region height
            total_image_area: Total image area
            
        Returns:
            bool: True if meets constraints
        """
        # Area constraints
        if area < self.config['min_text_area'] or area > self.config['max_text_area']:
            return False
        
        # Check relative to image size
        if area > total_image_area * 0.4:
            return False
        
        # Dimension constraints
        if height < self.config['min_height'] or height > self.config['max_height']:
            return False
        
        if width < self.config['min_width']:
            return False
        
        return True
    
    def check_shape_constraints(self, properties):
        """
        Check if region meets shape constraints
        
        Args:
            properties: Dictionary of contour properties
            
        Returns:
            bool: True if meets constraints
        """
        # Aspect ratio
        aspect_ratio = properties['aspect_ratio']
        if (aspect_ratio < self.config['min_aspect_ratio'] or 
            aspect_ratio > self.config['max_aspect_ratio']):
            return False
        
        # Solidity
        if properties['solidity'] < self.config['min_solidity']:
            return False
        
        # Compactness
        compactness = properties['compactness']
        if (compactness < self.config['min_compactness'] or 
            compactness > self.config['max_compactness']):
            return False
        
        # Circularity
        circularity = properties['circularity']
        if (circularity > self.config['max_circularity'] or 
            circularity < self.config['min_circularity']):
            return False
        
        # Fill ratio (extent)
        if properties['extent'] < self.config['min_fill_ratio']:
            return False
        
        return True
    
    def check_texture_constraints(self, region_mask, edge_density):
        """
        Check if region meets texture constraints
        
        Args:
            region_mask: Binary mask of the region
            edge_density: Edge density of the region
            
        Returns:
            bool: True if meets constraints
        """
        # Edge density check
        if (edge_density < self.config['min_edge_density'] or 
            edge_density > self.config['max_edge_density']):
            return False
        
        # Stroke width variation (for larger regions)
        height, width = region_mask.shape
        if width > 20 and height > 20:
            dist_transform = cv2.distanceTransform(region_mask, cv2.DIST_L2, 5)
            if dist_transform.max() > 0:
                stroke_variation = dist_transform.std() / dist_transform.max()
                if stroke_variation > self.config['max_stroke_variation']:
                    return False
        
        return True
    
    def filter_single_region(self, labels, label_id, stats, intersection_mask):
        """
        Filter a single connected component region
        
        Args:
            labels: Label image from connected components
            label_id: ID of the current label
            stats: Statistics from connected components
            intersection_mask: Binary mask to analyze
            
        Returns:
            tuple: (passes_filter, region_properties)
        """
        # Get basic properties
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        img_height, img_width = labels.shape
        total_image_area = img_height * img_width
        
        # Check size constraints
        if not self.check_size_constraints(area, width, height, total_image_area):
            return False, None
        
        # Get region mask
        region_mask = (labels == label_id).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None
        
        contour = contours[0]
        contour_area = cv2.contourArea(contour)
        
        # Skip if contour area is too small
        if contour_area < self.config['min_contour_area']:
            return False, None
        
        # Calculate contour properties
        properties = calculate_contour_properties(contour)
        
        # Check shape constraints
        if not self.check_shape_constraints(properties):
            return False, None
        
        # Calculate edge density
        roi = intersection_mask[y:y+height, x:x+width]
        edge_pixels = np.sum(roi > 0)
        edge_density = edge_pixels / (width * height) if width * height > 0 else 0
        
        # Check texture constraints
        if not self.check_texture_constraints(roi, edge_density):
            return False, None
        
        # Region passed all checks
        region_properties = {
            'bbox': (x, y, width, height),
            'area': int(area),
            'aspect_ratio': float(properties['aspect_ratio']),
            'solidity': float(properties['solidity']),
            'fill_ratio': float(properties['extent']),
            'edge_density': float(edge_density),
            'compactness': float(properties['compactness']),
            'circularity': float(properties['circularity'])
        }
        
        return True, region_properties
    
    def filter_regions(self, intersection_mask):
        """
        Filter all regions in the intersection mask
        
        Args:
            intersection_mask: Binary mask with potential text regions
            
        Returns:
            tuple: (filtered_mask, valid_regions)
        """
        logger.info("Filtering regions with enhanced criteria")
        
        # Find connected components
        num_labels, labels, stats, centroids = get_connected_components(intersection_mask)
        
        # Create output mask
        filtered_mask = np.zeros_like(intersection_mask)
        valid_regions = []
        
        # Process each component (skip background label 0)
        for i in range(1, num_labels):
            passes_filter, region_properties = self.filter_single_region(
                labels, i, stats, intersection_mask
            )
            
            if passes_filter and region_properties is not None:
                # Add to filtered mask
                filtered_mask[labels == i] = 255
                valid_regions.append(region_properties)
        
        logger.info(f"Filtered to {len(valid_regions)} valid regions from {num_labels-1} total")
        return filtered_mask, valid_regions