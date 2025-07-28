#!/usr/bin/env python3
"""
Morphological Operations Module
Handles morphological operations for connecting and cleaning text regions
"""

import cv2
import numpy as np
import logging
from config import MORPHOLOGY_CONFIG
from image_utils import get_connected_components

logger = logging.getLogger(__name__)


class MorphologyProcessor:
    """
    Applies morphological operations to connect text regions
    """
    
    def __init__(self, config=None):
        """
        Initialize morphology processor with configuration
        
        Args:
            config: Dictionary of morphology parameters
        """
        self.config = config or MORPHOLOGY_CONFIG
        
    def create_kernel(self, kernel_type, size=None):
        """
        Create morphological kernel
        
        Args:
            kernel_type: Type of kernel ('small', 'horizontal', 'vertical', 'ellipse', or 'custom')
            size: Kernel size for custom kernel
            
        Returns:
            kernel: Numpy array kernel
        """
        if kernel_type in self.config['kernels']:
            size = self.config['kernels'][kernel_type]
        elif size is None:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        if kernel_type == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        elif kernel_type in ['horizontal', 'vertical']:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        else:
            kernel = np.ones(size, np.uint8)
        
        return kernel
    
    def fill_small_gaps(self, mask):
        """
        Fill small gaps within regions
        
        Args:
            mask: Binary mask
            
        Returns:
            filled: Mask with small gaps filled
        """
        kernel = self.create_kernel('small')
        filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return filled
    
    def connect_horizontal_text(self, mask):
        """
        Connect text regions horizontally (for same-line text)
        
        Args:
            mask: Binary mask
            
        Returns:
            connected: Horizontally connected mask
        """
        kernel = self.create_kernel('horizontal')
        connected = cv2.dilate(mask, kernel, iterations=1)
        return connected
    
    def connect_vertical_text(self, mask):
        """
        Connect text regions vertically (for multi-line text)
        
        Args:
            mask: Binary mask
            
        Returns:
            connected: Vertically connected mask
        """
        kernel = self.create_kernel('vertical')
        connected = cv2.dilate(mask, kernel, iterations=1)
        return connected
    
    def smooth_boundaries(self, mask):
        """
        Smooth region boundaries
        
        Args:
            mask: Binary mask
            
        Returns:
            smoothed: Mask with smoothed boundaries
        """
        kernel = self.create_kernel('ellipse')
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return smoothed
    
    def remove_small_components(self, mask):
        """
        Remove small noise components
        
        Args:
            mask: Binary mask
            
        Returns:
            cleaned: Mask with small components removed
        """
        # Find connected components
        num_labels, labels, stats, _ = get_connected_components(mask)
        
        # Create cleaned mask
        cleaned_mask = np.zeros_like(mask)
        min_area = self.config['min_component_area']
        
        # Keep only components above minimum area
        components_removed = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == i] = 255
            else:
                components_removed += 1
        
        if components_removed > 0:
            logger.info(f"Removed {components_removed} small components")
        
        return cleaned_mask
    
    def apply_morphology(self, mask):
        """
        Apply complete morphological pipeline
        
        Args:
            mask: Binary mask
            
        Returns:
            processed: Mask after all morphological operations
        """
        logger.info("Applying morphological operations")
        
        # Step 1: Fill small gaps
        filled = self.fill_small_gaps(mask)
        
        # Step 2: Connect horizontal text
        connected_h = self.connect_horizontal_text(filled)
        
        # Step 3: Light vertical connection
        connected_v = self.connect_vertical_text(filled)
        
        # Step 4: Combine both directions
        combined = cv2.bitwise_or(connected_h, connected_v)
        
        # Step 5: Smooth boundaries
        smoothed = self.smooth_boundaries(combined)
        
        # Step 6: Remove small noise components
        cleaned = self.remove_small_components(smoothed)
        
        logger.info("Morphology operations completed")
        return cleaned
    
    def adaptive_morphology(self, mask, region_stats):
        """
        Apply adaptive morphology based on region characteristics
        
        Args:
            mask: Binary mask
            region_stats: Statistics about regions in the mask
            
        Returns:
            processed: Adaptively processed mask
        """
        # Analyze region characteristics
        if region_stats:
            avg_height = np.mean([stat['bbox'][3] for stat in region_stats])
            avg_width = np.mean([stat['bbox'][2] for stat in region_stats])
            
            # Adapt kernel sizes based on average text size
            h_kernel_width = int(min(avg_height * 0.8, 10))
            v_kernel_height = int(min(avg_width * 0.3, 5))
            
            # Create adaptive kernels
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_height))
            
            # Apply adaptive morphology
            filled = self.fill_small_gaps(mask)
            connected_h = cv2.dilate(filled, h_kernel, iterations=1)
            connected_v = cv2.dilate(filled, v_kernel, iterations=1)
            combined = cv2.bitwise_or(connected_h, connected_v)
            smoothed = self.smooth_boundaries(combined)
            cleaned = self.remove_small_components(smoothed)
            
            return cleaned
        else:
            # Fall back to standard morphology
            return self.apply_morphology(mask)