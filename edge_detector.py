#!/usr/bin/env python3
"""
Edge Detection Module
Handles edge detection operations for text extraction
"""

import cv2
import numpy as np
import logging
from config import EDGE_CONFIG, PREPROCESSING_CONFIG
from image_utils import apply_bilateral_filter, calculate_sobel_gradients, apply_morphology_operation

logger = logging.getLogger(__name__)


class EdgeDetector:
    """
    Edge detector for text region extraction
    """
    
    def __init__(self, config=None):
        """
        Initialize edge detector with configuration
        
        Args:
            config: Dictionary of edge detection parameters
        """
        self.config = config or EDGE_CONFIG
        self.preprocessing_config = PREPROCESSING_CONFIG['bilateral_filter']['edge_detection']
        
    def preprocess_for_edges(self, gray_image):
        """
        Preprocess image for edge detection
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            preprocessed: Preprocessed image
        """
        # Apply bilateral filter with edge-specific parameters
        preprocessed = apply_bilateral_filter(
            gray_image,
            d=self.preprocessing_config['d'],
            sigma_color=self.preprocessing_config['sigma_color'],
            sigma_space=self.preprocessing_config['sigma_space']
        )
        return preprocessed
    
    def detect_sobel_edges(self, gray_image):
        """
        Apply Sobel edge detection
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            edge_mask: Binary edge mask
        """
        logger.info("Applying Sobel edge detection")
        
        # Preprocess image
        filtered = self.preprocess_for_edges(gray_image)
        
        # Calculate gradients
        grad_x, grad_y, magnitude = calculate_sobel_gradients(filtered)
        
        # Normalize magnitude
        magnitude_normalized = np.uint8(np.clip(magnitude * 255.0 / np.max(magnitude), 0, 255))
        
        # Apply Otsu's threshold
        _, edge_mask = cv2.threshold(magnitude_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up the edge mask
        kernel = np.ones(self.config['morphology_kernel'], np.uint8)
        edge_mask = apply_morphology_operation(edge_mask, 'close', kernel)
        
        logger.info("Edge detection completed")
        return edge_mask
    
    def detect_canny_edges(self, gray_image, low_threshold=50, high_threshold=150):
        """
        Apply Canny edge detection (alternative method)
        
        Args:
            gray_image: Grayscale image
            low_threshold: Lower threshold for edge linking
            high_threshold: Upper threshold for edge detection
            
        Returns:
            edge_mask: Binary edge mask
        """
        logger.info("Applying Canny edge detection")
        
        # Preprocess image
        filtered = self.preprocess_for_edges(gray_image)
        
        # Apply Canny edge detection
        edges = cv2.Canny(filtered, low_threshold, high_threshold)
        
        # Optional: dilate edges slightly to connect broken edges
        kernel = np.ones((2, 2), np.uint8)
        edge_mask = cv2.dilate(edges, kernel, iterations=1)
        
        return edge_mask
    
    def combine_edge_methods(self, gray_image):
        """
        Combine multiple edge detection methods for better results
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            combined_mask: Combined edge mask
        """
        logger.info("Combining multiple edge detection methods")
        
        # Get Sobel edges
        sobel_edges = self.detect_sobel_edges(gray_image)
        
        # Get Canny edges
        canny_edges = self.detect_canny_edges(gray_image)
        
        # Combine both methods
        combined_mask = cv2.bitwise_or(sobel_edges, canny_edges)
        
        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        combined_mask = apply_morphology_operation(combined_mask, 'close', kernel)
        
        return combined_mask
    
    def get_edge_density(self, edge_mask, roi=None):
        """
        Calculate edge density in a region
        
        Args:
            edge_mask: Binary edge mask
            roi: Region of interest (x, y, w, h) or None for full image
            
        Returns:
            float: Edge density (ratio of edge pixels to total pixels)
        """
        if roi is not None:
            x, y, w, h = roi
            mask_roi = edge_mask[y:y+h, x:x+w]
        else:
            mask_roi = edge_mask
        
        edge_pixels = np.sum(mask_roi > 0)
        total_pixels = mask_roi.shape[0] * mask_roi.shape[1]
        
        density = edge_pixels / total_pixels if total_pixels > 0 else 0
        return density