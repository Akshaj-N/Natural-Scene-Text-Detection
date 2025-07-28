#!/usr/bin/env python3
"""
Image Preprocessing Module
Handles all preprocessing operations for the text extraction pipeline
"""

import cv2
import numpy as np
import logging
from config import PREPROCESSING_CONFIG
from image_utils import (
    resize_image, 
    convert_to_grayscale, 
    apply_bilateral_filter, 
    apply_clahe, 
    apply_otsu_threshold
)

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles all image preprocessing operations for text extraction
    """
    
    def __init__(self, config=None):
        """
        Initialize the preprocessor with configuration
        
        Args:
            config: Dictionary of preprocessing parameters
        """
        self.config = config or PREPROCESSING_CONFIG
        logger.info("Initialized ImagePreprocessor")
    
    def preprocess_image(self, image):
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input BGR image
            
        Returns:
            tuple: (resized, grayscale, enhanced, binary) images
        """
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Resize image
        resized = resize_image(image, self.config['target_size'])
        
        # Step 2: Convert to grayscale
        gray = convert_to_grayscale(resized)
        
        # Step 3: Denoise
        denoised = apply_bilateral_filter(
            gray,
            **self.config['bilateral_filter']['default']
        )
        
        # Step 4: Enhance contrast
        enhanced = apply_clahe(
            denoised, 
            self.config['clip_limit'], 
            self.config['tile_grid_size']
        )
        
        # Step 5: Binarize
        binary = apply_otsu_threshold(
            enhanced, 
            self.config['gaussian_blur_kernel']
        )
        
        logger.info("Preprocessing pipeline completed")
        return resized, gray, enhanced, binary
    
    def preprocess_for_edge_detection(self, gray_image):
        """
        Preprocessing specifically for edge detection
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            filtered: Preprocessed image ready for edge detection
        """
        return apply_bilateral_filter(
            gray_image,
            **self.config['bilateral_filter']['edge_detection']
        )
    
    def preprocess_for_mser(self, gray_image):
        """
        Preprocessing specifically for MSER detection
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            denoised: Preprocessed image ready for MSER
        """
        return apply_bilateral_filter(
            gray_image,
            **self.config['bilateral_filter']['mser']
        )