#!/usr/bin/env python3
"""
MSER Detection Module
Handles MSER (Maximally Stable Extremal Regions) detection for text extraction
"""

import cv2
import numpy as np
import logging
from config import MSER_CONFIG, MSER_MASK_CONFIG
from image_utils import apply_bilateral_filter, apply_morphology_operation

logger = logging.getLogger(__name__)


class MSERDetector:
    """
    MSER detector for text region extraction
    """
    
    def __init__(self, config=None):
        """
        Initialize MSER detector with configuration
        
        Args:
            config: Dictionary of MSER parameters (uses default if None)
        """
        self.config = config or MSER_CONFIG
        self.mask_config = MSER_MASK_CONFIG
        self.mser = self._create_mser_detector()
        
    def _create_mser_detector(self):
        """
        Create MSER detector with proper parameters for different OpenCV versions
        
        Returns:
            cv2.MSER: MSER detector object
        """
        try:
            # Try newer OpenCV version syntax first
            mser = cv2.MSER_create(
                delta=self.config['delta'],
                min_area=self.config['min_area'],
                max_area=self.config['max_area'],
                max_variation=self.config['max_variation'],
                min_diversity=self.config['min_diversity'],
                max_evolution=self.config['max_evolution'],
                area_threshold=self.config['area_threshold'],
                min_margin=self.config['min_margin'],
                edge_blur_size=self.config['edge_blur_size']
            )
            logger.info("Created MSER detector with custom parameters")
        except TypeError:
            try:
                # Try with underscores (some versions)
                mser = cv2.MSER_create(
                    _delta=self.config['delta'],
                    _min_area=self.config['min_area'],
                    _max_area=self.config['max_area'],
                    _max_variation=self.config['max_variation'],
                    _min_diversity=self.config['min_diversity'],
                    _max_evolution=self.config['max_evolution'],
                    _area_threshold=self.config['area_threshold'],
                    _min_margin=self.config['min_margin'],
                    _edge_blur_size=self.config['edge_blur_size']
                )
                logger.info("Created MSER detector with underscore parameters")
            except TypeError:
                # Fallback to default MSER
                logger.warning("Using default MSER parameters due to OpenCV version compatibility")
                mser = cv2.MSER_create()
        
        return mser
    
    def preprocess_for_mser(self, gray_image):
        """
        Preprocess image specifically for MSER detection
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            preprocessed: Preprocessed image
        """
        # Apply bilateral filter with MSER-specific parameters
        preprocessed = apply_bilateral_filter(
            gray_image, 
            d=5, 
            sigma_color=50, 
            sigma_space=50
        )
        return preprocessed
    
    def detect_regions(self, gray_image):
        """
        Detect MSER regions in the image
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            tuple: (regions, bboxes) where regions are MSER regions and bboxes are bounding boxes
        """
        # Preprocess image
        preprocessed = self.preprocess_for_mser(gray_image)
        
        # Detect regions
        regions, bboxes = self.mser.detectRegions(preprocessed)
        logger.info(f"Detected {len(regions)} MSER regions")
        
        return regions, bboxes
    
    def filter_region(self, region, img_shape):
        """
        Check if a single MSER region passes basic filtering criteria
        
        Args:
            region: MSER region points
            img_shape: Image dimensions (height, width)
            
        Returns:
            tuple: (passes_filter, hull, bbox)
        """
        # Check minimum points
        if len(region) < self.mask_config['min_region_points']:
            return False, None, None
        
        # Get convex hull
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(hull)
        
        img_height, img_width = img_shape
        
        # Check if too close to border
        border_margin = self.mask_config['border_margin']
        if (x < border_margin or y < border_margin or 
            x + w > img_width - border_margin or 
            y + h > img_height - border_margin):
            # Apply stricter criteria for border regions
            if w * h < self.mask_config['min_border_area']:
                return False, None, None
        
        # Check dimensions
        min_dim = self.mask_config['min_dimension']
        if w < min_dim or h < min_dim:
            return False, None, None
        
        # Check aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        min_ar, max_ar = self.mask_config['aspect_ratio_range']
        if aspect_ratio < min_ar or aspect_ratio > max_ar:
            return False, None, None
        
        return True, hull, (x, y, w, h)
    
    def create_mser_mask(self, gray_image, regions):
        """
        Create binary mask from MSER regions
        
        Args:
            gray_image: Original grayscale image
            regions: List of MSER regions
            
        Returns:
            mser_mask: Binary mask of MSER regions
        """
        logger.info("Creating MSER mask")
        
        # Initialize mask
        mser_mask = np.zeros_like(gray_image)
        img_shape = gray_image.shape
        
        # Process each region
        valid_regions = 0
        for region in regions:
            passes_filter, hull, bbox = self.filter_region(region, img_shape)
            
            if passes_filter and hull is not None:
                cv2.fillPoly(mser_mask, [hull], 255)
                valid_regions += 1
        
        logger.info(f"Created mask with {valid_regions} valid regions")
        
        # Clean up the mask
        kernel = np.ones(self.mask_config['cleanup_kernel'], np.uint8)
        mser_mask = apply_morphology_operation(mser_mask, 'open', kernel)
        
        return mser_mask
    
    def extract_mser_regions(self, gray_image):
        """
        Complete MSER extraction pipeline
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            tuple: (regions, mser_mask)
        """
        # Detect regions
        regions, _ = self.detect_regions(gray_image)
        
        # Create mask
        mser_mask = self.create_mser_mask(gray_image, regions)
        
        return regions, mser_mask