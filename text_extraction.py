#!/usr/bin/env python3
"""
Text Extraction Pipeline
Main pipeline that orchestrates all text extraction components
"""

import cv2
import numpy as np
import logging
from preprocessing import ImagePreprocessor
from mser_detector import MSERDetector
from edge_detector import EdgeDetector
from region_filter import RegionFilter
from morphology_operations import MorphologyProcessor
from region_merger import RegionMerger
from image_utils import draw_bounding_boxes

logger = logging.getLogger(__name__)


class TextExtractionPipeline:
    """
    Main text extraction pipeline that coordinates all components
    """
    
    def __init__(self):
        """
        Initialize all pipeline components
        """
        self.preprocessor = ImagePreprocessor()
        self.mser_detector = MSERDetector()
        self.edge_detector = EdgeDetector()
        self.region_filter = RegionFilter()
        self.morphology_processor = MorphologyProcessor()
        self.region_merger = RegionMerger()
        
        logger.info("Initialized TextExtractionPipeline with all components")
    
    def intersect_masks(self, mser_mask, edge_mask):
        """
        Intersect MSER and edge masks
        
        Args:
            mser_mask: Binary MSER mask
            edge_mask: Binary edge mask
            
        Returns:
            intersection: Intersection of both masks
        """
        logger.info("Creating intersection of MSER and edge masks")
        return cv2.bitwise_and(mser_mask, edge_mask)
    
    def process_image(self, image):
        """
        Main pipeline execution
        
        Args:
            image: Input BGR image
            
        Returns:
            dict: Dictionary containing all intermediate and final results
        """
        logger.info("Starting text extraction pipeline")
        
        try:
            # Step 1: Pre-processing
            logger.info("Step 1: Preprocessing")
            resized, gray, enhanced, binary = self.preprocessor.preprocess_image(image)
            
            # Step 2: Text region extraction
            logger.info("Step 2: MSER region extraction")
            mser_regions, mser_mask = self.mser_detector.extract_mser_regions(enhanced)
            
            logger.info("Step 3: Edge detection")
            edge_mask = self.edge_detector.detect_sobel_edges(enhanced)
            
            logger.info("Step 4: Mask intersection")
            intersection_mask = self.intersect_masks(mser_mask, edge_mask)
            
            # Step 3: Non-text region removal
            logger.info("Step 5: Region filtering")
            filtered_mask, valid_regions = self.region_filter.filter_regions(intersection_mask)
            
            # Step 4: Morphological operations
            logger.info("Step 6: Morphological operations")
            morphology_mask = self.morphology_processor.apply_morphology(filtered_mask)
            
            # Step 5: Region merging
            logger.info("Step 7: Region merging")
            final_boxes = self.region_merger.merge_text_regions(
                morphology_mask, 
                img_shape=resized.shape[:2]
            )
            
            # Step 6: Draw results
            logger.info("Step 8: Drawing results")
            result_image = draw_bounding_boxes(resized, final_boxes)
            
            logger.info(f"Pipeline completed successfully. Found {len(final_boxes)} text regions")
            
            return {
                'original': resized,
                'grayscale': gray,
                'enhanced': enhanced,
                'binary': binary,
                'mser_mask': mser_mask,
                'edge_mask': edge_mask,
                'intersection': intersection_mask,
                'filtered': filtered_mask,
                'morphology': morphology_mask,
                'result': result_image,
                'bounding_boxes': final_boxes,
                'valid_regions': valid_regions,
                'mser_count': len(mser_regions)
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}", exc_info=True)
            raise
    
    def get_configuration(self):
        """
        Get current configuration of all components
        
        Returns:
            dict: Configuration dictionary
        """
        return {
            'preprocessing': self.preprocessor.config,
            'mser': self.mser_detector.config,
            'edge': self.edge_detector.config,
            'filter': self.region_filter.config,
            'morphology': self.morphology_processor.config,
            'merge': self.region_merger.config
        }
    
    def process_with_custom_params(self, image, **kwargs):
        """
        Process image with custom parameters
        
        Args:
            image: Input BGR image
            **kwargs: Custom parameters for different components
            
        Returns:
            dict: Processing results
        """
        logger.info("Processing with custom parameters")
        # This method can be extended to allow custom parameter overrides
        # For now, just use the standard process
        return self.process_image(image)