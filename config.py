#!/usr/bin/env python3
"""
Configuration file for Text Extraction Pipeline
Contains all configurable parameters and thresholds
"""

# Flask Configuration
FLASK_CONFIG = {
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file size
    'UPLOAD_FOLDER': 'uploads',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
    'TEMPLATES_AUTO_RELOAD': True,
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'DEBUG': True
}

# Image Preprocessing Parameters
PREPROCESSING_CONFIG = {
    'target_size': (600, 600),
    'clip_limit': 2.0,
    'tile_grid_size': (8, 8),
    'bilateral_filter': {
        'default': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
        'edge_detection': {'d': 9, 'sigma_color': 50, 'sigma_space': 50},
        'mser': {'d': 5, 'sigma_color': 50, 'sigma_space': 50}
    },
    'gaussian_blur_kernel': (5, 5)
}

# MSER Detection Parameters
MSER_CONFIG = {
    'delta': 5,
    'min_area': 100,
    'max_area': 14400,
    'max_variation': 0.25,
    'min_diversity': 0.2,
    'max_evolution': 200,
    'area_threshold': 1.01,
    'min_margin': 0.003,
    'edge_blur_size': 5
}

# Text Region Filtering Parameters
FILTER_CONFIG = {
    # Size constraints
    'min_text_area': 200,
    'max_text_area': 12000,
    'min_pixels': 80,
    'min_height': 10,
    'max_height': 150,
    'min_width': 10,
    
    # Shape constraints
    'min_aspect_ratio': 0.25,
    'max_aspect_ratio': 10,
    'min_solidity': 0.5,
    'min_compactness': 1.8,
    'max_compactness': 18,
    'min_rectangularity': 0.55,
    
    # Texture constraints
    'min_density': 0.35,
    'max_density': 0.92,
    'min_edge_density': 0.15,
    'max_edge_density': 0.8,
    'max_stroke_variation': 0.8,
    
    # Other constraints
    'border_margin': 15,
    'max_holes': 2,
    'min_morphology_area': 180,
    'min_contour_area': 60,
    'min_fill_ratio': 0.25,
    'max_circularity': 0.9,
    'min_circularity': 0.1
}

# Morphological Operations Parameters
MORPHOLOGY_CONFIG = {
    'kernels': {
        'small': (2, 2),
        'horizontal': (5, 1),
        'vertical': (1, 3),
        'ellipse': (3, 3)
    },
    'min_component_area': 80
}

# Text Region Merging Parameters
MERGE_CONFIG = {
    'min_area_threshold': 400,
    'min_initial_area': 200,
    'min_merge_area': 250,
    'min_merge_width': 12,
    'min_merge_height': 10,
    'merge_aspect_ratio_range': (0.25, 10),
    'min_box_height': 8,
    'min_box_width': 8,
    'min_box_solidity': 0.45,
    'min_avg_solidity': 0.5,
    'border_threshold': 10,
    'border_min_area': 500,
    'border_min_height': 12,
    'overlap_threshold': 0.6,
    'horizontal_merge_factor': 1.2,
    'vertical_merge_factor': 0.4,
    'alignment_threshold': 0.6
}

# Edge Detection Parameters
EDGE_CONFIG = {
    'sobel_kernel_size': 3,
    'morphology_kernel': (2, 2)
}

# MSER Mask Creation Parameters
MSER_MASK_CONFIG = {
    'min_region_points': 50,
    'border_margin': 5,
    'min_border_area': 500,
    'min_dimension': 8,
    'aspect_ratio_range': (0.15, 15),
    'cleanup_kernel': (2, 2)
}