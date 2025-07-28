#!/usr/bin/env python3
"""
General Utility Functions
Common utilities used across the application
"""

import numpy as np
from flask import current_app
import logging

logger = logging.getLogger(__name__)


def allowed_file(filename):
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file
        
    Returns:
        bool: True if file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


def validate_image_file(request):
    """
    Validate image file in request
    
    Args:
        request: Flask request object
        
    Returns:
        str: Error message if validation fails, None if valid
    """
    if 'image' not in request.files:
        return 'No image file provided'
    
    file = request.files['image']
    
    if file.filename == '':
        return 'No file selected'
    
    if not allowed_file(file.filename):
        allowed_exts = ', '.join(current_app.config['ALLOWED_EXTENSIONS'])
        return f'Invalid file type. Allowed types: {allowed_exts}'
    
    return None


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object with Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_types(item) for item in obj)
    return obj


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: First box (x, y, w, h)
        box2: Second box (x, y, w, h)
        
    Returns:
        float: IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(boxes, scores=None, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes
    
    Args:
        boxes: List of boxes (x, y, w, h)
        scores: Optional list of confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        list: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes_array = np.array(boxes)
    
    # If no scores provided, use area as score
    if scores is None:
        scores = boxes_array[:, 2] * boxes_array[:, 3]  # w * h
    else:
        scores = np.array(scores)
    
    # Sort by scores
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        # Take the box with highest score
        current = indices[0]
        keep.append(current)
        
        # Calculate IoU with remaining boxes
        ious = []
        for idx in indices[1:]:
            iou = calculate_iou(boxes[current], boxes[idx])
            ious.append(iou)
        
        # Remove boxes with high IoU
        ious = np.array(ious)
        indices = indices[1:][ious <= iou_threshold]
    
    return keep


def format_time(seconds):
    """
    Format seconds into human-readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.0f}s"


def log_memory_usage():
    """
    Log current memory usage
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        logger.info(f"Memory usage: {memory_mb:.1f} MB")
    except ImportError:
        pass  # psutil not installed


def create_debug_image(images_dict, layout='grid'):
    """
    Create a debug image showing all pipeline stages
    
    Args:
        images_dict: Dictionary of images from pipeline
        layout: Layout type ('grid' or 'vertical')
        
    Returns:
        combined_image: Combined debug image
    """
    import cv2
    
    # Define which images to include and their labels
    stages = [
        ('original', 'Original'),
        ('grayscale', 'Grayscale'),
        ('enhanced', 'Enhanced'),
        ('binary', 'Binary'),
        ('mser_mask', 'MSER'),
        ('edge_mask', 'Edges'),
        ('intersection', 'Intersection'),
        ('filtered', 'Filtered'),
        ('morphology', 'Morphology'),
        ('result', 'Result')
    ]
    
    images = []
    labels = []
    
    for key, label in stages:
        if key in images_dict:
            img = images_dict[key]
            # Convert to 3-channel if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            images.append(img)
            labels.append(label)
    
    if not images:
        return None
    
    # Add labels to images
    labeled_images = []
    for img, label in zip(images, labels):
        labeled = img.copy()
        cv2.putText(labeled, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        labeled_images.append(labeled)
    
    if layout == 'grid':
        # Arrange in grid (2 columns)
        rows = []
        for i in range(0, len(labeled_images), 2):
            if i + 1 < len(labeled_images):
                row = np.hstack([labeled_images[i], labeled_images[i + 1]])
            else:
                # Pad last image if odd number
                blank = np.zeros_like(labeled_images[i])
                row = np.hstack([labeled_images[i], blank])
            rows.append(row)
        combined = np.vstack(rows)
    else:
        # Vertical layout
        combined = np.vstack(labeled_images)
    
    return combined