#!/usr/bin/env python3
"""
Image Processing Utilities
Common image processing functions used throughout the pipeline
"""

import cv2
import numpy as np
import base64
import io
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def resize_image(image, target_size):
    """
    Resize image to target dimensions
    
    Args:
        image: Input image
        target_size: Target dimensions (width, height)
        
    Returns:
        resized: Resized image
    """
    resized = cv2.resize(image, target_size)
    logger.debug(f"Resized image to {target_size}")
    return resized


def convert_to_grayscale(image):
    """
    Convert image to grayscale using standard formula
    Grayscale = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image: Input color image (BGR format)
        
    Returns:
        gray: Grayscale image
    """
    if len(image.shape) == 3:
        # OpenCV uses BGR, so we need to adjust
        b, g, r = cv2.split(image)
        gray = np.uint8(0.299 * r + 0.587 * g + 0.114 * b)
    else:
        gray = image
    return gray


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter to reduce noise while preserving edges
    
    Args:
        image: Input image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        filtered: Filtered image
    """
    filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        enhanced: Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced


def apply_otsu_threshold(image, blur_kernel_size=(5, 5)):
    """
    Apply Otsu's thresholding with optional Gaussian blur
    
    Args:
        image: Grayscale image
        blur_kernel_size: Kernel size for Gaussian blur
        
    Returns:
        binary: Binary image
    """
    if blur_kernel_size:
        blurred = cv2.GaussianBlur(image, blur_kernel_size, 0)
    else:
        blurred = image
    
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def calculate_sobel_gradients(image):
    """
    Calculate Sobel gradients in x and y directions
    
    Args:
        image: Grayscale image
        
    Returns:
        tuple: (grad_x, grad_y, magnitude)
    """
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return grad_x, grad_y, magnitude


def convert_image_to_base64(image_array, is_color=None):
    """
    Convert numpy array image to base64 string
    
    Args:
        image_array: Numpy array representing the image
        is_color: Boolean indicating if image is color (None for auto-detect)
        
    Returns:
        str: Base64 encoded image string
    """
    # Auto-detect if color or grayscale
    if is_color is None:
        is_color = len(image_array.shape) == 3
    
    # Convert to RGB
    if is_color:
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_rgb)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64


def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2, label="Text"):
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        boxes: List of bounding boxes (x, y, w, h)
        color: Box color in BGR format
        thickness: Line thickness
        label: Label to display
        
    Returns:
        result: Image with drawn boxes
    """
    result = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(result, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result


def get_connected_components(binary_mask):
    """
    Get connected components from binary mask
    
    Args:
        binary_mask: Binary image
        
    Returns:
        tuple: (num_labels, labels, stats, centroids)
    """
    return cv2.connectedComponentsWithStats(binary_mask, connectivity=8)


def calculate_contour_properties(contour):
    """
    Calculate various properties of a contour
    
    Args:
        contour: OpenCV contour
        
    Returns:
        dict: Dictionary of contour properties
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    x, y, w, h = cv2.boundingRect(contour)
    
    properties = {
        'area': area,
        'perimeter': perimeter,
        'hull_area': hull_area,
        'bbox': (x, y, w, h),
        'aspect_ratio': w / h if h > 0 else 0,
        'solidity': area / hull_area if hull_area > 0 else 0,
        'extent': area / (w * h) if w * h > 0 else 0
    }
    
    if perimeter > 0:
        properties['compactness'] = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else 0
        properties['circularity'] = 4 * np.pi * area / (perimeter * perimeter)
    else:
        properties['compactness'] = 0
        properties['circularity'] = 0
    
    return properties


def apply_morphology_operation(image, operation, kernel, iterations=1):
    """
    Apply morphological operation to image
    
    Args:
        image: Binary image
        operation: Operation type ('erode', 'dilate', 'open', 'close')
        kernel: Structuring element
        iterations: Number of iterations
        
    Returns:
        result: Processed image
    """
    operations = {
        'erode': cv2.MORPH_ERODE,
        'dilate': cv2.MORPH_DILATE,
        'open': cv2.MORPH_OPEN,
        'close': cv2.MORPH_CLOSE
    }
    
    if operation in operations:
        result = cv2.morphologyEx(image, operations[operation], kernel, iterations=iterations)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return result