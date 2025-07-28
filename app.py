#!/usr/bin/env python3
"""
Flask Web Application for Text Extraction Pipeline
Provides REST API endpoints for processing images and extracting text regions
"""

from flask import Flask, request, jsonify, render_template
import os
import io
import logging
from PIL import Image
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Import pipeline and utilities
from text_extraction import TextExtractionPipeline
from config import FLASK_CONFIG
from utils import convert_numpy_types, validate_image_file
from image_utils import convert_image_to_base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Apply Flask configuration
for key, value in FLASK_CONFIG.items():
    app.config[key] = value

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Initialize the text extraction pipeline
pipeline = TextExtractionPipeline()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Process uploaded image through the text extraction pipeline"""
    try:
        # Validate request
        validation_error = validate_image_file(request)
        if validation_error:
            return jsonify({'error': validation_error}), 400
        
        file = request.files['image']
        
        # Read and decode image
        logger.info(f"Processing image: {secure_filename(file.filename)}")
        image_stream = io.BytesIO(file.read())
        image = Image.open(image_stream)
        
        # Convert to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logger.info(f"Image shape: {image_cv.shape}")
        
        # Process image through pipeline
        results = pipeline.process_image(image_cv)
        
        # Convert results for JSON response
        response_data = {}
        
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                # Convert images to base64
                response_data[key] = convert_image_to_base64(value)
            else:
                # Convert other types (handles numpy types)
                response_data[key] = convert_numpy_types(value)
        
        # Add processing statistics
        response_data['statistics'] = {
            'total_regions_detected': len(results.get('bounding_boxes', [])),
            'mser_regions_found': results.get('mser_count', 0),
            'valid_regions_after_filtering': len(results.get('valid_regions', [])),
            'processing_status': 'success'
        }
        
        logger.info(f"Processing completed. Found {len(results.get('bounding_boxes', []))} text regions")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Error processing image: {str(e)}',
            'processing_status': 'failed'
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Text Extraction Pipeline',
        'version': '2.0.0',
        'components': {
            'preprocessing': 'ready',
            'mser_detection': 'ready',
            'edge_detection': 'ready',
            'region_filtering': 'ready',
            'morphology': 'ready',
            'region_merging': 'ready'
        }
    })


@app.route('/config')
def get_config():
    """Get current pipeline configuration"""
    try:
        config = pipeline.get_configuration()
        return jsonify({
            'status': 'success',
            'configuration': config
        })
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/info')
def get_info():
    """Get information about the API"""
    return jsonify({
        'service': 'Text Extraction Pipeline API',
        'version': '2.0.0',
        'description': 'Extract text regions from images using MSER and edge detection',
        'endpoints': {
            '/': 'Web interface',
            '/process': 'Process an image (POST)',
            '/health': 'Health check',
            '/config': 'Get current configuration',
            '/info': 'API information'
        },
        'supported_formats': list(app.config['ALLOWED_EXTENSIONS']),
        'max_file_size': f"{app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB"
    })


@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle file too large error"""
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({
        'error': f'File too large. Maximum size is {max_size_mb}MB.',
        'max_size_bytes': app.config['MAX_CONTENT_LENGTH']
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist. Check /info for available endpoints.'
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle method not allowed errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method used is not allowed for this endpoint.'
    }), 405


@app.errorhandler(500)
def internal_server_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}", exc_info=True)
    return jsonify({
        'error': 'Internal server error occurred',
        'message': 'An unexpected error occurred while processing your request.'
    }), 500


def initialize_app():
    """Initialize the application"""
    logger.info("Initializing Text Extraction Pipeline server...")
    logger.info(f"Configuration: {app.config}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Allowed extensions: {app.config['ALLOWED_EXTENSIONS']}")
    logger.info(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    

if __name__ == '__main__':
    # Initialize app
    initialize_app()
    
    # Run the Flask app
    logger.info("Starting Text Extraction Pipeline server...")
    app.run(
        debug=app.config['DEBUG'],
        host=app.config['HOST'],
        port=app.config['PORT']
    )