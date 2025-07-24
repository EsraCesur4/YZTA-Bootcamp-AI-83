import os
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import logging
from ultralytics import YOLO
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix PyTorch 2.6 compatibility issue with YOLO models
try:
    # Add safe globals for YOLO model loading
    torch.serialization.add_safe_globals([
        'ultralytics.nn.tasks.DetectionModel',
        'ultralytics.nn.modules.head.Detect',
        'ultralytics.nn.modules.conv.Conv',
        'ultralytics.nn.modules.block.C2f',
        'ultralytics.nn.modules.block.SPPF',
        'ultralytics.nn.modules.block.Bottleneck',
        'collections.OrderedDict',
        'torch.nn.modules.conv.Conv2d',
        'torch.nn.modules.batchnorm.BatchNorm2d',
        'torch.nn.modules.activation.SiLU',
        'torch.nn.modules.pooling.MaxPool2d',
        'torch.nn.modules.upsampling.Upsample',
        'torch.nn.modules.container.Sequential',
        'torch.nn.modules.container.ModuleList'
    ])
    logger.info("Added safe globals for YOLO model loading")
except Exception as e:
    logger.warning(f"Could not add safe globals: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model path
MODEL_PATH = r"C:\Users\USER\Downloads\YZTA-Bootcamp-AI-83\X-Ray_Workbench_AI\Dental_X_Ray\best.pt"  # Place your best.pt file in the same directory as this script

# Dental condition mapping (based on your model's class names)
# You may need to adjust these based on your actual model's class names
DENTAL_CONDITIONS = {
    0: 'Cavities',
    1: 'Fillings', 
    2: 'Impacted_Teeth',
    3: 'Implants'
}

# Global variable for model
yolo_model = None

def load_yolo_model():
    """Load the YOLO model with PyTorch 2.6 compatibility fixes"""
    global yolo_model
    
    try:
        logger.info("Loading YOLO model...")
        
        # Method 1: Try loading with safe globals (should work with the fix above)
        try:
            yolo_model = YOLO(MODEL_PATH)
            logger.info("YOLO model loaded successfully with safe globals!")
        except Exception as e1:
            logger.warning(f"Safe globals loading failed: {e1}")
            
            # Method 2: Use context manager approach
            try:
                logger.info("Trying context manager approach...")
                with torch.serialization.safe_globals(['ultralytics.nn.tasks.DetectionModel']):
                    yolo_model = YOLO(MODEL_PATH)
                logger.info("YOLO model loaded successfully with context manager!")
            except Exception as e2:
                logger.warning(f"Context manager loading failed: {e2}")
                
                # Method 3: Temporarily disable weights_only (less secure but works)
                try:
                    logger.info("Trying with weights_only=False (legacy mode)...")
                    # Monkey patch torch.load temporarily
                    original_load = torch.load
                    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
                    
                    yolo_model = YOLO(MODEL_PATH)
                    
                    # Restore original torch.load
                    torch.load = original_load
                    logger.info("YOLO model loaded successfully with legacy mode!")
                except Exception as e3:
                    logger.error(f"All loading methods failed: {e3}")
                    raise e3
        
        # Print model info
        logger.info(f"Model classes: {yolo_model.names}")
        logger.info(f"Model device: {yolo_model.device}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        return False

def preprocess_image(image):
    """Preprocess image for YOLO model"""
    try:
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
            
        # YOLO expects RGB format
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Image is already RGB
            return image_array
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # Convert RGBA to RGB
            return cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        else:
            # Convert grayscale to RGB
            return cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def yolo_to_classification(results):
    """Convert YOLO detection results to classification format"""
    try:
        # Initialize probabilities for all classes
        class_probabilities = [0.0] * len(DENTAL_CONDITIONS)
        
        if len(results.boxes) == 0:
            # No detections found - this shouldn't happen with your 4-class system
            # Return equal low probabilities
            return {
                'predicted_class': 'Cavities',  # Default
                'confidence': 25.0,
                'probabilities': [0.25, 0.25, 0.25, 0.25],
                'detection_count': 0,
                'detections': []
            }
        
        # Process all detections
        detections = []
        class_confidence_sums = [0.0] * len(DENTAL_CONDITIONS)
        class_counts = [0] * len(DENTAL_CONDITIONS)
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = DENTAL_CONDITIONS.get(class_id, f'Unknown_{class_id}')
            
            # Store detection info
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': box.xyxy[0].tolist()
            })
            
            # Accumulate confidence for each class
            if class_id < len(class_confidence_sums):
                class_confidence_sums[class_id] += confidence
                class_counts[class_id] += 1
        
        # Calculate average confidence for each detected class
        for i in range(len(class_confidence_sums)):
            if class_counts[i] > 0:
                class_probabilities[i] = class_confidence_sums[i] / class_counts[i]
        
        # Normalize probabilities to sum to 1
        total_prob = sum(class_probabilities)
        if total_prob > 0:
            class_probabilities = [p / total_prob for p in class_probabilities]
        else:
            # Fallback: equal probabilities
            class_probabilities = [0.25] * 4
        
        # Find the class with highest probability
        max_prob_index = np.argmax(class_probabilities)
        predicted_class = DENTAL_CONDITIONS[max_prob_index]
        confidence = class_probabilities[max_prob_index] * 100
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': class_probabilities,
            'detection_count': len(detections),
            'detections': detections
        }
        
    except Exception as e:
        logger.error(f"Error converting YOLO to classification: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_status = yolo_model is not None
        
        return jsonify({
            'status': 'healthy' if model_status else 'unhealthy',
            'model_loaded': model_status,
            'model_type': 'YOLO',
            'device': 'CPU',  # YOLO will auto-detect GPU if available
            'dental_conditions': DENTAL_CONDITIONS
        })
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if yolo_model is None:
            return jsonify({'error': 'YOLO model not loaded'}), 500
        
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and process the image
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Run YOLO prediction
        logger.info("Running YOLO prediction...")
        results = yolo_model.predict(processed_image, verbose=False)
        
        if not results or len(results) == 0:
            return jsonify({'error': 'No prediction results returned'}), 500
        
        # Convert YOLO results to classification format
        classification_result = yolo_to_classification(results[0])
        
        # Format response for the HTML interface
        response = {
            'predicted_class': classification_result['predicted_class'],
            'confidence': classification_result['confidence'],
            'probabilities': classification_result['probabilities'],
            
            # Additional YOLO-specific information
            'detection_details': {
                'total_detections': classification_result['detection_count'],
                'detections': classification_result['detections']
            }
        }
        
        logger.info(f"Prediction completed: {response['predicted_class']} with {response['confidence']:.1f}% confidence")
        logger.info(f"Total detections: {classification_result['detection_count']}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Alternative endpoint that returns full YOLO detection results"""
    try:
        if yolo_model is None:
            return jsonify({'error': 'YOLO model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
        
        # Run YOLO prediction
        results = yolo_model.predict(processed_image, verbose=False)
        
        if not results or len(results) == 0:
            return jsonify({'error': 'No prediction results returned'}), 500
        
        # Format full detection results
        detections = []
        for box in results[0].boxes:
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': DENTAL_CONDITIONS.get(int(box.cls[0]), 'Unknown'),
                'confidence': float(box.conf[0]),
                'bbox': {
                    'x1': float(box.xyxy[0][0]),
                    'y1': float(box.xyxy[0][1]),
                    'x2': float(box.xyxy[0][2]),
                    'y2': float(box.xyxy[0][3])
                }
            }
            detections.append(detection)
        
        response = {
            'total_detections': len(detections),
            'detections': detections,
            'image_size': {
                'width': image.width,
                'height': image.height
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if yolo_model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        return jsonify({
            'model_type': 'YOLO',
            'classes': yolo_model.names,
            'class_mapping': DENTAL_CONDITIONS,
            'input_size': 'Variable (YOLO auto-resizes)',
            'model_file': MODEL_PATH
        })
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting Dental YOLO Analysis API...")
    logger.info("Loading YOLO model...")
    
    if load_yolo_model():
        logger.info("Model loaded successfully!")
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to load YOLO model. Please check the model file.")
        print("\nModel Loading Failed!")
        print("Please verify:")
        print(f"1. YOLO model exists: {MODEL_PATH}")
        print("2. Ultralytics package is installed: pip install ultralytics")
        print("3. Model file is not corrupted")
        print("4. YOLO model was trained properly")