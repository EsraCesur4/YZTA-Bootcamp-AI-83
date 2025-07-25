# Your existing app.py code with Fly.io compatibility

import os
import io
import logging
import traceback
import numpy as np
from PIL import Image
import cv2

# Flask
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# TensorFlow (for Chest & Bone Fracture models)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# PyTorch & Ultralytics (for Dental YOLO model)
import torch
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# ==============================================================================
# MODEL PATHS & CONFIGURATIONS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

CHEST_MODEL_PATH = os.path.join(MODELS_DIR, 'model.h5')
FRACTURE_BINARY_MODEL_PATH = os.path.join(MODELS_DIR, 'fracture_classification_model.h5')
FRACTURE_MULTICLASS_MODEL_PATH = os.path.join(MODELS_DIR, 'fracture_classification_CNN.h5')
DENTAL_YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'best.pt')

# Model Configurations & Class Names
CHEST_CLASS_NAMES = ['Normal', 'Tuberculosis', 'Pneumonia', 'COVID-19']
CHEST_IMAGE_SIZE = 128

FRACTURE_TYPES = {
    0: 'Avulsion fracture', 1: 'Comminuted fracture', 2: 'Fracture Dislocation',
    3: 'Greenstick fracture', 4: 'Hairline Fracture', 5: 'Impacted fracture',
    6: 'Longitudinal fracture', 7: 'Oblique fracture', 8: 'Pathological fracture',
    9: 'Spiral Fracture'
}
FRACTURE_MULTICLASS_INPUT_SHAPE = (256, 256, 3)

DENTAL_CONDITIONS = {
    0: 'Cavities', 1: 'Fillings', 2: 'Impacted_Teeth', 3: 'Implants'
}

# Global Model Variables
chest_model = None
fracture_binary_model = None
fracture_multiclass_model = None
fracture_binary_input_shape = None
dental_yolo_model = None

# ==============================================================================
# MODEL LOADING FUNCTIONS (same as your original code)
# ==============================================================================

def load_chest_model():
    """Load the TensorFlow/Keras model for Chest X-Ray classification."""
    global chest_model
    try:
        logger.info(f"Loading Chest X-Ray model from: {CHEST_MODEL_PATH}")
        if not os.path.exists(CHEST_MODEL_PATH):
            logger.error(f"Chest model not found at {CHEST_MODEL_PATH}")
            return False
        chest_model = tf.keras.models.load_model(CHEST_MODEL_PATH, compile=False)
        logger.info("✓ Chest X-Ray model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading Chest X-Ray model: {e}")
        traceback.print_exc()
        return False

def create_fracture_binary_model():
    """Recreate binary model architecture if direct loading fails."""
    model = models.Sequential([
        layers.InputLayer(input_shape=(180, 180, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_fracture_multiclass_model():
    """Recreate multiclass model architecture if direct loading fails."""
    input_layer = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding="same")(input_layer)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=100)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(rate=0.2)(x)
    output_layer = layers.Dense(units=10, activation="softmax")(x)
    return models.Model(input_layer, output_layer)

def load_fracture_models():
    """Load both binary and multiclass models for bone fracture."""
    global fracture_binary_model, fracture_multiclass_model, fracture_binary_input_shape
    try:
        # Load Binary Model
        logger.info(f"Loading binary fracture model from {FRACTURE_BINARY_MODEL_PATH}")
        if not os.path.exists(FRACTURE_BINARY_MODEL_PATH):
            logger.error(f"Binary fracture model not found at {FRACTURE_BINARY_MODEL_PATH}")
            return False
        try:
            fracture_binary_model = keras.models.load_model(FRACTURE_BINARY_MODEL_PATH, compile=False)
        except Exception:
            logger.warning("Direct loading of binary fracture model failed. Recreating architecture...")
            fracture_binary_model = create_fracture_binary_model()
            fracture_binary_model.load_weights(FRACTURE_BINARY_MODEL_PATH)
        fracture_binary_input_shape = fracture_binary_model.input_shape[1:]
        logger.info(f"✓ Binary fracture model loaded. Input shape: {fracture_binary_input_shape}")

        # Load Multiclass Model
        logger.info(f"Loading multiclass fracture model from {FRACTURE_MULTICLASS_MODEL_PATH}")
        if not os.path.exists(FRACTURE_MULTICLASS_MODEL_PATH):
            logger.error(f"Multiclass fracture model not found at {FRACTURE_MULTICLASS_MODEL_PATH}")
            return False
        try:
            fracture_multiclass_model = keras.models.load_model(FRACTURE_MULTICLASS_MODEL_PATH, compile=False)
        except Exception:
            logger.warning("Direct loading of multiclass fracture model failed. Recreating architecture...")
            fracture_multiclass_model = create_fracture_multiclass_model()
            fracture_multiclass_model.load_weights(FRACTURE_MULTICLASS_MODEL_PATH)
        logger.info(f"✓ Multiclass fracture model loaded. Input shape: {FRACTURE_MULTICLASS_INPUT_SHAPE}")
        return True
    except Exception as e:
        logger.error(f"Error loading fracture models: {e}")
        traceback.print_exc()
        return False

def load_dental_yolo_model():
    """Load the YOLO model for Dental X-Ray analysis."""
    global dental_yolo_model
    try:
        logger.info(f"Loading Dental YOLO model from: {DENTAL_YOLO_MODEL_PATH}")
        if not os.path.exists(DENTAL_YOLO_MODEL_PATH):
            logger.error(f"YOLO model not found at {DENTAL_YOLO_MODEL_PATH}")
            return False
        dental_yolo_model = YOLO(DENTAL_YOLO_MODEL_PATH)
        logger.info("✓ Dental YOLO model loaded successfully")
        logger.info(f"YOLO model classes: {dental_yolo_model.names}")
        return True
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        traceback.print_exc()
        return False

# ==============================================================================
# PREDICTION FUNCTIONS (your existing functions)
# ==============================================================================

def preprocess_chest_image(image):
    """Preprocess image for the Chest X-Ray model."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((CHEST_IMAGE_SIZE, CHEST_IMAGE_SIZE))
    img_array = np.array(image)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_chest_image(image):
    """Predict using the 4-class Chest X-Ray model."""
    if chest_model is None:
        return None, "Chest X-Ray model not loaded"
    try:
        processed_image = preprocess_chest_image(image)
        predictions = chest_model.predict(processed_image, verbose=0)
        probabilities = predictions[0]
        predicted_index = np.argmax(probabilities)
        predicted_class = CHEST_CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100
        
        return {
            'success': True,
            'predicted_class': str(predicted_class),
            'confidence': float(confidence),
            'probabilities': [float(p) for p in probabilities.tolist()],
            'class_names': CHEST_CLASS_NAMES
        }, None
    except Exception as e:
        logger.error(f"Chest X-Ray prediction error: {e}")
        traceback.print_exc()
        return None, f"Prediction error: {str(e)}"

# Include all your other prediction functions here...

# ==============================================================================
# FLASK ROUTES (your existing routes)
# ==============================================================================

@app.route('/')
def route_index():
    return render_template('index.html')

@app.route('/bone-fracture')
def route_page2():
    return render_template('page2.html')

@app.route('/dental')
def route_page3():
    return render_template('page3.html')

@app.route("/health", methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "chest_xray": chest_model is not None,
            "fracture_binary": fracture_binary_model is not None,
            "fracture_multiclass": fracture_multiclass_model is not None,
            "dental_yolo": dental_yolo_model is not None,
        }
    })

@app.route('/chest/predict', methods=['POST'])
def predict_chest_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read()))
        result, error = predict_chest_image(image)
        if error:
            return jsonify({'error': error}), 500
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /chest/predict endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to process image'}), 400

# Include all your other endpoints here...

# ==============================================================================
# MAIN EXECUTION BLOCK - FLY.IO COMPATIBLE
# ==============================================================================

def initialize_models():
    """Initialize models at startup"""
    logger.info("Starting X-Ray Workbench AI Server...")
    logger.info("--- Loading AI Models ---")
    
    chest_loaded = load_chest_model()
    fracture_loaded = load_fracture_models() 
    dental_loaded = load_dental_yolo_model()
    
    logger.info("--- Model Loading Complete ---")
    total_loaded = sum([chest_loaded, fracture_loaded, dental_loaded])
    logger.info(f"Total models loaded: {total_loaded}/3")
    
    return total_loaded

# Initialize models when module loads
models_loaded_count = initialize_models()

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # For production (Gunicorn)
    logger.info("App ready for production server")