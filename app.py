# ==============================================================================
# IMPORTS & INITIAL SETUP
# ==============================================================================
import os
import io
import logging
import traceback  # Added for detailed error logging
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
# --- Corrected Relative Paths for Deployment ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

CHEST_MODEL_PATH = os.path.join(MODELS_DIR, 'model.h5')
FRACTURE_BINARY_MODEL_PATH = os.path.join(MODELS_DIR, 'fracture_classification_model.h5')
FRACTURE_MULTICLASS_MODEL_PATH = os.path.join(MODELS_DIR, 'fracture_classification_CNN.h5')
DENTAL_YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'best.pt')


# --- Model Configurations & Class Names ---
# 1. Chest X-Ray Model
CHEST_CLASS_NAMES = ['Normal', 'Tuberculosis', 'Pneumonia', 'COVID-19']
CHEST_IMAGE_SIZE = 128

# 2. Bone Fracture Models
FRACTURE_TYPES = {
    0: 'Avulsion fracture', 1: 'Comminuted fracture', 2: 'Fracture Dislocation',
    3: 'Greenstick fracture', 4: 'Hairline Fracture', 5: 'Impacted fracture',
    6: 'Longitudinal fracture', 7: 'Oblique fracture', 8: 'Pathological fracture',
    9: 'Spiral Fracture'
}
FRACTURE_MULTICLASS_INPUT_SHAPE = (256, 256, 3)

# 3. Dental YOLO Model
DENTAL_CONDITIONS = {
    0: 'Cavities', 1: 'Fillings', 2: 'Impacted_Teeth', 3: 'Implants'
}

# --- Global Model Variables ---
chest_model = None
fracture_binary_model = None
fracture_multiclass_model = None
fracture_binary_input_shape = None
dental_yolo_model = None

# ==============================================================================
# SECTION 1: CHEST X-RAY CLASSIFICATION (model.h5)
# ==============================================================================

def load_chest_model():
    """Load the TensorFlow/Keras model for Chest X-Ray classification."""
    global chest_model
    try:
        logger.info(f"Loading Chest X-Ray model from: {CHEST_MODEL_PATH}")
        chest_model = tf.keras.models.load_model(CHEST_MODEL_PATH, compile=False)
        logger.info("✓ Chest X-Ray model loaded successfully")
        chest_model.summary(print_fn=logger.info)
        return True
    except Exception as e:
        logger.error(f"Error loading Chest X-Ray model: {e}")
        traceback.print_exc()  # Log the full error traceback
        return False

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

# ==============================================================================
# SECTION 2: BONE FRACTURE CLASSIFICATION (2-Stage TF/Keras)
# ==============================================================================

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
        try:
            fracture_binary_model = keras.models.load_model(FRACTURE_BINARY_MODEL_PATH, compile=False)
        except Exception:
            logger.warning("Direct loading of binary fracture model failed. Recreating architecture...")
            fracture_binary_model = create_fracture_binary_model()
            fracture_binary_model.load_weights(FRACTURE_BINARY_MODEL_PATH)
        fracture_binary_input_shape = fracture_binary_model.input_shape[1:]
        logger.info(f"✓ Binary fracture model loaded. Input shape: {fracture_binary_input_shape}")
        fracture_binary_model.summary(print_fn=logger.info)

        # Load Multiclass Model
        logger.info(f"Loading multiclass fracture model from {FRACTURE_MULTICLASS_MODEL_PATH}")
        try:
            fracture_multiclass_model = keras.models.load_model(FRACTURE_MULTICLASS_MODEL_PATH, compile=False)
        except Exception:
            logger.warning("Direct loading of multiclass fracture model failed. Recreating architecture...")
            fracture_multiclass_model = create_fracture_multiclass_model()
            fracture_multiclass_model.load_weights(FRACTURE_MULTICLASS_MODEL_PATH)
        logger.info(f"✓ Multiclass fracture model loaded. Input shape: {FRACTURE_MULTICLASS_INPUT_SHAPE}")
        fracture_multiclass_model.summary(print_fn=logger.info)
        return True
    except Exception as e:
        logger.error(f"Error loading fracture models: {e}")
        traceback.print_exc()  # Log the full error traceback
        return False

def preprocess_image_fracture(image, target_size):
    """Preprocess image for a fracture model."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size[:2])
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fracture_binary(image):
    """Stage 1: Binary fracture detection."""
    processed_image = preprocess_image_fracture(image, fracture_binary_input_shape)
    prediction = fracture_binary_model.predict(processed_image, verbose=0)
    probability = float(prediction[0][0])
    is_fractured = probability > 0.5
    predicted_class = "Fractured" if is_fractured else "Not Fractured"
    confidence = probability * 100 if is_fractured else (1 - probability) * 100
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': [1 - probability, probability]
    }

def predict_fracture_type(image):
    """Stage 2: Multiclass fracture type classification."""
    processed_image = preprocess_image_fracture(image, FRACTURE_MULTICLASS_INPUT_SHAPE)
    predictions = fracture_multiclass_model.predict(processed_image, verbose=0)
    probabilities = predictions[0]
    predicted_class_index = np.argmax(probabilities)
    predicted_class = FRACTURE_TYPES[predicted_class_index]
    confidence = float(probabilities[predicted_class_index]) * 100
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities.tolist()
    }

# ==============================================================================
# SECTION 3: DENTAL ANOMALY DETECTION (YOLOv8)
# ==============================================================================

def load_dental_yolo_model():
    """Load the YOLO model for Dental X-Ray analysis."""
    global dental_yolo_model
    try:
        logger.info(f"Loading Dental YOLO model from: {DENTAL_YOLO_MODEL_PATH}")
        dental_yolo_model = YOLO(DENTAL_YOLO_MODEL_PATH)
        logger.info("✓ Dental YOLO model loaded successfully")
        logger.info(f"YOLO model classes: {dental_yolo_model.names}")
        return True
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        traceback.print_exc()  # Log the full error traceback
        return False

def preprocess_yolo_image(image):
    """Preprocess PIL image for YOLO model (convert to NumPy array in RGB)."""
    image_array = np.array(image.convert('RGB'))
    return image_array

def yolo_to_classification_summary(results):
    """Convert YOLO detection results into a single classification-like output."""
    if len(results.boxes) == 0:
        return {'predicted_class': 'No Detections', 'confidence': 100.0, 'probabilities': [], 'detections': []}

    class_confidence_sums = {i: 0.0 for i in range(len(DENTAL_CONDITIONS))}
    class_counts = {i: 0 for i in range(len(DENTAL_CONDITIONS))}
    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = DENTAL_CONDITIONS.get(class_id, f'Unknown_{class_id}')
        
        detections.append({
            'class_name': class_name, 'confidence': confidence, 'bbox': box.xyxy[0].tolist()
        })
        
        if class_id in class_confidence_sums:
            class_confidence_sums[class_id] += confidence
            class_counts[class_id] += 1
    
    avg_confidences = [class_confidence_sums[i] / class_counts[i] if class_counts[i] > 0 else 0.0 for i in range(len(DENTAL_CONDITIONS))]
    total_avg_conf = sum(avg_confidences)
    probabilities = [c / total_avg_conf if total_avg_conf > 0 else 0 for c in avg_confidences]

    if not detections:
        predicted_class = "No Detections"
        confidence = 100.0
    else:
        max_prob_index = np.argmax(probabilities)
        predicted_class = DENTAL_CONDITIONS[max_prob_index]
        confidence = probabilities[max_prob_index] * 100
        
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'detections': detections
    }

# ==============================================================================
# FLASK HTML RENDERING ROUTES
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
    
# ==============================================================================
# FLASK API ENDPOINTS
# ==============================================================================

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

@app.route('/fracture/predict', methods=['POST'])
def predict_fracture_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        binary_result = predict_fracture_binary(image)
        response = {'stage1_prediction': binary_result}
        
        if binary_result['prediction'] == 'Fractured':
            multiclass_result = predict_fracture_type(image)
            response['stage2_prediction'] = multiclass_result
        else:
            response['stage2_prediction'] = None
            
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /fracture/predict endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to process image'}), 400

@app.route('/dental/predict', methods=['POST'])
def predict_dental_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_yolo_image(image)
        
        results = dental_yolo_model.predict(processed_image, verbose=False)
        if not results:
            return jsonify({'error': 'Model returned no results'}), 500
        
        summary = yolo_to_classification_summary(results[0])
        
        return jsonify({
            'predicted_class': summary['predicted_class'],
            'confidence': summary['confidence'],
            'probabilities': summary['probabilities'],
            'detection_details': {
                'total_detections': len(summary['detections']),
                'detections': summary['detections']
            }
        })
    except Exception as e:
        logger.error(f"Error in /dental/predict endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to process image'}), 400

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    logger.info("Starting X-Ray Workbench AI Server...")
    
    # Load all models at startup
    logger.info("\n--- Loading AI Models ---")
    load_chest_model()
    load_fracture_models()
    load_dental_yolo_model()
    logger.info("--- Model Loading Complete ---\n")
    
    logger.info("Starting Flask development server...")
    # This app.run() is for local development only.
    # A production server like Gunicorn will be used for deployment.
    app.run(host='0.0.0.0', port=5000, debug=True)