import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model paths
BINARY_MODEL_PATH = r"C:\Users\USER\Downloads\YZTA-Bootcamp-AI-83\X-Ray_Workbench_AI\Bone_X_Ray\fracture_classification_model.h5"
MULTICLASS_MODEL_PATH = r"C:\Users\USER\Downloads\YZTA-Bootcamp-AI-83\X-Ray_Workbench_AI\Bone_X_Ray\fracture_classification_CNN.h5"

# Fracture type mapping (0-9)
FRACTURE_TYPES = {
    0: 'Avulsion fracture',
    1: 'Comminuted fracture',
    2: 'Fracture Dislocation',
    3: 'Greenstick fracture',
    4: 'Hairline Fracture',
    5: 'Impacted fracture',
    6: 'Longitudinal fracture',
    7: 'Oblique fracture',
    8: 'Pathological fracture',
    9: 'Spiral Fracture'
}

# Global variables for models
binary_model = None
multiclass_model = None
binary_input_shape = None
multiclass_input_shape = (256, 256, 3)

def create_binary_model():
    """Create binary model architecture"""
    from tensorflow.keras import layers, models
    
    # Use 180x180 based on the error message
    IMG_HEIGHT, IMG_WIDTH = 180, 180
    
    model = models.Sequential([
        layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_multiclass_model():
    """Create multiclass model architecture"""
    from tensorflow.keras import layers, models
    
    input_layer = layers.Input(shape=(256, 256, 3))

    x = layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=2,
        padding="same"
    )(input_layer)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(rate=0.2)(x)

    x = layers.Conv2D(
        64,
        3,
        strides=2,
        padding="same"
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(rate=0.2)(x)

    x = layers.Conv2D(
        128,
        3,
        strides=2,
        padding="same"
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(rate=0.2)(x)

    x = layers.Conv2D(
        256,
        3,
        strides=2,
        padding="same"
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(rate=0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(units=100)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(rate=0.2)(x)

    output_layer = layers.Dense(units=10, activation="softmax")(x)

    model = models.Model(input_layer, output_layer)
    return model

def load_models():
    """Load both binary and multiclass models with compatibility handling"""
    global binary_model, multiclass_model, binary_input_shape
    
    try:
        logger.info("Loading binary fracture detection model...")
        
        # Method 1: Try loading directly with compile=False
        try:
            binary_model = keras.models.load_model(BINARY_MODEL_PATH, compile=False)
            binary_input_shape = binary_model.input_shape[1:]  # Remove batch dimension
            logger.info(f"Binary model loaded successfully (Method 1). Input shape: {binary_input_shape}")
        except Exception as e1:
            logger.warning(f"Method 1 failed: {str(e1)}")
            
            # Method 2: Recreate architecture and load weights
            try:
                logger.info("Trying to recreate binary model architecture and load weights...")
                binary_model = create_binary_model()
                binary_model.load_weights(BINARY_MODEL_PATH)
                binary_input_shape = binary_model.input_shape[1:]
                logger.info(f"Binary model loaded successfully (Method 2). Input shape: {binary_input_shape}")
            except Exception as e2:
                logger.error(f"Method 2 failed: {str(e2)}")
                
                # Method 3: Try with custom objects
                try:
                    logger.info("Trying with custom objects...")
                    import tensorflow.keras.utils as utils
                    binary_model = keras.models.load_model(
                        BINARY_MODEL_PATH, 
                        compile=False,
                        custom_objects={'InputLayer': layers.InputLayer}
                    )
                    binary_input_shape = binary_model.input_shape[1:]
                    logger.info(f"Binary model loaded successfully (Method 3). Input shape: {binary_input_shape}")
                except Exception as e3:
                    logger.error(f"All methods failed for binary model: {str(e3)}")
                    raise e3
        
        logger.info("Loading multiclass fracture classification model...")
        
        # Similar approach for multiclass model
        try:
            multiclass_model = keras.models.load_model(MULTICLASS_MODEL_PATH, compile=False)
            logger.info(f"Multiclass model loaded successfully (Method 1). Input shape: {multiclass_input_shape}")
        except Exception as e1:
            logger.warning(f"Multiclass Method 1 failed: {str(e1)}")
            
            try:
                logger.info("Trying to recreate multiclass model architecture and load weights...")
                multiclass_model = create_multiclass_model()
                multiclass_model.load_weights(MULTICLASS_MODEL_PATH)
                logger.info(f"Multiclass model loaded successfully (Method 2). Input shape: {multiclass_input_shape}")
            except Exception as e2:
                logger.error(f"Multiclass Method 2 failed: {str(e2)}")
                
                try:
                    logger.info("Trying multiclass with custom objects...")
                    multiclass_model = keras.models.load_model(
                        MULTICLASS_MODEL_PATH, 
                        compile=False,
                        custom_objects={'InputLayer': layers.InputLayer}
                    )
                    logger.info(f"Multiclass model loaded successfully (Method 3). Input shape: {multiclass_input_shape}")
                except Exception as e3:
                    logger.error(f"All methods failed for multiclass model: {str(e3)}")
                    raise e3
        
        # Print model summaries for verification
        logger.info("Binary model summary:")
        binary_model.summary()
        logger.info("Multiclass model summary:")
        multiclass_model.summary()
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def preprocess_image_binary(image, target_size):
    """Preprocess image for binary model"""
    try:
        # Resize image to match binary model input
        image = image.resize(target_size[:2])
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image for binary model: {str(e)}")
        raise

def preprocess_image_multiclass(image, target_size=(256, 256)):
    """Preprocess image for multiclass model"""
    try:
        # Resize image to 256x256 for multiclass model
        image = image.resize(target_size)
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image for multiclass model: {str(e)}")
        raise

def predict_fracture_binary(image):
    """Stage 1: Binary fracture detection"""
    try:
        # Preprocess image for binary model
        processed_image = preprocess_image_binary(image, binary_input_shape)
        
        # Make prediction
        prediction = binary_model.predict(processed_image, verbose=0)
        probability = float(prediction[0][0])
        
        # Binary classification: sigmoid output > 0.5 means "Fractured"
        is_fractured = probability > 0.5
        predicted_class = "Fractured" if is_fractured else "Not Fractured"
        confidence = probability * 100 if is_fractured else (1 - probability) * 100
        
        # Probabilities for both classes
        prob_not_fractured = 1 - probability
        prob_fractured = probability
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': [prob_not_fractured, prob_fractured],
            'raw_output': probability
        }
        
    except Exception as e:
        logger.error(f"Error in binary prediction: {str(e)}")
        raise

def predict_fracture_type(image):
    """Stage 2: Multiclass fracture type classification"""
    try:
        # Preprocess image for multiclass model
        processed_image = preprocess_image_multiclass(image)
        
        # Make prediction
        predictions = multiclass_model.predict(processed_image, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted class
        predicted_class_index = np.argmax(probabilities)
        predicted_class = FRACTURE_TYPES[predicted_class_index]
        confidence = float(probabilities[predicted_class_index]) * 100
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'predicted_index': int(predicted_class_index)
        }
        
    except Exception as e:
        logger.error(f"Error in multiclass prediction: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_status = binary_model is not None and multiclass_model is not None
        
        # Get device info
        device_info = "CPU"
        if tf.config.list_physical_devices('GPU'):
            device_info = "GPU"
        
        return jsonify({
            'status': 'healthy' if model_status else 'unhealthy',
            'model_loaded': model_status,
            'device': device_info,
            'binary_model_shape': binary_input_shape,
            'multiclass_model_shape': multiclass_input_shape,
            'fracture_types': FRACTURE_TYPES
        })
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Two-stage prediction endpoint"""
    try:
        # Check if models are loaded
        if binary_model is None or multiclass_model is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
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
        
        # Stage 1: Binary fracture detection
        logger.info("Running Stage 1: Binary fracture detection...")
        binary_result = predict_fracture_binary(image)
        
        response = {
            'stage1_prediction': binary_result['prediction'],
            'stage1_confidence': binary_result['confidence'],
            'stage1_probabilities': binary_result['probabilities']
        }
        
        # Stage 2: Fracture type classification (only if fracture detected)
        if binary_result['prediction'] == 'Fractured':
            logger.info("Fracture detected. Running Stage 2: Fracture type classification...")
            multiclass_result = predict_fracture_type(image)
            
            response.update({
                'stage2_prediction': multiclass_result['prediction'],
                'stage2_confidence': multiclass_result['confidence'],
                'stage2_probabilities': multiclass_result['probabilities'],
                'stage2_predicted_index': multiclass_result['predicted_index']
            })
        else:
            logger.info("No fracture detected. Skipping Stage 2.")
            response.update({
                'stage2_prediction': None,
                'stage2_confidence': None,
                'stage2_probabilities': None,
                'stage2_predicted_index': None
            })
        
        logger.info(f"Prediction completed: {response['stage1_prediction']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/fracture-types', methods=['GET'])
def get_fracture_types():
    """Get available fracture types"""
    return jsonify(FRACTURE_TYPES)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models on startup
    logger.info("Starting Bone Fracture Analysis API...")
    logger.info("Loading AI models...")
    
    if load_models():
        logger.info("Models loaded successfully!")
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to load models. Please check the model paths and files.")
        print("\nModel Loading Failed!")
        print("Please verify:")
        print(f"1. Binary model exists: {BINARY_MODEL_PATH}")
        print(f"2. Multiclass model exists: {MULTICLASS_MODEL_PATH}")
        print("3. Model files are not corrupted")
        print("4. TensorFlow/Keras versions are compatible")