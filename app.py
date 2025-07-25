# ==============================================================================
# IMPORTS & INITIAL SETUP
# ==============================================================================
import os
import io
import sys
import logging
import traceback
import requests
import numpy as np
from PIL import Image

# Flask
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
CORS(app)

logger.info("Flask app created successfully")

# Import heavy libraries with error handling
try:
    import tensorflow as tf
    logger.info(f"TensorFlow imported: {tf.__version__}")
except ImportError as e:
    logger.error(f"TensorFlow import failed: {e}")
    tf = None

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    logger.info("Keras imported successfully")
except ImportError as e:
    logger.error(f"Keras import failed: {e}")
    keras = None
    layers = None
    models = None

try:
    import torch
    logger.info(f"PyTorch imported: {torch.__version__}")
except ImportError as e:
    logger.error(f"PyTorch import failed: {e}")
    torch = None

try:
    from ultralytics import YOLO
    logger.info("YOLO imported successfully")
except ImportError as e:
    logger.error(f"YOLO import failed: {e}")
    YOLO = None

# ==============================================================================
# MODEL PATHS & CONFIGURATIONS
# ==============================================================================

def get_models_directory():
    """Get the models directory, checking multiple possible locations."""
    # Check for persistent disk mount point
    persistent_disk_path = '/opt/render/project/src/persistent_models'
    
    # Check for environment variable override
    if 'MODELS_PATH' in os.environ:
        models_dir = os.environ['MODELS_PATH']
        logger.info(f"Using MODELS_PATH environment variable: {models_dir}")
        return models_dir
    
    # Check if persistent disk is mounted
    if os.path.exists(persistent_disk_path):
        models_dir = os.path.join(persistent_disk_path, 'Models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        logger.info(f"Using persistent disk models directory: {models_dir}")
        return models_dir
    
    # Fall back to local directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'Models')
    logger.info(f"Using local models directory: {models_dir}")
    return models_dir

# Get the models directory
MODELS_DIR = get_models_directory()

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    logger.info(f"Creating models directory: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)

# Model file paths
CHEST_MODEL_PATH = os.path.join(MODELS_DIR, 'model.h5')
FRACTURE_BINARY_MODEL_PATH = os.path.join(MODELS_DIR, 'fracture_classification_model.h5')
FRACTURE_MULTICLASS_MODEL_PATH = os.path.join(MODELS_DIR, 'fracture_classification_CNN.h5')
DENTAL_YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'best.pt')

# Model configurations
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
# GOOGLE DRIVE MODEL DOWNLOAD FUNCTIONS WITH YOUR EXACT LINKS
# ==============================================================================

def download_model_from_url(url, filename, max_retries=3):
    """Download a model file from a direct URL."""
    local_path = os.path.join(MODELS_DIR, filename)
    
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        logger.info(f"✓ Model {filename} already exists ({file_size / (1024*1024):.2f} MB)")
        return True
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {filename} (attempt {attempt + 1}/{max_retries})")
            logger.info(f"URL: {url}")
            
            # Use session to handle redirects properly
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            response = session.get(url, stream=True, timeout=600, allow_redirects=True)
            response.raise_for_status()
            
            # Check if we got an HTML page instead of file content
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                logger.warning(f"Received HTML content for {filename}, trying alternative method...")
                # Try to extract download link from HTML if it's a Google Drive warning page
                if 'drive.google.com' in url:
                    # Extract file ID and try different approach
                    file_id = url.split('id=')[1].split('&')[0]
                    alt_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                    logger.info(f"Trying alternative URL: {alt_url}")
                    response = session.get(alt_url, stream=True, timeout=600, allow_redirects=True)
                    response.raise_for_status()
            
            # Save the file
            logger.info(f"Saving {filename}...")
            total_downloaded = 0
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_downloaded += len(chunk)
                        
                        # Log progress every 50MB
                        if total_downloaded % (50 * 1024 * 1024) == 0:
                            logger.info(f"Downloaded {total_downloaded / (1024*1024):.1f} MB of {filename}")
            
            file_size = os.path.getsize(local_path)
            if file_size < 1024:  # File too small, probably an error page
                logger.error(f"Downloaded file {filename} is too small ({file_size} bytes), probably an error")
                os.remove(local_path)
                continue
            
            logger.info(f"✓ Successfully downloaded {filename} ({file_size / (1024*1024):.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {filename}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)  # Clean up partial download
            
            if attempt == max_retries - 1:
                logger.error(f"All download attempts failed for {filename}")
    
    return False

def download_all_models_from_google_drive():
    """Download all model files from your Google Drive links."""
    
    # YOUR EXACT GOOGLE DRIVE DOWNLOAD LINKS
    model_downloads = {
        'model.h5': 'https://drive.google.com/uc?id=1-sOrJ-FiFgZ2zdl0uItI3cpiI7MKtyGg&export=download',
        'fracture_classification_model.h5': 'https://drive.google.com/uc?id=1Gk_EdL8YZigmYsn7ogS__qjw-noZgLAl&export=download',
        'fracture_classification_CNN.h5': 'https://drive.google.com/uc?id=1tX2FEvVYTFSFKqJKSOmaFt-lXw4I6B4u&export=download',
        'best.pt': 'https://drive.google.com/uc?id=1e9b2vT5y9OtKqYyNl2gb_Fl5BRx__3TA&export=download'
    }
    
    logger.info("="*60)
    logger.info("DOWNLOADING MODELS FROM GOOGLE DRIVE")
    logger.info("="*60)
    
    success_count = 0
    total_models = len(model_downloads)
    
    for filename, url in model_downloads.items():
        logger.info(f"\nDownloading {filename}...")
        if download_model_from_url(url, filename):
            success_count += 1
        else:
            logger.error(f"❌ Failed to download {filename}")
    
    logger.info("="*60)
    logger.info(f"DOWNLOAD SUMMARY: {success_count}/{total_models} models downloaded successfully")
    logger.info("="*60)
    
    return success_count > 0

def ensure_models_available():
    """Ensure all model files are available, downloading if necessary."""
    model_files = [
        'model.h5',
        'fracture_classification_model.h5',
        'fracture_classification_CNN.h5',
        'best.pt'
    ]
    
    logger.info("Checking model availability...")
    missing_files = []
    
    for filename in model_files:
        local_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(local_path):
            missing_files.append(filename)
            logger.warning(f"❌ Model {filename} not found locally")
        else:
            file_size = os.path.getsize(local_path)
            logger.info(f"✓ Found {filename} ({file_size / (1024*1024):.2f} MB)")
    
    # If any files are missing, download them from Google Drive
    if missing_files:
        logger.info(f"Need to download {len(missing_files)} missing model files...")
        return download_all_models_from_google_drive()
    
    return True

# ==============================================================================
# MODEL LOADING FUNCTIONS
# ==============================================================================

def load_chest_model():
    """Load the TensorFlow/Keras model for Chest X-Ray classification."""
    global chest_model
    if tf is None or keras is None:
        logger.error("TensorFlow/Keras not available - cannot load chest model")
        return False
    
    try:
        if not os.path.exists(CHEST_MODEL_PATH):
            logger.error(f"Chest model file not found: {CHEST_MODEL_PATH}")
            return False
        
        logger.info(f"Loading chest model from: {CHEST_MODEL_PATH}")
        chest_model = tf.keras.models.load_model(CHEST_MODEL_PATH, compile=False)
        logger.info("✓ Chest X-Ray model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading chest model: {e}")
        traceback.print_exc()
        return False

def create_fracture_binary_model():
    """Recreate binary model architecture."""
    if models is None or layers is None:
        return None
    
    model = models.Sequential([
        layers.InputLayer(input_shape=(180, 180, 3)),
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

def create_fracture_multiclass_model():
    """Recreate multiclass model architecture."""
    if layers is None or models is None:
        return None
    
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
    
    if keras is None:
        logger.error("Keras not available - cannot load fracture models")
        return False
    
    try:
        # Check if files exist
        if not os.path.exists(FRACTURE_BINARY_MODEL_PATH):
            logger.error(f"Binary fracture model not found: {FRACTURE_BINARY_MODEL_PATH}")
            return False
        
        if not os.path.exists(FRACTURE_MULTICLASS_MODEL_PATH):
            logger.error(f"Multiclass fracture model not found: {FRACTURE_MULTICLASS_MODEL_PATH}")
            return False
        
        # Load Binary Model
        logger.info("Loading binary fracture model...")
        try:
            fracture_binary_model = keras.models.load_model(FRACTURE_BINARY_MODEL_PATH, compile=False)
        except Exception:
            logger.info("Direct loading failed, using custom architecture...")
            fracture_binary_model = create_fracture_binary_model()
            if fracture_binary_model:
                fracture_binary_model.load_weights(FRACTURE_BINARY_MODEL_PATH)
        
        if fracture_binary_model:
            fracture_binary_input_shape = fracture_binary_model.input_shape[1:]
            logger.info(f"✓ Binary fracture model loaded")

        # Load Multiclass Model
        logger.info("Loading multiclass fracture model...")
        try:
            fracture_multiclass_model = keras.models.load_model(FRACTURE_MULTICLASS_MODEL_PATH, compile=False)
        except Exception:
            logger.info("Direct loading failed, using custom architecture...")
            fracture_multiclass_model = create_fracture_multiclass_model()
            if fracture_multiclass_model:
                fracture_multiclass_model.load_weights(FRACTURE_MULTICLASS_MODEL_PATH)
        
        if fracture_multiclass_model:
            logger.info(f"✓ Multiclass fracture model loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading fracture models: {e}")
        traceback.print_exc()
        return False

def load_dental_yolo_model():
    """Load the YOLO model for Dental X-Ray analysis."""
    global dental_yolo_model
    
    if YOLO is None:
        logger.error("YOLO not available - cannot load dental model")
        return False
    
    try:
        if not os.path.exists(DENTAL_YOLO_MODEL_PATH):
            logger.error(f"YOLO model not found: {DENTAL_YOLO_MODEL_PATH}")
            return False
        
        logger.info(f"Loading dental YOLO model...")
        dental_yolo_model = YOLO(DENTAL_YOLO_MODEL_PATH)
        logger.info("✓ Dental YOLO model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        traceback.print_exc()
        return False

# ==============================================================================
# PREDICTION FUNCTIONS
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

# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route("/health", methods=['GET'])
def health_check():
    """Health check with detailed status."""
    model_status = {
        "chest_xray": chest_model is not None,
        "fracture_binary": fracture_binary_model is not None,
        "fracture_multiclass": fracture_multiclass_model is not None,
        "dental_yolo": dental_yolo_model is not None,
    }
    
    debug_info = {
        "models_directory": MODELS_DIR,
        "models_dir_exists": os.path.exists(MODELS_DIR),
        "python_version": sys.version,
        "tensorflow_available": tf is not None,
        "torch_available": torch is not None,
        "yolo_available": YOLO is not None,
    }
    
    if os.path.exists(MODELS_DIR):
        try:
            debug_info["models_dir_contents"] = os.listdir(MODELS_DIR)
        except Exception as e:
            debug_info["models_dir_error"] = str(e)
    
    # Check each model file
    model_files = {
        "chest_model": CHEST_MODEL_PATH,
        "fracture_binary": FRACTURE_BINARY_MODEL_PATH,
        "fracture_multiclass": FRACTURE_MULTICLASS_MODEL_PATH,
        "dental_yolo": DENTAL_YOLO_MODEL_PATH
    }
    
    debug_info["model_files"] = {}
    for name, path in model_files.items():
        debug_info["model_files"][name] = {
            "path": path,
            "exists": os.path.exists(path),
            "size_mb": round(os.path.getsize(path) / (1024*1024), 2) if os.path.exists(path) else 0
        }
    
    models_loaded_count = sum(model_status.values())
    
    return jsonify({
        "status": "healthy" if models_loaded_count > 0 else "unhealthy",
        "models_loaded": model_status,
        "models_loaded_count": f"{models_loaded_count}/4",
        "debug_info": debug_info
    })

@app.route('/')
def route_index():
    """Home page route."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index.html: {e}")
        return f"Template error: {e}", 500

@app.route('/bone-fracture')
def route_page2():
    """Bone fracture page route."""
    try:
        return render_template('page2.html')
    except Exception as e:
        logger.error(f"Error rendering page2.html: {e}")
        return f"Template error: {e}", 500

@app.route('/dental')
def route_page3():
    """Dental page route."""
    try:
        return render_template('page3.html')
    except Exception as e:
        logger.error(f"Error rendering page3.html: {e}")
        return f"Template error: {e}", 500

@app.route('/chest/predict', methods=['POST'])
def predict_chest_endpoint():
    """Chest X-ray prediction endpoint."""
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

# ==============================================================================
# INITIALIZATION
# ==============================================================================

def initialize_models():
    """Initialize all models at startup."""
    logger.info("="*80)
    logger.info("INITIALIZING RADIOLOGY AI WORKBENCH")
    logger.info("="*80)
    
    # Ensure models are available (download if needed)
    logger.info("Ensuring model availability...")
    models_available = ensure_models_available()
    
    if not models_available:
        logger.error("Failed to ensure model availability")
        return 0
    
    # Load models
    logger.info("\nLoading AI models...")
    chest_loaded = load_chest_model()
    fracture_loaded = load_fracture_models()
    dental_loaded = load_dental_yolo_model()
    
    # Log results
    total_loaded = sum([chest_loaded, fracture_loaded, dental_loaded])
    logger.info("="*50)
    logger.info("MODEL LOADING SUMMARY")
    logger.info("="*50)
    logger.info(f"Chest X-Ray model: {'✓ LOADED' if chest_loaded else '❌ FAILED'}")
    logger.info(f"Fracture models: {'✓ LOADED' if fracture_loaded else '❌ FAILED'}")
    logger.info(f"Dental YOLO model: {'✓ LOADED' if dental_loaded else '❌ FAILED'}")
    logger.info(f"Total models loaded: {total_loaded}/3")
    
    if total_loaded == 0:
        logger.error("❌ NO MODELS LOADED - Server will start but predictions will fail")
    elif total_loaded < 3:
        logger.warning(f"⚠️  Only {total_loaded}/3 models loaded - Some features may not work")
    else:
        logger.info("✅ ALL MODELS LOADED SUCCESSFULLY - Server ready!")
    
    logger.info("="*80)
    
    return total_loaded

# Initialize models when module is imported (for Gunicorn)
logger.info("Flask app module loaded - initializing models...")
models_loaded = initialize_models()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    logger.info("Running development server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    logger.info(f"Module imported - Flask app ready for Gunicorn")

logger.info(f"Flask app object ready: {type(app)}")