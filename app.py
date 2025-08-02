# ==============================================================================
# IMPORTS & INITIAL SETUP
# ==============================================================================
import os
import io
import logging
import traceback
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import logging

# Flask
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# PyTorch & Ultralytics
import torch
from ultralytics import YOLO

from datetime import timedelta
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-default-secret-key-change-this')
CORS(app, origins=["*"])  # Allow all origins for Hugging Face Spaces

app.permanent_session_lifetime = timedelta(hours=24)

# ==============================================================================
# MODEL PATHS & CONFIGURATIONS FOR HUGGING FACE SPACES
# ==============================================================================
# Get the current working directory
BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

# Model paths
CHEST_MODEL_PATH = os.path.join(MODELS_DIR, 'model.h5')
FRACTURE_BINARY_MODEL_PATH = os.path.join(MODELS_DIR, 'fracture_classification_model.h5')
FRACTURE_MULTICLASS_MODEL_PATH = os.path.join(MODELS_DIR, 'fracture_classification_CNN.h5')
DENTAL_YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'best.pt')
OCULAR_MODEL_PATH = os.path.join(MODELS_DIR, 'efficientnetb3-Eye Disease-96.19.h5')  # Your EfficientNetB3 model

# Log model paths for debugging
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Models directory: {MODELS_DIR}")
logger.info(f"Model files exist:")
logger.info(f"  Chest model: {os.path.exists(CHEST_MODEL_PATH)}")
logger.info(f"  Fracture binary: {os.path.exists(FRACTURE_BINARY_MODEL_PATH)}")
logger.info(f"  Fracture multiclass: {os.path.exists(FRACTURE_MULTICLASS_MODEL_PATH)}")
logger.info(f"  Dental YOLO: {os.path.exists(DENTAL_YOLO_MODEL_PATH)}")

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

# 4. Ocular Disease Model
OCULAR_CLASS_NAMES = ['ARMD', 'cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
OCULAR_IMAGE_SIZE = 224 

# --- Global Model Variables ---
chest_model = None
fracture_binary_model = None
fracture_multiclass_model = None
fracture_binary_input_shape = None
dental_yolo_model = None
ocular_model = None

# Model loading status
model_loading_status = {
    'chest_xray': False,
    'fracture_binary': False,
    'fracture_multiclass': False,
    'dental_yolo': False,
    'ocular': False
}

# ==============================================================================
# SECTION 1: CHEST X-RAY CLASSIFICATION (model.h5)
# ==============================================================================

def load_chest_model():
    """Load the TensorFlow/Keras model for Chest X-Ray classification."""
    global chest_model, model_loading_status
    try:
        if not os.path.exists(CHEST_MODEL_PATH):
            logger.error(f"Chest model file not found: {CHEST_MODEL_PATH}")
            return False
            
        logger.info(f"Loading Chest X-Ray model from: {CHEST_MODEL_PATH}")
        
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        chest_model = tf.keras.models.load_model(CHEST_MODEL_PATH, compile=False)
        model_loading_status['chest_xray'] = True
        logger.info("✓ Chest X-Ray model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading Chest X-Ray model: {e}")
        traceback.print_exc()
        model_loading_status['chest_xray'] = False
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
    global fracture_binary_model, fracture_multiclass_model, fracture_binary_input_shape, model_loading_status
    
    binary_loaded = False
    multiclass_loaded = False
    
    try:
        # Load Binary Model
        if os.path.exists(FRACTURE_BINARY_MODEL_PATH):
            logger.info(f"Loading binary fracture model from {FRACTURE_BINARY_MODEL_PATH}")
            try:
                fracture_binary_model = keras.models.load_model(FRACTURE_BINARY_MODEL_PATH, compile=False)
                binary_loaded = True
            except Exception:
                logger.warning("Direct loading of binary fracture model failed. Recreating architecture...")
                fracture_binary_model = create_fracture_binary_model()
                fracture_binary_model.load_weights(FRACTURE_BINARY_MODEL_PATH)
                binary_loaded = True
            fracture_binary_input_shape = fracture_binary_model.input_shape[1:]
            logger.info(f"✓ Binary fracture model loaded. Input shape: {fracture_binary_input_shape}")
        else:
            logger.error(f"Binary fracture model file not found: {FRACTURE_BINARY_MODEL_PATH}")

        # Load Multiclass Model
        if os.path.exists(FRACTURE_MULTICLASS_MODEL_PATH):
            logger.info(f"Loading multiclass fracture model from {FRACTURE_MULTICLASS_MODEL_PATH}")
            try:
                fracture_multiclass_model = keras.models.load_model(FRACTURE_MULTICLASS_MODEL_PATH, compile=False)
                multiclass_loaded = True
            except Exception:
                logger.warning("Direct loading of multiclass fracture model failed. Recreating architecture...")
                fracture_multiclass_model = create_fracture_multiclass_model()
                fracture_multiclass_model.load_weights(FRACTURE_MULTICLASS_MODEL_PATH)
                multiclass_loaded = True
            logger.info(f"✓ Multiclass fracture model loaded. Input shape: {FRACTURE_MULTICLASS_INPUT_SHAPE}")
        else:
            logger.error(f"Multiclass fracture model file not found: {FRACTURE_MULTICLASS_MODEL_PATH}")
            
        model_loading_status['fracture_binary'] = binary_loaded
        model_loading_status['fracture_multiclass'] = multiclass_loaded
        
        return binary_loaded and multiclass_loaded
        
    except Exception as e:
        logger.error(f"Error loading fracture models: {e}")
        traceback.print_exc()
        model_loading_status['fracture_binary'] = False
        model_loading_status['fracture_multiclass'] = False
        return False

def preprocess_image_fracture_binary(image):
    """Preprocess for binary fracture model (180x180, normalized)"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Binary model uses 180x180
    image = image.resize((180, 180))
    img_array = np.array(image, dtype=np.float32)
    
    # Binary model was trained with /255 normalization
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def preprocess_image_fracture_multiclass(image):
    """Preprocess for multiclass fracture model (256x256, normalized)"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Multiclass model uses 256x256
    image = image.resize((256, 256))
    img_array = np.array(image, dtype=np.float32)
    
    # Multiclass model was trained with /255 normalization
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_fracture_binary(image):
    """Stage 1: Binary fracture detection."""
    processed_image = preprocess_image_fracture_binary(image) 
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
    processed_image = preprocess_image_fracture_multiclass(image) 
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
    global dental_yolo_model, model_loading_status
    try:
        if not os.path.exists(DENTAL_YOLO_MODEL_PATH):
            logger.error(f"Dental YOLO model file not found: {DENTAL_YOLO_MODEL_PATH}")
            model_loading_status['dental_yolo'] = False
            return False
            
        logger.info(f"Loading Dental YOLO model from: {DENTAL_YOLO_MODEL_PATH}")
        dental_yolo_model = YOLO(DENTAL_YOLO_MODEL_PATH)
        model_loading_status['dental_yolo'] = True
        logger.info("✓ Dental YOLO model loaded successfully")
        logger.info(f"YOLO model classes: {dental_yolo_model.names}")
        return True
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        traceback.print_exc()
        model_loading_status['dental_yolo'] = False
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
# SECTION 4: OCULAR DISEASE CLASSIFICATION (.h5)
# ==============================================================================

def load_ocular_model():
    """Load the TensorFlow/Keras model for Ocular Disease classification."""
    global ocular_model, model_loading_status
    try:
        if not os.path.exists(OCULAR_MODEL_PATH):
            logger.error(f"Ocular model file not found: {OCULAR_MODEL_PATH}")
            return False
            
        logger.info(f"Loading Ocular Disease model from: {OCULAR_MODEL_PATH}")
        
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        ocular_model = tf.keras.models.load_model(OCULAR_MODEL_PATH, compile=False)
        model_loading_status['ocular'] = True
        logger.info("✓ Ocular Disease model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading Ocular Disease model: {e}")
        traceback.print_exc()
        model_loading_status['ocular'] = False
        return False

def preprocess_ocular_image(image):
    """Preprocess image to match training - NO NORMALIZATION!"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    logger.info(f"Raw pixel range: {img_array.min()} to {img_array.max()}")
    return img_array

def predict_ocular_image(image):
    """Predict using the 5-class Ocular Disease model."""
    if ocular_model is None:
        return None, "Ocular Disease model not loaded"
    try:
        processed_image = preprocess_ocular_image(image)
        predictions = ocular_model.predict(processed_image, verbose=0)
        probabilities = predictions[0]
        predicted_index = np.argmax(probabilities)
        predicted_class = OCULAR_CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100
        
        return {
            'success': True,
            'predicted_class': str(predicted_class),
            'confidence': float(confidence),
            'probabilities': [float(p) for p in probabilities.tolist()],
            'class_names': OCULAR_CLASS_NAMES
        }, None
    except Exception as e:
        logger.error(f"Ocular Disease prediction error: {e}")
        traceback.print_exc()
        return None, f"Prediction error: {str(e)}"


# ==============================================================================
# BACKEND INTEGRATION FOR LOGIN AND GEMINI
# ==============================================================================
import os
import re
from datetime import datetime
from flask import session, redirect, url_for
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import io

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            logger.info(f"Unauthorized access attempt to {request.endpoint}")
            # Redirect to login with the current URL as 'next' parameter
            return redirect(f'/login?next={request.url}')
        return f(*args, **kwargs)
    return decorated_function

# Doctor Database (in production, use a real database)
DOCTORS_DB = {
    'dr.ahmet': {
        'password': 'password123',
        'name': 'Dr. Ahmet Demir',
        'id': '100238',
        'department': 'Göğüs Hastalıkları',
        'title': 'Uzman Doktor'
    },
    'dr.ayse': {
        'password': 'password456',
        'name': 'Dr. Ayşe Kaya',
        'id': '100239',
        'department': 'Radyoloji',
        'title': 'Başhekim Yardımcısı'
    }
}

FACILITY_INFO = {
    'name': 'Premier Health Center',
    'id': '83472',
    'address': 'London, England',
    'phone': '+90 212 xxx xxxx'
}

# Patient data extraction functions
def extract_patient_info_from_filename(filename):
    """
    Extract patient information from filename format:
    25514963207_Mustafa_Yılmaz_25_M.png
    Returns: dict with patient info or None if format doesn't match
    """
    try:
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Pattern: ID_FirstName_LastName_Age_Gender
        pattern = r'^(\d+)_([^_]+)_([^_]+)_(\d+)_([MFmf])$'
        match = re.match(pattern, name_without_ext)
        
        if match:
            patient_id, first_name, last_name, age, gender = match.groups()
            
            return {
                'patient_id': patient_id,
                'patient_name': f"{first_name} {last_name}",
                'first_name': first_name,
                'last_name': last_name,
                'age': int(age),
                'gender': gender.upper(),
                'upload_datetime': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'upload_date': datetime.now().strftime("%Y-%m-%d"),
                'upload_time': datetime.now().strftime("%H:%M")
            }
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error extracting patient info: {e}")
        return None

def get_patient_info_or_unknown(filename):
    """Get patient info or return unknown values"""
    patient_info = extract_patient_info_from_filename(filename)
    
    if patient_info:
        return patient_info
    else:
        return {
            'patient_id': 'Unknown',
            'patient_name': 'Unknown (not logged in)',
            'first_name': 'Unknown',
            'last_name': '',
            'age': 'Unknown',
            'gender': 'Unknown',
            'upload_datetime': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'upload_date': datetime.now().strftime("%Y-%m-%d"),
            'upload_time': datetime.now().strftime("%H:%M")
        }

def get_current_doctor():
    """Get current logged in doctor or return unknown"""
    if session.get('logged_in') and session.get('doctor'):
        return session['doctor']
    else:
        return {
            'name': 'Unknown (not logged in)',
            'id': 'Unknown',
            'department': 'Unknown',
            'title': 'Unknown'
        }

# Gemini AI functions
def analyze_image_with_gemini(image_data, analysis_results=None):
    """
    Analyze medical image with Gemini to determine exam type and findings
    """
    try:
        # Convert PIL image to bytes for Gemini
        if hasattr(image_data, 'save'):
            img_byte_arr = io.BytesIO()
            image_data.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
        else:
            img_byte_arr = image_data
        
        # Prepare prompt for medical image analysis
        prompt = f"""
        You are a medical imaging expert radiologist. Analyze this medical X-ray image and provide a professional assessment.

        Please determine:
        1. **Exam Type**: What type of X-ray examination is this? (e.g., "Chest PA", "Chest Lateral", "Bone X-ray - Femur", "Dental Panoramic")
        2. **Anatomical Region**: What body part/region is being examined?
        3. **Image Quality**: Assess the technical quality (Excellent/Good/Fair/Poor)
        4. **Key Structures**: What anatomical structures are clearly visible?
        5. **Technical Notes**: Any technical observations about positioning, exposure, etc.

        Provide your analysis in a professional, medical format. Be specific about the examination type and anatomical details.

        Keep the response concise but medically accurate, as this will be included in a formal radiology report.
        """
        
        # Send to Gemini
        response = gemini_model.generate_content([prompt, img_byte_arr])
        
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini image analysis error: {e}")
        return "Image analysis not available. Technical analysis by radiologist required."

def generate_medical_report_with_gemini(patient_info, doctor_info, ai_results, image_analysis, reason_of_visit=""):
    """
    Generate comprehensive medical report using Gemini AI
    """
    try:
        # Prepare AI findings summary
        ai_findings = ""
        if ai_results:
            if 'predicted_class' in ai_results:
                ai_findings += f"AI Classification: {ai_results['predicted_class']} "
                if 'confidence' in ai_results:
                    ai_findings += f"(Confidence: {ai_results.get('confidence', 0):.1f}%)\n"
            
            if 'detection_details' in ai_results and ai_results['detection_details']['total_detections'] > 0:
                ai_findings += f"Objects Detected: {ai_results['detection_details']['total_detections']} findings\n"
        
        # Determine examination type
        exam_type = "Medical X-ray Examination"
        current_path = request.path if hasattr(request, 'path') else ""
        if 'chest' in current_path or ai_results.get('predicted_class') in ['Normal', 'COVID-19', 'Pneumonia', 'Tuberculosis']:
            exam_type = "Chest X-ray PA/Lateral"
        elif 'fracture' in current_path or 'bone' in current_path:
            exam_type = "Bone X-ray Examination"
        elif 'dental' in current_path:
            exam_type = "Dental X-ray Examination"

        prompt = f"""
        You are a senior radiologist writing a professional medical report. Generate a comprehensive, clinically appropriate radiology report based on the following information:

        **Patient Information:**
        - Name: {patient_info['patient_name']}
        - ID: {patient_info['patient_id']}
        - Age: {patient_info['age']} years old
        - Gender: {patient_info['gender']}
        - Examination Date: {patient_info['upload_date']}
        - Examination Time: {patient_info['upload_time']}

        **Clinical Information:**
        - Referring Physician: {doctor_info['name']} ({doctor_info['title']})
        - Department: {doctor_info['department']}
        - Physician ID: {doctor_info['id']}
        - Reason for Examination: {reason_of_visit if reason_of_visit else 'Clinical evaluation as requested'}
        - Facility: {FACILITY_INFO['name']}

        **Examination Type:** {exam_type}

        **AI Analysis Results:**
        {ai_findings}

        **Image Quality Assessment:**
        {image_analysis if image_analysis else 'Standard quality medical imaging study'}

        Please generate a professional radiology report with these sections:

        **CLINICAL INFORMATION:**
        Include patient demographics, clinical history, and examination indication

        **TECHNIQUE:**
        Describe the imaging technique and technical parameters

        **FINDINGS:**
        Provide detailed radiological findings. Incorporate the AI analysis results appropriately, describing what is observed in medical terminology. If AI detected specific conditions, describe the radiological appearance that supports or correlates with these findings.

        **IMPRESSION:**
        Provide a clear, concise summary of the key findings and their clinical significance

        **RECOMMENDATIONS:**
        Suggest appropriate follow-up, additional studies, or clinical correlation as needed

        Write in professional medical language appropriate for a formal radiology report. Ensure the report is clinically useful and follows standard radiology reporting conventions.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini report generation error: {e}")
        return f"""MEDICAL IMAGING REPORT

PATIENT INFORMATION:
Patient Name: {patient_info['patient_name']}
Patient ID: {patient_info['patient_id']}
Age: {patient_info['age']} | Gender: {patient_info['gender']}

EXAMINATION: {exam_type}
Date: {patient_info['upload_date']} | Time: {patient_info['upload_time']}
Referring Physician: {doctor_info['name']}
Department: {doctor_info['department']}

CLINICAL INFORMATION:
Reason for examination: {reason_of_visit if reason_of_visit else 'Clinical evaluation'}

TECHNIQUE:
Standard digital radiography

FINDINGS:
{ai_findings if ai_findings else 'Image analysis completed. Radiologist review recommended.'}

IMPRESSION:
Clinical correlation recommended.

RECOMMENDATIONS:
Follow-up as clinically indicated.

Report generated with AI assistance. Manual review by radiologist recommended.
Reported by: {doctor_info['name']}
Date: {patient_info['upload_datetime']}"""

def ask_gemini_medical_question(question, context=""):
    """
    AI Assistant for medical questions
    """
    try:
        prompt = f"""
        You are a medical AI assistant specializing in radiology and medical imaging. You provide educational information to healthcare professionals.

        Context Information: {context}
        
        Question: {question}

        Please provide a helpful, accurate medical response that:
        1. Uses appropriate medical terminology
        2. Provides educational value for healthcare professionals
        3. Includes relevant clinical considerations
        4. Suggests when additional consultation might be needed

        Remember to:
        - Be specific and evidence-based in your responses
        - Include differential diagnoses when appropriate
        - Mention relevant follow-up or additional studies
        - Always emphasize clinical correlation

        Important: This is for educational purposes to assist healthcare professionals. All final medical decisions should involve direct patient evaluation and clinical judgment.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini assistant error: {e}")
        return "I apologize, but I'm unable to process your question at the moment. Please consult with a colleague or refer to medical literature for guidance."

# ==============================================================================
# FLASK HTML RENDERING ROUTES
# ==============================================================================

@app.route('/')
def route_home():
    """Landing page - shows login if not logged in, otherwise redirects to main app"""
    if session.get('logged_in'):
        # User is already logged in, send them to the main app
        return redirect('/main')
    else:
        # User is not logged in, show login page
        return render_template('login.html')

@app.route('/main')
def route_index():
    """Main application page"""
    # Pass doctor info to template if logged in
    doctor = session.get('doctor', {})
    logged_in = session.get('logged_in', False)
    
    if logged_in:
        logger.info(f"Main app accessed by: {doctor.get('name', 'Unknown')}")
    
    return render_template('index.html', doctor=doctor, logged_in=logged_in)

@app.route('/bone-fracture')
def route_page2():
    """Bone fracture page"""
    return render_template('page2.html')

@app.route('/dental')
def route_page3():
    """Dental page"""
    return render_template('page3.html')

@app.route('/ocular')
def route_ocular():
    """Ocular disease page"""
    return render_template('ocular.html')  # or whatever you named your HTML file

# ==============================================================================
# LOGIN/LOGOUT ROUTES
# ==============================================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, send them to main app
    if session.get('logged_in'):
        logger.info("User already logged in, redirecting to main app")
        return redirect('/main')
    
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            remember = request.form.get('remember')
            
            logger.info(f"Login attempt - Username: '{username}'")
            
            if not username or not password:
                logger.warning("Missing username or password")
                return render_template('login.html', error='Please enter both username and password')
            
            # Check credentials
            if username in DOCTORS_DB and DOCTORS_DB[username]['password'] == password:
                # Clear any existing session
                session.clear()
                
                # Set up new session
                session['doctor'] = DOCTORS_DB[username]
                session['logged_in'] = True
                session['login_time'] = datetime.now().isoformat()
                
                if remember:
                    session.permanent = True
                
                session.modified = True
                
                logger.info(f"✓ LOGIN SUCCESS: {DOCTORS_DB[username]['name']} - Redirecting to main app")
                
                # Redirect to main application (index.html)
                return redirect('/main')
                
            else:
                logger.warning(f"✗ LOGIN FAILED: Invalid credentials for '{username}'")
                return render_template('login.html', error='Invalid username or password')
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return render_template('login.html', error='Login system error. Please try again.')
    
    # GET request - show login form
    return render_template('login.html')

@app.route('/logout')
def logout():
    if session.get('doctor'):
        logger.info(f"Doctor logged out: {session['doctor']['name']}")
    session.clear()
    logger.info("User logged out, redirecting to home (login)")
    return redirect('/')  # This will show login page since user is no longer logged in

@app.route('/api/login-status')
def login_status():
    """API endpoint to check login status"""
    if session.get('logged_in') and session.get('doctor'):
        return jsonify({
            'logged_in': True,
            'doctor': session['doctor']
        })
    else:
        return jsonify({
            'logged_in': False,
            'doctor': None
        })

# ==============================================================================
# FLASK API ENDPOINTS
# ==============================================================================

@app.route("/health", methods=['GET'])
def health_check():
    gemini_status = bool(os.getenv('GEMINI_API_KEY'))
    
    return jsonify({
        "status": "healthy",
        "models_loaded": model_loading_status,
        "environment": "huggingface_spaces",
        "gemini_enabled": gemini_status,
        "login_enabled": True,
        "facility": FACILITY_INFO['name']
    })

@app.route('/chest/predict', methods=['POST'])
def predict_chest_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Lazy load model if not loaded
    if chest_model is None:
        logger.info("Lazy loading chest model...")
        if not load_chest_model():
            return jsonify({'error': 'Chest model failed to load'}), 500
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        # Get AI prediction
        result, error = predict_chest_image(image)
        if error:
            return jsonify({'error': error}), 500
        
        # Extract patient info from filename
        patient_info = get_patient_info_or_unknown(file.filename)
        
        # Get current doctor info
        doctor_info = get_current_doctor()
        
        # Analyze image with Gemini
        image_analysis = None
        if os.getenv('GEMINI_API_KEY'):
            try:
                image_analysis = analyze_image_with_gemini(image, result)
            except Exception as e:
                logger.warning(f"Gemini analysis failed: {e}")
                image_analysis = "Gemini analysis unavailable"
        
        # Add enhanced information to result
        result.update({
            'patient_info': patient_info,
            'doctor_info': doctor_info,
            'image_analysis': image_analysis,
            'filename': file.filename,
            'facility_info': FACILITY_INFO
        })
        
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
    
    # Lazy load models if not loaded
    if fracture_binary_model is None or fracture_multiclass_model is None:
        logger.info("Lazy loading fracture models...")
        if not load_fracture_models():
            return jsonify({'error': 'Fracture models failed to load'}), 500
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        binary_result = predict_fracture_binary(image)
        response = {'stage1_prediction': binary_result}
        
        if binary_result['prediction'] == 'Fractured':
            multiclass_result = predict_fracture_type(image)
            response['stage2_prediction'] = multiclass_result
        else:
            response['stage2_prediction'] = None
        
        # Add patient and doctor info
        patient_info = get_patient_info_or_unknown(file.filename)
        doctor_info = get_current_doctor()
        
        # Gemini analysis
        image_analysis = None
        if os.getenv('GEMINI_API_KEY'):
            try:
                image_analysis = analyze_image_with_gemini(image, response)
            except Exception as e:
                logger.warning(f"Gemini analysis failed: {e}")
        
        response.update({
            'patient_info': patient_info,
            'doctor_info': doctor_info,
            'image_analysis': image_analysis,
            'filename': file.filename,
            'facility_info': FACILITY_INFO
        })
            
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
    
    # Lazy load model if not loaded
    if dental_yolo_model is None:
        logger.info("Lazy loading dental YOLO model...")
        if not load_dental_yolo_model():
            return jsonify({'error': 'Dental model failed to load'}), 500
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_yolo_image(image)
        
        results = dental_yolo_model.predict(processed_image, verbose=False)
        if not results:
            return jsonify({'error': 'Model returned no results'}), 500
        
        summary = yolo_to_classification_summary(results[0])
        
        # Add patient and doctor info
        patient_info = get_patient_info_or_unknown(file.filename)
        doctor_info = get_current_doctor()
        
        # Gemini analysis
        image_analysis = None
        if os.getenv('GEMINI_API_KEY'):
            try:
                image_analysis = analyze_image_with_gemini(image, summary)
            except Exception as e:
                logger.warning(f"Gemini analysis failed: {e}")
        
        response = {
            'predicted_class': summary['predicted_class'],
            'confidence': summary['confidence'],
            'probabilities': summary['probabilities'],
            'detection_details': {
                'total_detections': len(summary['detections']),
                'detections': summary['detections']
            },
            'patient_info': patient_info,
            'doctor_info': doctor_info,
            'image_analysis': image_analysis,
            'filename': file.filename,
            'facility_info': FACILITY_INFO
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in /dental/predict endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to process image'}), 400
    
@app.route('/ocular/predict', methods=['POST'])
def predict_ocular_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Lazy load model if not loaded
    if ocular_model is None:
        logger.info("Lazy loading ocular model...")
        if not load_ocular_model():
            return jsonify({'error': 'Ocular model failed to load'}), 500
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        # Get AI prediction
        result, error = predict_ocular_image(image)
        if error:
            return jsonify({'error': error}), 500
        
        # Extract patient info from filename
        patient_info = get_patient_info_or_unknown(file.filename)
        
        # Get current doctor info
        doctor_info = get_current_doctor()
        
        # Analyze image with Gemini
        image_analysis = None
        if os.getenv('GEMINI_API_KEY'):
            try:
                image_analysis = analyze_image_with_gemini(image, result)
            except Exception as e:
                logger.warning(f"Gemini analysis failed: {e}")
                image_analysis = "Gemini analysis unavailable"
        
        # Add enhanced information to result
        result.update({
            'patient_info': patient_info,
            'doctor_info': doctor_info,
            'image_analysis': image_analysis,
            'filename': file.filename,
            'facility_info': FACILITY_INFO
        })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /ocular/predict endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to process image'}), 400

@app.route('/request-access')
def route_request_access():
    """Serves the access request page."""
    return render_template('request_access.html')

@app.route('/handle-request-access', methods=['POST'])
def handle_request_access():
    """Handles the submission of the access request form."""
    try:
        full_name = request.form.get('fullName')
        email = request.form.get('email')
        department = request.form.get('department')
        justification = request.form.get('justification')
        
        # In a real app, you would save this to a database and notify admins.
        # For this demo, we just log it to the console.
        logger.info("="*30)
        logger.info("NEW ACCESS REQUEST RECEIVED")
        logger.info(f"  Name: {full_name}")
        logger.info(f"  Email: {email}")
        logger.info(f"  Department: {department}")
        logger.info(f"  Justification: {justification}")
        logger.info("="*30)

        # can render a simple confirmation page.
        return """
            <body style='background-color:#1a1a1a; color:white; font-family:sans-serif; text-align:center; padding-top:50px;'>
                <h1>Request Submitted</h1>
                <p>Thank you, {}. Your request has been sent for administrative review.</p>
                <a href='/login' style='color:#00d4ff;'>Return to Login</a>
            </body>
        """.format(full_name)

    except Exception as e:
        logger.error(f"Error handling access request: {e}")
        return "An error occurred. Please try again.", 500

@app.route('/forgot-password')
def route_forgot_password():
    """Serves the forgot password page."""
    return render_template('forgot_password.html')

@app.route('/handle-forgot-password', methods=['POST'])
def handle_forgot_password():
    """Handles the submission of the forgot password form."""
    try:
        identifier = request.form.get('identifier')
        
        # In a real app, you would find the user, generate a token,
        # and email them a reset link.
        # For this demo, we log it to the console.
        logger.info("="*30)
        logger.info("PASSWORD RESET REQUEST")
        logger.info(f"  Identifier: {identifier}")
        logger.info("  (SIMULATION) A password reset link would be sent if this user exists.")
        logger.info("="*30)

        return """
            <body style='background-color:#1a1a1a; color:white; font-family:sans-serif; text-align:center; padding-top:50px;'>
                <h1>Check Your Email</h1>
                <p>If an account exists for "{}", a password reset link has been sent.</p>
                <p>(This is a simulation. No email was actually sent.)</p>
                <a href='/login' style='color:#00d4ff;'>Return to Login</a>
            </body>
        """.format(identifier)

    except Exception as e:
        logger.error(f"Error handling password reset: {e}")
        return "An error occurred. Please try again.", 500

# ==============================================================================
# GEMINI-POWERED ENDPOINTS
# ==============================================================================
@app.route('/generate-report', methods=['POST'])
def generate_report_endpoint():
    try:
        data = request.json
        
        ai_results = data.get('ai_results')
        patient_info = data.get('patient_info')
        doctor_info = data.get('doctor_info')
        image_analysis = data.get('image_analysis')
        reason_of_visit = data.get('reason_of_visit', '')
        
        if not os.getenv('GEMINI_API_KEY'):
            return jsonify({'error': 'Gemini API not configured'}), 500
        
        # Generate report with Gemini
        report = generate_medical_report_with_gemini(
            patient_info, doctor_info, ai_results, image_analysis, reason_of_visit
        )
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({'error': 'Failed to generate report'}), 500

@app.route('/ask-assistant', methods=['POST'])
def ask_assistant_endpoint():
    try:
        data = request.json
        question = data.get('question', '')
        context = data.get('context', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if not os.getenv('GEMINI_API_KEY'):
            return jsonify({'error': 'Gemini API not configured'}), 500
        
        # Get response from Gemini
        response = ask_gemini_medical_question(question, context)
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        logger.error(f"Error in assistant: {e}")
        return jsonify({'error': 'Failed to get assistant response'}), 500

@app.route('/analyze-with-gemini', methods=['POST'])
def analyze_with_gemini_endpoint():
    try:
        data = request.json
        image_data = data.get('image_data')
        ai_results = data.get('ai_results')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        if not os.getenv('GEMINI_API_KEY'):
            return jsonify({'error': 'Gemini API not configured'}), 500
        
        # Convert base64 image to PIL Image
        image_data = image_data.split(',')[1] 
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Analyze with Gemini
        analysis = analyze_image_with_gemini(image, ai_results)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error in Gemini analysis: {e}")
        return jsonify({'error': 'Failed to analyze with Gemini'}), 500


def generate_enhanced_text_report(patient_info, doctor_info, facility_info, ai_results, image_analysis, reason_of_visit, exam_type):
    """
    Generate a clean, well-formatted text medical report
    """
    try:
        # Get current date and time
        current_datetime = datetime.now()
        report_date = current_datetime.strftime("%Y-%m-%d")
        report_time = current_datetime.strftime("%H:%M")
        
        # Prepare the text report
        report_lines = []
        
        # Header
        report_lines.extend([
            "=" * 80,
            "MEDICAL IMAGING REPORT".center(80),
            "=" * 80,
            "",
            f"Generated: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # Facility Information
        report_lines.extend([
            "FACILITY INFORMATION:",
            "-" * 40,
            f"Hospital: {facility_info.get('name', 'Unknown Hospital')}",
            f"Address: {facility_info.get('address', 'N/A')}",
            f"Phone: {facility_info.get('phone', 'N/A')}",
            f"Department: {doctor_info.get('department', 'Radiology')}",
            ""
        ])
        
        # Patient Information
        report_lines.extend([
            "PATIENT INFORMATION:",
            "-" * 40,
            f"Patient Name: {patient_info.get('patient_name', 'Unknown Patient')}",
            f"Patient ID: {patient_info.get('patient_id', 'N/A')}",
            f"Age: {patient_info.get('age', 'Unknown')} years",
            f"Gender: {format_gender(patient_info.get('gender', 'Unknown'))}",
            ""
        ])
        
        # Examination Details
        report_lines.extend([
            "EXAMINATION DETAILS:",
            "-" * 40,
            f"Exam Type: {exam_type}",
            f"Date & Time: {patient_info.get('upload_datetime', current_datetime.strftime('%Y-%m-%d %H:%M'))}",
            f"Referring Physician: {doctor_info.get('name', 'Unknown Doctor')}",
            f"Physician Title: {doctor_info.get('title', 'Physician')}",
            ""
        ])
        
        # Clinical Information
        report_lines.extend([
            "CLINICAL INFORMATION:",
            "-" * 40,
            f"Reason for Visit: {reason_of_visit or 'Routine examination'}",
            ""
        ])
        
        # AI Analysis Results
        report_lines.extend([
            "AI ANALYSIS RESULTS:",
            "-" * 40
        ])
        
        # Handle different AI result formats
        if ai_results.get('predicted_class'):
            # Single prediction (chest, dental)
            predicted_class = ai_results['predicted_class']
            confidence = ai_results.get('confidence', 0)
            probabilities = ai_results.get('probabilities', [])
            class_names = ai_results.get('class_names', [])
            
            report_lines.extend([
                f"Primary Finding: {predicted_class.replace('_', ' ').title()}",
                f"Confidence: {confidence:.1f}%",
                ""
            ])
            
            if class_names and probabilities:
                report_lines.append("Classification Probabilities:")
                for i, class_name in enumerate(class_names):
                    prob = probabilities[i] * 100 if i < len(probabilities) else 0
                    status = "✓" if class_name == predicted_class else " "
                    report_lines.append(f"  [{status}] {class_name.replace('_', ' ')}: {prob:.1f}%")
                report_lines.append("")
        
        elif ai_results.get('stage1_prediction'):
            # Two-stage prediction (fracture)
            stage1 = ai_results['stage1_prediction']
            stage2 = ai_results.get('stage2_prediction')
            
            report_lines.extend([
                "STAGE 1 - FRACTURE DETECTION:",
                f"Result: {stage1['prediction']}",
                f"Confidence: {stage1['confidence']:.1f}%",
                ""
            ])
            
            if stage1.get('probabilities'):
                report_lines.extend([
                    "Stage 1 Probabilities:",
                    f"  Not Fractured: {stage1['probabilities'][0] * 100:.1f}%",
                    f"  Fractured: {stage1['probabilities'][1] * 100:.1f}%",
                    ""
                ])
            
            if stage2:
                report_lines.extend([
                    "STAGE 2 - FRACTURE TYPE CLASSIFICATION:",
                    f"Fracture Type: {stage2['prediction']}",
                    f"Confidence: {stage2['confidence']:.1f}%",
                    ""
                ])
        
        # Detection details (for dental YOLO)
        if ai_results.get('detection_details'):
            detection_details = ai_results['detection_details']
            total_detections = detection_details.get('total_detections', 0)
            
            report_lines.extend([
                "OBJECT DETECTION RESULTS:",
                f"Total Objects Detected: {total_detections}",
                ""
            ])
            
            if detection_details.get('detections'):
                report_lines.append("Detected Objects:")
                for detection in detection_details['detections']:
                    class_name = detection.get('class_name', 'Unknown')
                    confidence = detection.get('confidence', 0) * 100
                    report_lines.append(f"  • {class_name.replace('_', ' ')}: {confidence:.1f}% confidence")
                report_lines.append("")
        
        # AI Generated Analysis (if available)
        if image_analysis:
            report_lines.extend([
                "AI GENERATED ANALYSIS:",
                "-" * 40,
                image_analysis,
                ""
            ])
        
        # Clinical Impression
        report_lines.extend([
            "IMPRESSION:",
            "-" * 40,
            generate_clinical_impression(ai_results),
            ""
        ])
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 40,
            generate_recommendations(ai_results),
            ""
        ])
        
        # Footer
        report_lines.extend([
            "REPORT INFORMATION:",
            "-" * 40,
            f"Reviewed by: {doctor_info.get('name', 'AI System')}",
            f"Report Date: {report_date}",
            f"Report Time: {report_time}",
            "",
            "DISCLAIMER:",
            "-" * 40,
            "This report contains AI-assisted analysis results. All findings should be",
            "verified by a qualified radiologist before making clinical decisions. This AI",
            "analysis is for diagnostic assistance only and should not replace professional",
            "medical judgment.",
            "",
            "=" * 80
        ])
        
        # Join all lines
        report_text = "\n".join(report_lines)
        
        return {
            'success': True,
            'report_text': report_text,
            'filename': generate_filename(patient_info, exam_type)
        }
        
    except Exception as e:
        logger.error(f"Error generating text report: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def format_gender(gender):
    """Format gender for display"""
    if gender and gender.upper() == 'M':
        return 'Male'
    elif gender and gender.upper() == 'F':
        return 'Female'
    else:
        return gender or 'Unknown'

def generate_clinical_impression(ai_results):
    """Generate clinical impression based on AI results"""
    if ai_results.get('predicted_class'):
        predicted_class = ai_results['predicted_class']
        confidence = ai_results.get('confidence', 0)
        
        if predicted_class.lower() == 'normal':
            return f"AI analysis suggests normal findings with {confidence:.1f}% confidence. No significant pathological findings detected."
        else:
            return f"AI analysis indicates {predicted_class.replace('_', ' ').lower()} with {confidence:.1f}% confidence. Clinical correlation and radiologist review recommended."
    
    elif ai_results.get('stage1_prediction'):
        stage1 = ai_results['stage1_prediction']
        stage2 = ai_results.get('stage2_prediction')
        
        if stage1['prediction'] == 'Fractured' and stage2:
            return f"AI analysis detected a fracture classified as {stage2['prediction']} with {stage2['confidence']:.1f}% confidence. Orthopedic consultation recommended."
        else:
            return f"AI analysis shows {stage1['prediction'].lower()} with {stage1['confidence']:.1f}% confidence."
    
    return "AI analysis completed. Clinical correlation recommended."

def generate_recommendations(ai_results):
    """Generate recommendations based on AI results"""
    recommendations = []
    
    if ai_results.get('predicted_class'):
        predicted_class = ai_results['predicted_class'].lower()
        
        if predicted_class == 'normal':
            recommendations.extend([
                "1. No immediate action required based on AI analysis",
                "2. Clinical correlation with patient symptoms",
                "3. Consider follow-up imaging if symptoms persist",
                "4. Radiologist review and final interpretation"
            ])
        elif 'covid' in predicted_class:
            recommendations.extend([
                "1. RT-PCR testing for COVID-19 confirmation",
                "2. Isolation protocols if positive",
                "3. Monitor oxygen saturation and vital signs",
                "4. Consider antiviral treatment per guidelines",
                "5. Contact tracing if confirmed positive"
            ])
        elif 'pneumonia' in predicted_class:
            recommendations.extend([
                "1. Clinical correlation with symptoms and vitals",
                "2. Consider blood cultures and sputum analysis",
                "3. Antibiotic therapy based on severity and guidelines",
                "4. Follow-up imaging if no clinical improvement",
                "5. Monitor respiratory status"
            ])
        elif 'tuberculosis' in predicted_class:
            recommendations.extend([
                "1. Sputum testing for acid-fast bacilli",
                "2. TB skin test or interferon-gamma release assay",
                "3. Contact tracing investigation",
                "4. Isolation until TB ruled out",
                "5. Consider anti-TB therapy if confirmed"
            ])
        else:
            recommendations.extend([
                "1. Radiologist review and interpretation",
                "2. Clinical correlation recommended",
                "3. Consider additional imaging if indicated",
                "4. Follow-up as clinically appropriate"
            ])
    
    elif ai_results.get('stage1_prediction'):
        stage1 = ai_results['stage1_prediction']
        
        if stage1['prediction'] == 'Fractured':
            recommendations.extend([
                "1. Immediate orthopedic consultation",
                "2. Immobilize affected area",
                "3. Pain management as appropriate",
                "4. Consider CT or MRI for surgical planning",
                "5. Monitor neurovascular status"
            ])
        else:
            recommendations.extend([
                "1. Clinical correlation with symptoms",
                "2. Consider soft tissue injury evaluation",
                "3. Follow-up imaging in 7-10 days if symptoms worsen",
                "4. Physical therapy evaluation if appropriate"
            ])
    
    else:
        recommendations.extend([
            "1. Radiologist review and final interpretation",
            "2. Clinical correlation recommended",
            "3. Follow-up as clinically indicated"
        ])
    
    return "\n".join(recommendations)

def generate_filename(patient_info, exam_type):
    """Generate appropriate filename for the report"""
    patient_name = patient_info.get('patient_name', 'Unknown')
    patient_name = patient_name.replace(' ', '_').replace('/', '_')
    
    exam_short = exam_type.replace(' ', '_').replace('-', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return f"medical_report_{patient_name}_{exam_short}_{timestamp}.txt"

# ==============================================================================
# FLASK ENDPOINT FOR SIMPLE TEXT REPORT
# ==============================================================================

@app.route('/generate-text-report', methods=['POST'])
def generate_text_report_endpoint():
    """
    Generate enhanced text medical report
    """
    try:
        data = request.json
        
        # Extract required data
        patient_info = data.get('patient_info', {})
        doctor_info = data.get('doctor_info', {})
        facility_info = data.get('facility_info', FACILITY_INFO)
        ai_results = data.get('ai_results', {})
        image_analysis = data.get('image_analysis', '')
        reason_of_visit = data.get('reason_of_visit', '')
        exam_type = data.get('exam_type', 'Medical Imaging Examination')
        
        # Validate required data
        if not ai_results:
            return jsonify({'error': 'AI results are required'}), 400
        
        # Generate text report
        report_result = generate_enhanced_text_report(
            patient_info=patient_info,
            doctor_info=doctor_info,
            facility_info=facility_info,
            ai_results=ai_results,
            image_analysis=image_analysis,
            reason_of_visit=reason_of_visit,
            exam_type=exam_type
        )
        
        if not report_result['success']:
            return jsonify({'error': report_result['error']}), 500
        
        return jsonify({
            'success': True,
            'report_text': report_result['report_text'],
            'filename': report_result['filename']
        })
        
    except Exception as e:
        logger.error(f"Error in text report generation: {e}")
        return jsonify({'error': 'Failed to generate text report'}), 500
    

# ==============================================================================
# MAIN EXECUTION BLOCK FOR HUGGING FACE SPACES
# ==============================================================================
if __name__ == '__main__':
    logger.info("Starting MedScan Ai Server for Hugging Face Spaces...")
    
    # Try to load models at startup (optional - will lazy load if needed)
    logger.info("\n--- Attempting to Load AI Models ---")
    try:
        load_chest_model()
    except Exception as e:
        logger.warning(f"Chest model not loaded at startup: {e}")
    
    try:
        load_fracture_models()
    except Exception as e:
        logger.warning(f"Fracture models not loaded at startup: {e}")
    
    try:
        load_dental_yolo_model()
    except Exception as e:
        logger.warning(f"Dental model not loaded at startup: {e}")

    try:
        load_ocular_model()  # Add this
    except Exception as e:
        logger.warning(f"Ocular model not loaded at startup: {e}")
    
    logger.info("--- Model Loading Attempt Complete ---\n")
    logger.info(f"Model status: {model_loading_status}")
    
    # Run the Flask app
    # Hugging Face Spaces expects the app to run on 0.0.0.0:7860
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)