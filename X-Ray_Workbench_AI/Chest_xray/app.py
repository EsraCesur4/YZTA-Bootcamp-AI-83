from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)


model = None
class_names = ['Normal', 'Tuberculosis', 'Pneumonia', 'COVID-19']
IMAGE_SIZE = 128  # Based on your model architecture

def load_model():
    """Load the TensorFlow/Keras model"""
    global model
    
    MODEL_PATH = r"C:\Users\USER\Downloads\YZTA-Bootcamp-AI-83\X-Ray_Workbench_AI\Chest_xray\model.h5"
    
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✓ Model loaded successfully")
        
        # Test the model
        test_input = np.random.random((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        test_output = model.predict(test_input, verbose=0)
        print(f"✓ Model test: Input shape {test_input.shape} -> Output shape {test_output.shape}")
        print(f"✓ Expected 4 classes: {len(class_names)}")
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_image(image):
    """Preprocess image for the model"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image):
    """Predict using the 4-class model"""
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, "Failed to preprocess image"
        
        print(f"\n🔍 4-CLASS PREDICTION:")
        print(f"  Input shape: {processed_image.shape}")
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        probabilities = predictions[0]  # Remove batch dimension
        
        print(f"  Raw predictions: {predictions}")
        print(f"  Probabilities: {probabilities}")
        
        # Get predicted class
        predicted_index = np.argmax(probabilities)
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100
        
        print(f"  Predicted class: {predicted_class} (index: {predicted_index})")
        print(f"  Confidence: {confidence:.2f}%")
        
        # Detailed probability breakdown
        print(f"  Probability breakdown:")
        for i, class_name in enumerate(class_names):
            prob_percent = probabilities[i] * 100
            status = "✓ PREDICTED" if i == predicted_index else ""
            print(f"    {class_name:12}: {prob_percent:6.2f}% {status}")
        
        return {
            'success': True,
            'predicted_class': str(predicted_class),
            'predicted_index': int(predicted_index),  # 👈 önemli
            'confidence': float(confidence),          # 👈 önemli
            'probabilities': [float(p) for p in probabilities.tolist()],  # 👈 önemli
            'class_names': list(map(str, class_names)),
            'model_type': '4-class CNN'
        }, None

        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Prediction error: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': 'CPU/GPU (TensorFlow)',
        'system': '4-class_cnn_classifier',
        'classes': class_names,
        'image_size': IMAGE_SIZE
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        print(f"\n{'='*60}")
        print(f"4-CLASS PREDICTION: {image_file.filename}")
        print(f"{'='*60}")
        
        try:
            # Load image
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            print(f"  Original image: {image.size} pixels, mode: {image.mode}")
            
        except Exception as e:
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400
        
        # Make prediction
        result, error = predict_image(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        print(f"{'='*60}")
        print(f"FINAL: {result['predicted_class']} ({result['confidence']:.1f}%)")
        print(f"{'='*60}\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Server error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available classes"""
    return jsonify({
        'classes': class_names,
        'num_classes': len(class_names)
    })

if __name__ == '__main__':
    print("Starting 4-Class Chest X-Ray AI Server...")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("Running on CPU")
    
    # Load model
    success = load_model()
    
    if not success:
        print("Failed to load model!")
        print("Please check the model path and ensure the .hdf5 file exists")
        exit(1)
    
    print(f"\n4-Class server ready!")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Model type: Custom CNN")
    
    app.run(debug=False, host='0.0.0.0', port=5000)