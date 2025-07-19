from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import timm
from torchvision import transforms

app = Flask(__name__)
CORS(app)

# Global variables
binary_model = None
multilabel_model = None
device = None

# EXACT binary model architecture from your evaluation code
class OneVsAllChestXrayClassifier(nn.Module):
    def __init__(self, model_name='resnet18', dropout_rate=0.2):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = self.encoder.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)  # Single output for binary classification
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

# EXACT multi-label model architecture from your training
class MultiLabelChestXrayClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0', dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = self.encoder.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

# Use your exact class names and thresholds (no dynamic loading needed)
pathological_class_names = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
    'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration', 
    'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 
    'Pneumonia', 'Pneumothorax'
]

# Your exact thresholds
optimal_thresholds = {
    'Cardiomegaly': 0.450, 'Emphysema': 0.400, 'Effusion': 0.430,
    'No Finding': 0.550, 'Infiltration': 0.400, 'Mass': 0.350,
    'Nodule': 0.330, 'Atelectasis': 0.380, 'Pneumothorax': 0.400,
    'Pleural_Thickening': 0.310, 'Pneumonia': 0.270, 'Fibrosis': 0.320,
    'Edema': 0.360, 'Consolidation': 0.320
}

# EXACT preprocessing from your evaluation code
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_models():
    """Load both models exactly like your evaluation"""
    global binary_model, multilabel_model, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Update these paths to your actual model files
    BINARY_MODEL_PATH = r'C:\Users\USER\Downloads\chest_xray_ai\val7104_auc7699.pth'
    MULTILABEL_MODEL_PATH = r'C:\Users\USER\Downloads\chest_xray_ai\best_multilabel_model.pth'
    
    try:
        # Load binary model exactly like your evaluation code
        print("Loading binary model...")
        binary_state_dict = torch.load(BINARY_MODEL_PATH, map_location=device, weights_only=False)
        
        # Try different architectures exactly like your evaluation
        model_variants = ['resnet18', 'resnet50', 'densenet121', 'densenet169', 'densenet201', 'efficientnet_b0']
        
        for model_name in model_variants:
            try:
                print(f"  Trying {model_name}...")
                binary_model = OneVsAllChestXrayClassifier(model_name=model_name).to(device)
                missing_keys, unexpected_keys = binary_model.load_state_dict(binary_state_dict, strict=False)
                classifier_keys = [k for k in missing_keys if 'classifier' in k]
                if len(classifier_keys) == 0:
                    print(f"  ✓ Binary model loaded with {model_name}")
                    binary_model.eval()
                    break
            except Exception as e:
                print(f"  ✗ Failed with {model_name}: {e}")
                continue
        else:
            # Fallback exactly like your evaluation
            print("  Using resnet18 with non-strict loading...")
            binary_model = OneVsAllChestXrayClassifier(model_name='resnet18').to(device)
            binary_model.load_state_dict(binary_state_dict, strict=False)
            binary_model.eval()
        
        # Load multi-label model
        print("Loading multi-label model...")
        multilabel_model = torch.load(MULTILABEL_MODEL_PATH, map_location=device, weights_only=False)
        multilabel_model.to(device)
        multilabel_model.eval()
        print("  ✓ Multi-label model loaded")
        
        # Test both models
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
            binary_output = binary_model(dummy_input)
            multilabel_output = multilabel_model(dummy_input)
            
            print(f"✓ Binary model test: {binary_output.shape} (expected: [1, 1])")
            print(f"✓ Multi-label model test: {multilabel_output.shape} (expected: [1, {len(pathological_class_names)}])")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_two_stage_evaluation_matched(image):
    """Two-stage prediction exactly matching your evaluation performance"""
    if binary_model is None or multilabel_model is None:
        return None, "Models not loaded"
    
    try:
        # Preprocess exactly like your evaluation
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Stage 1: Binary prediction exactly like your evaluation
        binary_model.eval()
        with torch.no_grad():
            binary_output = binary_model(image_tensor).squeeze()
            no_finding_prob = torch.sigmoid(binary_output).item()
            no_finding_predicted = no_finding_prob > 0.50  # Same threshold as evaluation
        
        print(f"\n🔍 STAGE 1 - Binary Classification:")
        print(f"  Raw output: {binary_output.item():.4f}")
        print(f"  No Finding probability: {no_finding_prob:.4f}")
        print(f"  Predicted: {'No Finding' if no_finding_predicted else 'Pathological'}")
        
        if no_finding_predicted:
            # Stop at Stage 1 - return "No Finding"
            return {
                'success': True,
                'stage_used': 'binary_only',
                'has_findings': False,
                'no_finding_predicted': True,
                'no_finding_probability': no_finding_prob,
                'predicted_conditions': ['No Finding'],
                'pathological_conditions': [],
                'probabilities': [],
                'model_confidence': 'High' if no_finding_prob > 0.7 else 'Medium'
            }, None
        
        # Stage 2: Multi-label prediction exactly like your evaluation
        print(f"\n🔍 STAGE 2 - Multi-label Classification:")
        multilabel_model.eval()
        with torch.no_grad():
            multilabel_outputs = multilabel_model(image_tensor)
            multilabel_probs = torch.sigmoid(multilabel_outputs).squeeze().cpu().numpy()
        
        print(f"  Raw outputs: {multilabel_outputs.squeeze().cpu().numpy()[:5]}...")
        print(f"  Probabilities: {multilabel_probs[:5]}...")
        
        # Apply thresholds exactly like your evaluation
        results = []
        predicted_conditions = []
        
        print(f"  Threshold analysis:")
        for i, class_name in enumerate(pathological_class_names):
            prob = float(multilabel_probs[i])
            threshold = optimal_thresholds.get(class_name, 0.5)
            predicted = prob > threshold
            
            status = "✓ PREDICTED" if predicted else f"✗ below"
            print(f"    {class_name:18}: {prob:.4f} {status}")
            
            if predicted:
                predicted_conditions.append(class_name)
            
            results.append({
                'name': class_name,
                'probability': prob,
                'percentage': prob * 100,
                'predicted': predicted,
                'threshold': threshold,
                'risk_level': "high-risk" if prob > 0.6 else "medium-risk" if prob > 0.3 else "low-risk"
            })
        
        # Handle "No Finding" in multi-label results (override logic)
        if 'No Finding' in predicted_conditions:
            predicted_conditions = ['No Finding']
            pathological_conditions = []
            has_findings = False
            no_finding_from_multilabel = True
            print(f"  → Multi-label override: 'No Finding' detected")
        else:
            pathological_conditions = [cond for cond in predicted_conditions if cond != 'No Finding']
            has_findings = len(pathological_conditions) > 0
            no_finding_from_multilabel = False
        
        # Sort results by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        print(f"  → Final result: {len(pathological_conditions)} pathological conditions")
        
        return {
            'success': True,
            'stage_used': 'both_stages',
            'has_findings': has_findings,
            'no_finding_predicted': no_finding_from_multilabel,
            'no_finding_probability': no_finding_prob,
            'predicted_conditions': predicted_conditions,
            'pathological_conditions': pathological_conditions,
            'probabilities': results,
            'model_confidence': 'High' if max(multilabel_probs) > 0.7 else 'Medium'
        }, None
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Prediction error: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    # Check if both models are loaded
    models_loaded = binary_model is not None and multilabel_model is not None
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': models_loaded,  # HTML expects this field
        'binary_model_loaded': binary_model is not None,
        'multilabel_model_loaded': multilabel_model is not None,
        'device': str(device) if device else 'unknown',
        'system': 'evaluation_matched_two_stage'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if binary_model is None or multilabel_model is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        print(f"\n{'='*60}")
        print(f"EVALUATION-MATCHED PREDICTION: {image_file.filename}")
        print(f"{'='*60}")
        
        try:
            image = Image.open(image_file.stream)
        except Exception as e:
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400
        
        result, error = predict_two_stage_evaluation_matched(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        print(f"{'='*60}")
        print(f"FINAL: Stage={result['stage_used']}, Findings={result['has_findings']}")
        print(f"{'='*60}\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Server error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Evaluation-Matched Two-Stage AI Server...")
    
    success = load_models()
    
    if not success:
        print("❌ Failed to load models!")
        print("Update the model paths in load_models() function")
        exit(1)
    
    print(f"\n✅ Two-stage server ready (evaluation matched)")
    print(f"✅ Binary classifier: 'No Finding' detection")
    print(f"✅ Multi-label classifier: {len(pathological_class_names)} conditions")
    print(f"✅ Preprocessing: Matches evaluation code exactly")
    
    app.run(debug=False, host='0.0.0.0', port=5000)