# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = None
try:
    # Look for a trained model first
    model_path = Path('C:/Users/sasha/OneDrive/Desktop/yolo/dataset/best_model.pt')


    if model_path.exists():
        model = YOLO(model_path)
        print(f"Loaded trained model from {model_path}")
    else:
        # Fall back to pre-trained model
        model = YOLO('yolov8n.pt')
        print("Loaded pre-trained YOLOv8n model")
except Exception as e:
    print(f"Error loading model: {e}")

# Define class names
class_names = ['healthy', 'wssv', 'ems', 'ihhnv', 'fungal']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/detect', methods=['POST'])
# def detect():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400
    
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No image selected'}), 400
    
#     # Save the uploaded file
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)
    
#     # Process the image
#     try:
#         # Run detection
#         results = model(file_path, conf=0.25)
        
#         # Get highest confidence prediction
#         if len(results[0].boxes) > 0:
#             confidences = results[0].boxes.conf.cpu().numpy()
#             classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
#             if len(classes) > 0:
#                 max_conf_idx = np.argmax(confidences)
#                 pred_class = int(classes[max_conf_idx])
#                 confidence = float(confidences[max_conf_idx])
#             else:
#                 pred_class = 0  # Default to healthy if no detection
#                 confidence = 0.0
#         else:
#             pred_class = 0  # Default to healthy if no detection
#             confidence = 0.0
        
#         # Get class name
#         class_name = class_names[pred_class] if pred_class < len(class_names) else "unknown"
        
#         # Generate recommendations based on class
#         recommendations = get_recommendations(class_name)
#         description = get_description(class_name)
        
#         # Create result image with annotations
#         img = cv2.imread(file_path)
#         for result in results:
#             result_plotted = result.plot()
        
#         # Save the result image
#         result_path = os.path.join(UPLOAD_FOLDER, f"result_{file.filename}")
#         cv2.imwrite(result_path, result_plotted)
        
#         # Return results
#         return jsonify({
#             'class': class_name,
#             'confidence': round(confidence * 100, 2),
#             'description': description,
#             'recommendations': recommendations,
#             'image_path': f"/uploads/result_{file.filename}"
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/detect', methods=['POST'])
# def detect():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No image selected'}), 400

#     # Save the uploaded file
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)

#     try:
#         # Run detection
#         results = model(file_path, conf=0.25)
#         boxes = results[0].boxes

#         if boxes is not None and len(boxes) > 0:
#             confidences = boxes.conf.cpu().numpy()
#             classes = boxes.cls.cpu().numpy().astype(int)

#             # Get highest confidence detection overall
#             max_conf_idx = int(np.argmax(confidences))
#             pred_class = int(classes[max_conf_idx])
#             confidence = float(confidences[max_conf_idx])

#             # Optionally: Apply a minimum threshold for "disease"
#             MIN_DISEASE_CONFIDENCE = 0.60  # or even 0.65 depending on your use-case
#         if pred_class != 0 and confidence < MIN_DISEASE_CONFIDENCE:
#             pred_class = 0  # treat as healthy
#             confidence = 1.0  # full confidence for healthy

#         else:
#             # No detections at all
#             pred_class = 0
#             confidence = 0.0

#         # Get class names from model or fallback
#         try:
#             detected_class_names = model.names
#         except:
#             detected_class_names = class_names

#         class_name = detected_class_names.get(pred_class, "unknown") if isinstance(detected_class_names, dict) else class_names[pred_class]

#         # Get description and recommendations
#         description = get_description(class_name)
#         recommendations = get_recommendations(class_name)

#         # Annotated result image
#         img = cv2.imread(file_path)
#         for result in results:
#             result_plotted = result.plot()
#         result_path = os.path.join(UPLOAD_FOLDER, f"result_{file.filename}")
#         cv2.imwrite(result_path, result_plotted)

#         return jsonify({
#             'class': class_name,
#             'confidence': round(confidence * 100, 2),
#             'description': description,
#             'recommendations': recommendations,
#             'image_path': f"/uploads/result_{file.filename}"
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        results = model(file_path, conf=0.25)
        boxes = results[0].boxes

        # Default to healthy
        pred_class = 1
        confidence = 1.0

        if boxes is not None and len(boxes) > 0:
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            max_conf_idx = int(np.argmax(confidences))
            pred_class_candidate = int(classes[max_conf_idx])
            confidence_candidate = float(confidences[max_conf_idx])

            # MIN_DISEASE_CONFIDENCE = 0.50

            # if pred_class_candidate != 0 and confidence_candidate >= MIN_DISEASE_CONFIDENCE:
            #     pred_class = pred_class_candidate
            #     confidence = confidence_candidate
            # else:
            #     pred_class = 0
            #     confidence = 1.0  # Assume healthy with full confidence
            MIN_DISEASE_CONFIDENCE = 0.50

            pred_class = 1  # Healthy by default
            confidence = 1.0

            if boxes is not None and len(boxes) > 0:
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                # Filter out boxes that are confidently diseased (class != 0)
                filtered_indices = [i for i, (cls, conf) in enumerate(zip(classes, confidences))
                                    if cls != 1 and conf >= MIN_DISEASE_CONFIDENCE]

                # if filtered_indices:
                #     # Pick the one with the highest confidence among filtered
                #     best_idx = max(filtered_indices, key=lambda i: confidences[i])
                #     pred_class = int(classes[best_idx])
                #     confidence = float(confidences[best_idx])
                if filtered_indices:
                    best_idx = max(filtered_indices, key=lambda i: confidences[i])
                    pred_class = int(classes[best_idx])
                    confidence = float(confidences[best_idx])
                else:
                    # Explicitly mark as healthy
                    pred_class = 1
                    confidence = 1.0



        # Get class names
        try:
            detected_class_names = model.names
        except:
            detected_class_names = class_names

        class_name = detected_class_names.get(pred_class, "unknown") if isinstance(detected_class_names, dict) else class_names[pred_class]

        # Get description and recommendations
        description = get_description(class_name)
        recommendations = get_recommendations(class_name)

        # Annotate image
        img = cv2.imread(file_path)
        for result in results:
            result_plotted = result.plot()
        result_path = os.path.join(UPLOAD_FOLDER, f"result_{file.filename}")
        cv2.imwrite(result_path, result_plotted)

        return jsonify({
            'class': class_name,
            'confidence': round(confidence * 100, 2),
            'description': description,
            'recommendations': recommendations,
            'image_path': f"/uploads/result_{file.filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_recommendations(class_name):
    recommendations = {
        'healthy': [
            'Continue regular health monitoring',
            'Maintain optimal water quality',
            'Ensure proper nutrition'
        ],
        'wssv': [
            'Isolate affected individuals',
            'Implement emergency disease control measures',
            'Consider emergency harvest if severe',
            'Improve biosecurity protocols'
        ],
        'ems': [
            'Apply approved probiotics',
            'Improve water quality immediately',
            'Consider antibiotic treatment if allowed',
            'Implement biofloc technology if applicable'
        ],
        'ihhnv': [
            'Monitor population for growth rates',
            'Consider selective harvesting',
            'Implement strict biosecurity',
            'Use SPF stock for next crop'
        ],
        'fungal': [
            'Treat with approved antifungal treatments',
            'Improve water quality and aeration',
            'Remove affected individuals',
            'Reduce organic load in ponds'
        ]
    }
    return recommendations.get(class_name, ['Consult with an aquaculture specialist'])

def get_description(class_name):
    descriptions = {
        'healthy': 'This shrimp appears healthy with no visible signs of disease.',
        'wssv': 'Signs of White Spot Syndrome Virus detected, characterized by white spots on the carapace and reddish discoloration.',
        'ems': 'Early Mortality Syndrome/Acute Hepatopancreatic Necrosis Disease detected, characterized by pale hepatopancreas.',
        'ihhnv': 'Infectious Hypodermal and Hematopoietic Necrosis Virus detected, typically causing deformities and growth retardation.',
        'fungal': 'Fungal infection detected, often appearing as cotton-like growths or discolored patches on the body or gills.'
    }
    return descriptions.get(class_name, 'Unknown condition detected. Please consult with a specialist.')

if __name__ == '__main__':
    app.run(debug=True)
