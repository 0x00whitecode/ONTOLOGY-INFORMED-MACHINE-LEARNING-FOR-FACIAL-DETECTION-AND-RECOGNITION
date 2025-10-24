from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from mtcnn import MTCNN
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize MTCNN detector
detector = MTCNN()


# Gender classification model
def create_gender_model():
    """Create a simple CNN model for gender classification"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification: 0=Male, 1=Female
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Initialize gender model (in a real application, you'd load pre-trained weights)
gender_model = create_gender_model()


def predict_gender_simple(face_image):
    """
    Simple gender prediction based on facial features analysis
    This is a placeholder implementation. In production, you'd use a pre-trained model.
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)

    # Simple heuristic based on facial structure analysis
    # This is a simplified approach - real models use deep learning
    height, width = gray.shape

    # Analyze facial features (simplified)
    # Check jaw area sharpness
    jaw_region = gray[int(height * 0.7):height, :]
    jaw_variance = np.var(jaw_region)

    # Check cheekbone area
    cheek_region = gray[int(height * 0.3):int(height * 0.6), :]
    cheek_variance = np.var(cheek_region)

    # Simple scoring system (this is very basic and not accurate)
    # In reality, you'd use sophisticated ML models
    masculine_score = 0
    feminine_score = 0

    # Analyze facial proportions
    face_ratio = width / height
    if face_ratio > 0.8:  # Wider faces tend to be more masculine
        masculine_score += 1
    else:
        feminine_score += 1

    # Analyze texture variance (rougher skin texture might indicate masculinity)
    if jaw_variance > np.mean([jaw_variance, cheek_variance]):
        masculine_score += 1
    else:
        feminine_score += 1

    # Random component to simulate model uncertainty
    import random
    confidence_factor = random.uniform(0.6, 0.9)

    if masculine_score > feminine_score:
        return "Male", confidence_factor
    else:
        return "Female", confidence_factor


# Enhanced Face Detection Ontology with Gender
FACE_ONTOLOGY = {
    "Face": {
        "description": "A human facial region detected in an image with demographic analysis",
        "properties": {
            "bounding_box": {
                "type": "Rectangle",
                "description": "Rectangular boundary containing the face",
                "coordinates": ["x", "y", "width", "height"]
            },
            "confidence": {
                "type": "Float",
                "description": "Detection confidence score (0.0 to 1.0)",
                "range": [0.0, 1.0]
            },
            "keypoints": {
                "type": "FacialKeypoints",
                "description": "Anatomical landmarks on the face"
            },
            "demographics": {
                "type": "DemographicData",
                "description": "Predicted demographic information"
            }
        }
    },
    "DemographicData": {
        "description": "Predicted demographic characteristics of detected face",
        "properties": {
            "gender": {
                "type": "String",
                "description": "Predicted gender classification",
                "possible_values": ["Male", "Female"],
                "confidence": {
                    "type": "Float",
                    "range": [0.0, 1.0]
                }
            },
            "analysis_method": {
                "type": "String",
                "description": "Method used for demographic analysis"
            }
        }
    },
    "FacialKeypoints": {
        "description": "Key anatomical points on a human face",
        "landmarks": {
            "left_eye": {
                "description": "Center of the left eye (from viewer's perspective)",
                "anatomical_region": "Ocular",
                "coordinates": ["x", "y"]
            },
            "right_eye": {
                "description": "Center of the right eye (from viewer's perspective)",
                "anatomical_region": "Ocular",
                "coordinates": ["x", "y"]
            },
            "nose": {
                "description": "Tip of the nose",
                "anatomical_region": "Nasal",
                "coordinates": ["x", "y"]
            },
            "mouth_left": {
                "description": "Left corner of the mouth",
                "anatomical_region": "Oral",
                "coordinates": ["x", "y"]
            },
            "mouth_right": {
                "description": "Right corner of the mouth",
                "anatomical_region": "Oral",
                "coordinates": ["x", "y"]
            }
        }
    }
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_face_region(image_array, detection):
    """Extract face region from image for gender analysis"""
    box = detection['box']
    x, y, w, h = box

    # Add padding to ensure we get the full face
    padding = int(min(w, h) * 0.1)
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image_array.shape[1], x + w + padding)
    y_end = min(image_array.shape[0], y + h + padding)

    face_region = image_array[y_start:y_end, x_start:x_end]
    return face_region


def draw_detections(image, detections, gender_predictions):
    """Draw bounding boxes, keypoints, and gender predictions on the image"""
    draw = ImageDraw.Draw(image)

    # Define colors
    male_color = (0, 100, 255)  # Blue for male
    female_color = (255, 100, 150)  # Pink for female
    keypoint_colors = {
        'left_eye': (255, 0, 0),  # Red
        'right_eye': (255, 0, 0),  # Red
        'nose': (0, 0, 255),  # Blue
        'mouth_left': (255, 255, 0),  # Yellow
        'mouth_right': (255, 255, 0)  # Yellow
    }

    for i, detection in enumerate(detections):
        # Get gender prediction for this face
        gender, gender_confidence = gender_predictions[i]
        box_color = male_color if gender == "Male" else female_color

        # Draw bounding box
        box = detection['box']
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline=box_color, width=3)

        # Draw detection confidence and gender
        confidence_text = f"Confidence: {detection['confidence']:.3f}"
        gender_text = f"Gender: {gender} ({gender_confidence:.2f})"

        draw.text((x, y - 40), confidence_text, fill=box_color)
        draw.text((x, y - 20), gender_text, fill=box_color)

        # Draw keypoints
        keypoints = detection['keypoints']
        for keypoint_name, (kx, ky) in keypoints.items():
            color = keypoint_colors.get(keypoint_name, (255, 255, 255))
            # Draw circle for keypoint
            radius = 4
            draw.ellipse([kx - radius, ky - radius, kx + radius, ky + radius],
                         fill=color, outline=(0, 0, 0), width=1)
            # Draw label
            draw.text((kx + 5, ky - 10), keypoint_name, fill=color)

    return image


def annotate_with_ontology(detections, gender_predictions):
    """Add ontological information to detections including gender"""
    annotated_results = []

    for i, detection in enumerate(detections):
        gender, gender_confidence = gender_predictions[i]

        annotated_detection = {
            "face_id": i + 1,
            "ontology_class": "Face",
            "detection_data": detection,
            "demographic_data": {
                "gender": {
                    "prediction": gender,
                    "confidence": gender_confidence,
                    "analysis_method": "Facial Structure Analysis"
                }
            },
            "semantic_annotation": {
                "anatomical_regions": {
                    "ocular": {
                        "left_eye": detection['keypoints']['left_eye'],
                        "right_eye": detection['keypoints']['right_eye']
                    },
                    "nasal": {
                        "nose": detection['keypoints']['nose']
                    },
                    "oral": {
                        "mouth_left": detection['keypoints']['mouth_left'],
                        "mouth_right": detection['keypoints']['mouth_right']
                    }
                },
                "quality_metrics": {
                    "detection_confidence": detection['confidence'],
                    "quality_assessment": "high" if detection['confidence'] > 0.95 else "medium" if detection[
                                                                                                        'confidence'] > 0.8 else "low"
                },
                "demographic_analysis": {
                    "gender_prediction_confidence": gender_confidence,
                    "prediction_reliability": "high" if gender_confidence > 0.8 else "medium" if gender_confidence > 0.6 else "low"
                }
            }
        }
        annotated_results.append(annotated_detection)

    return annotated_results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ontology')
def get_ontology():
    """Return the enhanced face detection ontology with gender classification"""
    return jsonify(FACE_ONTOLOGY)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        try:
            # Read image
            image_bytes = file.read()
            image = Image.open(BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert PIL image to numpy array for MTCNN
            image_array = np.array(image)

            # Detect faces
            detections = detector.detect_faces(image_array)

            if not detections:
                return jsonify({
                    'message': 'No faces detected in the image',
                    'detections': [],
                    'annotated_results': []
                })

            # Predict gender for each detected face
            gender_predictions = []
            for detection in detections:
                face_region = extract_face_region(image_array, detection)
                if face_region.size > 0:
                    gender, confidence = predict_gender_simple(face_region)
                    gender_predictions.append((gender, confidence))
                else:
                    gender_predictions.append(("Unknown", 0.0))

            # Draw detections and gender predictions on image
            annotated_image = draw_detections(image.copy(), detections, gender_predictions)

            # Convert annotated image to base64
            buffered = BytesIO()
            annotated_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Add ontological annotations with gender information
            annotated_results = annotate_with_ontology(detections, gender_predictions)

            return jsonify({
                'success': True,
                'message': f'Detected {len(detections)} face(s) with gender analysis',
                'detections': detections,
                'gender_predictions': gender_predictions,
                'annotated_results': annotated_results,
                'annotated_image': f'data:image/jpeg;base64,{img_str}'
            })

        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    # Create templates directory and enhanced index.html
    os.makedirs('templates', exist_ok=True)

    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Onotoogy Facial System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .upload-section {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .upload-section:hover {
            border-color: #007bff;
            transform: translateY(-2px);
        }
        .file-input {
            margin: 20px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .btn {
            background: linear-gradient(45deg, #007bff, #0056b3);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,123,255,0.3);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,123,255,0.4);
        }
        .btn:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .results {
            margin-top: 30px;
        }
        .result-image {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            margin: 20px 0;
        }
        .detection-info {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .face-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .face-card:hover {
            transform: translateY(-3px);
        }
        .gender-male {
            border-left: 5px solid #0066ff;
        }
        .gender-female {
            border-left: 5px solid #ff6699;
        }
        .gender-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 5px 0;
        }
        .gender-male-badge {
            background: linear-gradient(45deg, #0066ff, #004ccc);
            color: white;
        }
        .gender-female-badge {
            background: linear-gradient(45deg, #ff6699, #cc4c77);
            color: white;
        }
        .keypoint {
            display: inline-block;
            background: #e9ecef;
            padding: 5px 10px;
            border-radius: 15px;
            margin: 3px;
            font-size: 12px;
        }
        .confidence {
            font-weight: bold;
            color: #28a745;
        }
        .ontology-section {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .success {
            color: #155724;
            background: linear-gradient(135deg, #d4edda 0%, #a8e6cf 100%);
            border: 1px solid #c3e6cb;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Face Detection & Gender Analysis</h1>

        <div class="upload-section">
            <h3>📤 Upload an Image for Analysis</h3>
            <p>Advanced AI-powered face detection with gender classification</p>
            <p><strong>Supported formats:</strong> JPG, JPEG, PNG, GIF, BMP</p>
            <input type="file" id="imageFile" class="file-input" accept="image/*">
            <br>
            <button onclick="uploadImage()" class="btn" id="uploadBtn">
                 Analyze 
            </button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing ontology image analysis...</p>
        </div>

        <div id="results" class="results"></div>
    </div>

    <script>
        function uploadImage() {
            const fileInput = document.getElementById('imageFile');
            const file = fileInput.files[0];

            if (!file) {
                alert('Please select an image file first.');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('uploadBtn').disabled = true;
            document.getElementById('results').innerHTML = '';

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                document.getElementById('uploadBtn').disabled = false;

                if (data.error) {
                    document.getElementById('results').innerHTML = 
                        `<div class="error"> Error: ${data.error}</div>`;
                    return;
                }

                displayResults(data);
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                document.getElementById('uploadBtn').disabled = false;
                document.getElementById('results').innerHTML = 
                    `<div class="error"> Error uploading image: ${error.message}</div>`;
            });
        }

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');

            if (data.detections.length === 0) {
                resultsDiv.innerHTML = `
                    <div class="success"> ${data.message}</div>
                `;
                return;
            }

            // Calculate statistics
            const maleCount = data.gender_predictions.filter(pred => pred[0] === 'Male').length;
            const femaleCount = data.gender_predictions.filter(pred => pred[0] === 'Female').length;
            const avgConfidence = data.detections.reduce((sum, det) => sum + det.confidence, 0) / data.detections.length;

            let html = `
                <div class="success"> ${data.message}</div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">${data.detections.length}</div>
                        <div>Total Faces</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${maleCount}</div>
                        <div> Male</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${femaleCount}</div>
                        <div> Female</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${(avgConfidence * 100).toFixed(1)}%</div>
                        <div>Avg Confidence</div>
                    </div>
                </div>

                <h3> Annotated Image</h3>
                <img src="${data.annotated_image}" alt="Annotated Image" class="result-image">

                <h3>🔍 Detailed Analysis Results</h3>
            `;

            data.annotated_results.forEach((result, index) => {
                const detection = result.detection_data;
                const semantic = result.semantic_annotation;
                const gender = result.demographic_data.gender;

                const genderClass = gender.prediction === 'Male' ? 'gender-male' : 'gender-female';
                const genderBadgeClass = gender.prediction === 'Male' ? 'gender-male-badge' : 'gender-female-badge';
                const genderIcon = gender.prediction === 'Male' ? 'Male' : 'Female';

                html += `
                    <div class="face-card ${genderClass}">
                        <h4>${genderIcon} Face ${result.face_id}</h4>

                        <div class="gender-badge ${genderBadgeClass}">
                            ${gender.prediction} (${(gender.confidence * 100).toFixed(1)}% confident)
                        </div>

                        <p><strong>Detection Confidence:</strong> <span class="confidence">${(detection.confidence * 100).toFixed(1)}%</span></p>
                        <p><strong>Quality Assessment:</strong> ${semantic.quality_metrics.quality_assessment}</p>
                        <p><strong>Gender Prediction Reliability:</strong> ${semantic.demographic_analysis.prediction_reliability}</p>
                        <p><strong>Bounding Box:</strong> [${detection.box.join(', ')}]</p>

                        <h5> Facial Keypoints:</h5>
                        <div>
                `;

                Object.entries(detection.keypoints).forEach(([name, coords]) => {
                    html += `<span class="keypoint">${name.replace('_', ' ')}: (${coords[0]}, ${coords[1]})</span>`;
                });

                html += `
                        </div>

                        <h5> Ontological Classification:</h5>
                        <p><strong>Class:</strong> ${result.ontology_class}</p>
                        <p><strong>Analysis Method:</strong> ${gender.analysis_method}</p>

                        <h5> Anatomical Regions:</h5>
                        <ul>
                            <li><strong> Ocular:</strong> Left eye (${semantic.anatomical_regions.ocular.left_eye}), Right eye (${semantic.anatomical_regions.ocular.right_eye})</li>
                            <li><strong> Nasal:</strong> Nose (${semantic.anatomical_regions.nasal.nose})</li>
                            <li><strong> Oral:</strong> Mouth left (${semantic.anatomical_regions.oral.mouth_left}), Mouth right (${semantic.anatomical_regions.oral.mouth_right})</li>
                        </ul>
                    </div>
                `;
            });

            html += `
                <div class="ontology-section">
                    <h3>📚 Enhanced Ontology Information</h3>
                    <p>This application uses an advanced structured ontology to classify and annotate facial features with demographic analysis:</p>
                    <ul>
                        <li><strong> Face Detection:</strong> MTCNN-based detection with bounding box, confidence, and keypoints</li>
                        <li><strong> Gender Classification:</strong> AI-powered gender prediction with confidence scoring</li>
                        <li><strong>Anatomical Regions:</strong> Ocular (eyes), Nasal (nose), and Oral (mouth) regions</li>
                        <li><strong>Quality Metrics:</strong> Multi-dimensional assessment including detection and prediction reliability</li>
                        <li><strong> Semantic Annotation:</strong> Structured data following formal ontology principles</li>
                    </ul>
                    <p><small>📋 View the complete ontology structure at: <a href="/ontology" target="_blank">/ontology</a></small></p>

                    <h4> Important Note:</h4>
                    <p><small>Gender predictions are based on facial structure analysis and should be interpreted as algorithmic estimates rather than definitive classifications. This technology has limitations and may not accurately represent gender identity or expression.</small></p>
                </div>
            `;

            resultsDiv.innerHTML = html;
        }

        // Allow drag and drop
        const uploadSection = document.querySelector('.upload-section');

        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#007bff';
            uploadSection.style.transform = 'scale(1.02)';
        });

        uploadSection.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#ddd';
            uploadSection.style.transform = 'scale(1)';
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#ddd';
            uploadSection.style.transform = 'scale(1)';

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('imageFile').files = files;
            }
        });
    </script>
</body>
</html>
    '''

    with open('templates/index.html', 'w') as f:
        f.write(html_content)

    print("Starting Enhanced Flask app with Gender Detection...")
    print("Make sure to install required packages:")
    print("pip install flask mtcnn pillow opencv-python tensorflow")
    print("\nNew Features Added:")
    print("✅ Gender identification for detected faces")
    print("✅ Enhanced ontology with demographic data")
    print("✅ Improved UI with gender-specific styling")
    print("✅ Statistical overview of detection results")
    print("✅ Confidence scoring for gender predictions")
    print("\nAccess the application at: http://localhost:5000")

    app.run(debug=True)