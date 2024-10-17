import os
import uuid
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from datetime import datetime
import logging

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'Output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
LOGO_PATH = "logo.jpg"

# Ensure upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        logging.warning("No pose landmarks detected.")
        return None

    height, width, _ = image.shape
    keypoints = {}

    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        x, y = int(landmark.x * width), int(landmark.y * height)
        key = mp_pose.PoseLandmark(idx).name.lower()
        keypoints[key] = (x, y)

    # Estimate ASIS (Anterior Superior Iliac Spine) position
    if 'left_hip' in keypoints and 'right_hip' in keypoints:
        left_hip = np.array(keypoints['left_hip'])
        right_hip = np.array(keypoints['right_hip'])
        asis = tuple(((left_hip + right_hip) / 2).astype(int))
        asis = (asis[0], asis[1] - int(0.1 * (right_hip[1] - left_hip[1])))  # Adjust Y coordinate slightly upward
        keypoints['asis'] = asis

    return keypoints

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Clamp to avoid numerical issues
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_knee_deviation(hip, knee, ankle):
    # Calculate the vectors
    hip_to_ankle = np.array(ankle) - np.array(hip)
    hip_to_knee = np.array(knee) - np.array(hip)

    # Normalize the hip-to-ankle vector to get the vertical line direction
    if np.linalg.norm(hip_to_ankle) == 0:
        return 0, 'Aligned'

    hip_to_ankle_norm = hip_to_ankle / np.linalg.norm(hip_to_ankle)

    # Project the knee onto the hip-ankle line
    projection_length = np.dot(hip_to_knee, hip_to_ankle_norm)
    projection = projection_length * hip_to_ankle_norm

    # Calculate the acute angle at the knee
    knee_angle = calculate_angle(hip, knee, ankle)
    acute_knee_angle = 180 - knee_angle

    # Check if the knee is anterior or posterior relative to the vertical line
    deviation = hip_to_knee - projection

    # If the knee is anterior to the line, it's flexed; if posterior, it's hyperextended.
    if deviation[0] > 0:  # Assuming positive x is anterior
        return acute_knee_angle, 'Flexed'
    else:
        return acute_knee_angle, 'Hyperextended'

def analyze_anterior_view(keypoints, image_shape):
    # Existing function remains unchanged
    # ...
    # (Omitted for brevity; use the same implementation as provided)
    pass  # Replace with the original implementation

def analyze_lateral_view(keypoints, image_shape):
    # Existing function remains unchanged
    # ...
    # (Omitted for brevity; use the same implementation as provided)
    pass  # Replace with the original implementation

def draw_landmarks_and_angles(image, keypoints, view, analysis_results):
    # Existing function remains unchanged
    # ...
    # (Omitted for brevity; use the same implementation as provided)
    pass  # Replace with the original implementation

def generate_report(anterior_results, lateral_results, anterior_image_path, lateral_image_path):
    # Existing function remains unchanged
    # ...
    # (Omitted for brevity; use the same implementation as provided)
    pass  # Replace with the original implementation

def key_to_title(key):
    return ' '.join(word.capitalize() for word in key.replace('-', '_').split('_'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the files
        if 'anterior_image' not in request.files or 'lateral_image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        anterior_file = request.files['anterior_image']
        lateral_file = request.files['lateral_image']

        if anterior_file.filename == '' or lateral_file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if anterior_file and allowed_file(anterior_file.filename) and lateral_file and allowed_file(lateral_file.filename):
            anterior_filename = secure_filename(anterior_file.filename)
            lateral_filename = secure_filename(lateral_file.filename)

            anterior_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{anterior_filename}")
            lateral_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{lateral_filename}")

            anterior_file.save(anterior_path)
            lateral_file.save(lateral_path)

            logging.info(f"Uploaded anterior image: {anterior_path}")
            logging.info(f"Uploaded lateral image: {lateral_path}")

            # Process images
            try:
                anterior_image = cv2.imread(anterior_path)
                lateral_image = cv2.imread(lateral_path)

                if anterior_image is None or lateral_image is None:
                    flash("Failed to read one or both images.")
                    return redirect(request.url)

                anterior_keypoints = detect_keypoints(anterior_image)
                lateral_keypoints = detect_keypoints(lateral_image)

                anterior_results = analyze_anterior_view(anterior_keypoints, anterior_image.shape)
                lateral_results = analyze_lateral_view(lateral_keypoints, lateral_image.shape)

                annotated_anterior = draw_landmarks_and_angles(anterior_image, anterior_keypoints or {}, 'anterior',
                                                               anterior_results)
                annotated_lateral = draw_landmarks_and_angles(lateral_image, lateral_keypoints or {}, 'lateral',
                                                              lateral_results)

                # Generate unique filenames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                annotated_anterior_filename = f'annotated_anterior_{timestamp}.jpg'
                annotated_lateral_filename = f'annotated_lateral_{timestamp}.jpg'

                annotated_anterior_path = os.path.join(app.config['OUTPUT_FOLDER'], annotated_anterior_filename)
                annotated_lateral_path = os.path.join(app.config['OUTPUT_FOLDER'], annotated_lateral_filename)

                cv2.imwrite(annotated_anterior_path, annotated_anterior)
                cv2.imwrite(annotated_lateral_path, annotated_lateral)

                logging.info(f"Annotated anterior image saved: {annotated_anterior_path}")
                logging.info(f"Annotated lateral image saved: {annotated_lateral_path}")

                # Generate PDF report
                report_filename = f'posture_analysis_report_{timestamp}.pdf'
                report_path = os.path.join(app.config['OUTPUT_FOLDER'], report_filename)

                generate_report(anterior_results, lateral_results, annotated_anterior_path, annotated_lateral_path)

                logging.info(f"Report generated: {report_path}")

                # Cleanup uploaded images
                os.remove(anterior_path)
                os.remove(lateral_path)
                os.remove(annotated_anterior_path)
                os.remove(annotated_lateral_path)

                return redirect(url_for('download_file', filename=report_filename))
            except Exception as e:
                logging.error(f"An error occurred during processing: {str(e)}")
                flash("An error occurred during processing. Please try again.")
                return redirect(request.url)
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)

    return render_template('index.html', logo=LOGO_PATH)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)