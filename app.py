import os
import cv2
import numpy as np
import mediapipe as mp
import requests
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Image as RLImage, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from datetime import datetime
import logging
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tempfile
import io

# =============================
# Configuration and Setup
# =============================

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_default_secret_key')  # Replace with your secret key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload size
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png']
app.config['UPLOAD_PATH'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_PATH'], exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# =============================
# Intervention Content
# =============================

INTERVENTION_CONTENT = {
    "Head Tilt": [
        "1. Neck Stretch: Gently tilt your head towards your shoulder and hold for 15 seconds on each side.",
        "2. Chin Tucks: Slowly tuck your chin to your chest and hold for 10 seconds. Repeat 5 times."
    ],
    "Shoulder Tilt": [
        "1. Shoulder Rolls: Roll your shoulders forward in a circular motion 10 times, then backward 10 times.",
        "2. Wall Angels: Stand against a wall and slowly raise and lower your arms in a 'snow angel' motion."
    ],
    "Pelvic Tilt": [
        "1. Cat-Cow Stretch: On all fours, alternate arching and rounding your back. Repeat 10 times.",
        "2. Bridge Exercise: Lie on your back with knees bent, lift your hips towards the ceiling and hold for 5 seconds. Repeat 10 times."
    ],
    "Forward Head": [
        "1. Chin Tucks: Similar to above.",
        "2. Thoracic Extensions: Use a foam roller to extend your upper back gently."
    ],
    "Knee Angle": [
        "1. Quad Stretch: Stand on one leg, pull your other foot towards your buttocks, and hold for 15 seconds. Repeat on both sides.",
        "2. Hamstring Curls: Bend your knees and bring your heels towards your buttocks. Repeat 15 times."
    ],
    "Feet Rotation": [
        "1. Toe Taps: Tap your toes forward and backward repeatedly for 30 seconds.",
        "2. Ankle Circles: Rotate your ankles in both clockwise and counter-clockwise directions."
    ],
    "General Recommendations": [
        "1. Maintain proper posture throughout the day.",
        "2. Take regular breaks to stretch if you sit for extended periods.",
        "3. Strengthen your core muscles to support better posture."
    ]
}

# =============================
# Helper Functions
# =============================

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in app.config['UPLOAD_EXTENSIONS']

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
    results = {}

    if not keypoints:
        return {'error': ('Detection failed', 0, 'Unable to analyze')}

    # Head Tilt
    if 'left_eye' in keypoints and 'right_eye' in keypoints:
        left_eye = np.array(keypoints['left_eye'])
        right_eye = np.array(keypoints['right_eye'])
        head_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        head_tilt = abs(180 - abs(head_angle))
        if head_tilt > 3:
            direction = 'Right' if head_angle > 0 else 'Left'
            results['head_tilt'] = ('Severe', head_tilt, direction)
        elif 2 <= head_tilt <= 3:
            direction = 'Right' if head_angle > 0 else 'Left'
            results['head_tilt'] = ('Mild', head_tilt, direction)
        else:
            results['head_tilt'] = ('Normal', head_tilt, 'Centered')
    else:
        results['head_tilt'] = ('Not detected', 0, 'Unable to analyze')

    # Shoulder Horizontal Tilt
    if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
        left_shoulder = np.array(keypoints['left_shoulder'])
        right_shoulder = np.array(keypoints['right_shoulder'])
        shoulder_angle = np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0]))
        shoulder_tilt = 180 - abs(shoulder_angle)
        if shoulder_tilt > 3:
            direction = 'Right' if shoulder_angle > 0 else 'Left'
            results['shoulder_tilt'] = ('Severe', shoulder_tilt, direction)
        elif 2 <= shoulder_tilt <= 3:
            direction = 'Right' if shoulder_angle > 0 else 'Left'
            results['shoulder_tilt'] = ('Mild', shoulder_tilt, direction)
        else:
            results['shoulder_tilt'] = ('Normal', shoulder_tilt, 'Level')
    else:
        results['shoulder_tilt'] = ('Not detected', 0, 'Unable to analyze')

    # Pelvic Horizontal Tilt
    if 'left_hip' in keypoints and 'right_hip' in keypoints:
        left_hip = np.array(keypoints['left_hip'])
        right_hip = np.array(keypoints['right_hip'])
        pelvic_angle = np.degrees(np.arctan2(right_hip[1] - left_hip[1], right_hip[0] - left_hip[0]))
        pelvic_tilt = 180 - abs(pelvic_angle)
        if pelvic_tilt > 3:
            direction = 'Right' if pelvic_angle > 0 else 'Left'
            results['pelvic_tilt'] = ('Severe', pelvic_tilt, direction)
        elif 2 <= pelvic_tilt <= 3:
            direction = 'Right' if pelvic_angle > 0 else 'Left'
            results['pelvic_tilt'] = ('Mild', pelvic_tilt, direction)
        else:
            results['pelvic_tilt'] = ('Normal', pelvic_tilt, 'Level')
    else:
        results['pelvic_tilt'] = ('Not detected', 0, 'Unable to analyze')

    # Knee Valgus/Varus
    for side in ['left', 'right']:
        if f'{side}_hip' in keypoints and f'{side}_knee' in keypoints and f'{side}_ankle' in keypoints:
            hip = np.array(keypoints[f'{side}_hip'])
            knee = np.array(keypoints[f'{side}_knee'])
            ankle = np.array(keypoints[f'{side}_ankle'])

            hip_ankle_vector = ankle - hip
            hip_knee_vector = knee - hip
            cross_product = np.cross(hip_ankle_vector[:2], hip_knee_vector[:2])
            angle = abs(calculate_angle(hip, knee, ankle) - 180)
            knee_deviation = 'Valgus' if (cross_product > 0 and side == 'left') or (cross_product < 0 and side == 'right') else 'Varus'

            if angle > 15:
                results[f'{side}_knee'] = ('Severe', angle, knee_deviation)
            elif 5 <= angle <= 15:
                results[f'{side}_knee'] = ('Mild', angle, knee_deviation)
            else:
                results[f'{side}_knee'] = ('Normal', angle, 'Aligned')
        else:
            results[f'{side}_knee'] = ('Not detected', 0, 'Unable to analyze')

    # Feet Rotation
    for side in ['left', 'right']:
        if f'{side}_ankle' in keypoints and f'{side}_foot_index' in keypoints:
            ankle = np.array(keypoints[f'{side}_ankle'])
            toe = np.array(keypoints[f'{side}_foot_index'])
            foot_angle = np.degrees(np.arctan2(toe[0] - ankle[0], ankle[1] - toe[1]))
            foot_angle = 180 - abs(foot_angle)

            if foot_angle > 30:
                results[f'{side}_foot_rotation'] = ('Severe', foot_angle, 'Externally rotated')
            elif 18 < foot_angle <= 30:
                results[f'{side}_foot_rotation'] = ('Mild', foot_angle, 'Externally rotated')
            elif 5 <= foot_angle <= 18:
                results[f'{side}_foot_rotation'] = ('Normal', foot_angle, 'Aligned')
            elif 0 <= foot_angle < 5:
                results[f'{side}_foot_rotation'] = ('Mild', foot_angle, 'Internally rotated')
            else:
                results[f'{side}_foot_rotation'] = ('Severe', abs(foot_angle), 'Internally rotated')
        else:
            results[f'{side}_foot_rotation'] = ('Not detected', 0, 'Unable to analyze')

    return results

def analyze_lateral_view(keypoints, image_shape):
    results = {}
    height, width = image_shape[:2]
    vertical_line_x = width // 2

    if not keypoints:
        return {'error': ('Detection failed', 0, 'Unable to analyze')}

    # Forward Head
    if 'right_ear' in keypoints and 'right_shoulder' in keypoints:
        ear = np.array(keypoints['right_ear'])
        shoulder = np.array(keypoints['right_shoulder'])
        forward_head_distance = (ear[0] - shoulder[0]) / width * 100  # Convert to percentage of image width
        if forward_head_distance > 5:  # Assuming 5% of image width as threshold (adjust as needed)
            results['forward_head'] = ('Severe', forward_head_distance, 'Forward')
        elif forward_head_distance > 2:
            results['forward_head'] = ('Mild', forward_head_distance, 'Forward')
        else:
            results['forward_head'] = ('Normal', forward_head_distance, 'Aligned')
    else:
        results['forward_head'] = ('Not detected', 0, 'Unable to analyze')

    # Round Shoulders
    if 'right_shoulder' in keypoints and 'right_hip' in keypoints:
        shoulder = np.array(keypoints['right_shoulder'])
        hip = np.array(keypoints['right_hip'])
        shoulder_angle = np.degrees(np.arctan2(shoulder[0] - hip[0], hip[1] - shoulder[1]))
        if shoulder_angle > 30:
            results['round_shoulders'] = ('Severe', shoulder_angle, 'Rounded')
        elif shoulder_angle > 20:
            results['round_shoulders'] = ('Mild', shoulder_angle, 'Rounded')
        else:
            results['round_shoulders'] = ('Normal', shoulder_angle, 'Aligned')
    else:
        results['round_shoulders'] = ('Not detected', 0, 'Unable to analyze')

    # Pelvic Tilt
    if 'right_hip' in keypoints and 'right_knee' in keypoints:
        hip = keypoints['right_hip']
        knee = keypoints['right_knee']
        ankle = keypoints.get('right_ankle', hip)  # Use hip if ankle not detected to avoid errors

        pelvic_angle = np.degrees(np.arctan2(knee[0] - hip[0], hip[1] - knee[1]))

        if pelvic_angle > 0:
            results['pelvic_tilt'] = ('Posterior', pelvic_angle, 'Tilted')
        elif pelvic_angle < 0:
            results['pelvic_tilt'] = ('Anterior', abs(pelvic_angle), 'Tilted')
        else:
            results['pelvic_tilt'] = ('Normal', pelvic_angle, 'Neutral')
    else:
        results['pelvic_tilt'] = ('Not detected', 0, 'Unable to analyze')

    # Knee Flexion/Hyperextension (Right Lateral View)
    if 'right_hip' in keypoints and 'right_knee' in keypoints and 'right_ankle' in keypoints:
        hip = keypoints['right_hip']
        knee = keypoints['right_knee']
        ankle = keypoints['right_ankle']
        deviation_angle, direction = calculate_knee_deviation(hip, knee, ankle)

        # Determine severity based on the deviation angle
        if direction == 'Flexed':
            if deviation_angle > 10:
                status = 'Severe'
            elif 5 < deviation_angle <= 10:
                status = 'Mild'
            else:
                status = 'Normal'
        elif direction == 'Hyperextended':
            if deviation_angle > 10:
                status = 'Severe'
            elif 5 < deviation_angle <= 10:
                status = 'Mild'
            else:
                status = 'Normal'

        # Format the result as: "Knee Angle: Mild/Severe/Normal (Angle) - Hyperextended/Flexed"
        results['knee_angle'] = (status, deviation_angle, direction)
    else:
        results['knee_angle'] = ('Not detected', 0, 'Unable to analyze')

    return results

def draw_landmarks_and_angles(image, keypoints, view, analysis_results):
    annotated_image = image.copy()
    height, width, _ = annotated_image.shape

    # Draw vertical line for perfect alignment
    cv2.line(annotated_image, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)  # Blue vertical line

    def draw_point(p, color):
        cv2.circle(annotated_image, p, 5, color, -1)

    def draw_line(p1, p2, color):
        cv2.line(annotated_image, p1, p2, color, 2)

    # Draw keypoints
    for part, point in keypoints.items():
        draw_point(point, (0, 255, 0))

    # Define connections and draw them
    if view == 'anterior':
        connections = [
            ('left_eye', 'right_eye', 'head_tilt'),
            ('left_shoulder', 'right_shoulder', 'shoulder_tilt'),
            ('left_hip', 'right_hip', 'pelvic_tilt'),
            ('left_shoulder', 'left_elbow', None),
            ('left_elbow', 'left_wrist', None),
            ('right_shoulder', 'right_elbow', None),
            ('right_elbow', 'right_wrist', None),
            ('left_shoulder', 'left_hip', None),
            ('right_shoulder', 'right_hip', None),
            ('left_hip', 'left_knee', 'left_knee'),
            ('right_hip', 'right_knee', 'right_knee'),
            ('left_knee', 'left_ankle', 'left_knee'),
            ('right_knee', 'right_ankle', 'right_knee'),
            ('left_ankle', 'left_foot_index', 'left_foot_rotation'),
            ('right_ankle', 'right_foot_index', 'right_foot_rotation')
        ]
    elif view == 'lateral':
        connections = [
            ('right_ear', 'right_shoulder', 'forward_head'),
            ('right_shoulder', 'right_hip', 'round_shoulders'),
            ('right_hip', 'right_knee', 'pelvic_tilt'),
            ('right_knee', 'right_ankle', 'knee_angle')
        ]

    for start, end, analysis_key in connections:
        color = (0, 255, 0)  # Default green
        if analysis_key and analysis_key in analysis_results:
            status = analysis_results[analysis_key][0]
            if 'Severe' in status:
                color = (0, 0, 255)  # Red if severe
            elif 'Mild' in status:
                color = (0, 165, 255)  # Orange if mild
        draw_line(keypoints[start], keypoints[end], color)

    if view == 'anterior':
        if ('left_shoulder' in keypoints and 'right_shoulder' in keypoints and
            'left_hip' in keypoints and 'right_hip' in keypoints):
            mid_shoulder = tuple(((np.array(keypoints['left_shoulder']) +
                                   np.array(keypoints['right_shoulder'])) / 2).astype(int))
            mid_hip = tuple(((np.array(keypoints['left_hip']) +
                             np.array(keypoints['right_hip'])) / 2).astype(int))
            if 'left_eye' in keypoints and 'right_eye' in keypoints:
                mid_eye = tuple(((np.array(keypoints['left_eye']) +
                                 np.array(keypoints['right_eye'])) / 2).astype(int))
                draw_line(mid_eye, mid_shoulder, (0, 255, 0))
            draw_line(mid_shoulder, mid_hip, (0, 255, 0))

    return annotated_image

def generate_report(anterior_results, lateral_results, anterior_image_path, lateral_image_path, report_path):
    try:
        doc = SimpleDocTemplate(report_path, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                                leftMargin=2.5 * cm, rightMargin=0.5 * inch)
        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        styles.add(ParagraphStyle(name='CustomBodyText', parent=styles['Normal'], fontSize=9, leading=12))
        styles.add(ParagraphStyle(name='CustomHeading1', parent=styles['Heading1'], fontSize=16, textColor=colors.darkblue,
                                  alignment=1))
        styles.add(ParagraphStyle(name='CustomHeading2', parent=styles['Heading2'], fontSize=12, textColor=colors.darkgreen,
                                  spaceAfter=6))
        styles.add(ParagraphStyle(name='CustomHeading3', parent=styles['Heading3'], fontSize=10, textColor=colors.black,
                                  spaceBefore=6, spaceAfter=3, bold=True))

        # Background and frame
        background_color = colors.Color(1, 0.9, 0.8)  # Light orange
        frame_color = colors.white

        # Add logo from external URL
        logo_url = "https://your-cdn.com/logo.jpg"  # Replace with your actual logo URL
        try:
            response = requests.get(logo_url)
            response.raise_for_status()
            logo_image = PILImage.open(io.BytesIO(response.content))
            logo_byte_arr = io.BytesIO()
            logo_image.save(logo_byte_arr, format='JPEG')
            logo_byte_arr.seek(0)
            logo = RLImage(logo_byte_arr, width=6 * inch, height=0.5 * inch, kind='proportional')
            story.append(logo)
            story.append(Spacer(1, 6))
        except Exception as e:
            logging.error(f"Error fetching logo: {str(e)}")
            story.append(Paragraph("Posture Analysis Report", styles['CustomHeading1']))
            story.append(Spacer(1, 6))

        # Title
        if logo_url:
            pass  # Title already added if logo is fetched
        else:
            story.append(Paragraph("Posture Analysis Report", styles['CustomHeading1']))
            story.append(Spacer(1, 6))

        # Date
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['CustomBodyText']))
        story.append(Spacer(1, 6))

        # Images
        # Convert image paths to in-memory images
        def convert_cv2_to_rlimage(path, width, height):
            pil_image = PILImage.open(path)
            pil_image = pil_image.convert("RGB")
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            return RLImage(img_byte_arr, width=width, height=height)

        img_width = 3 * inch
        img_height = 4 * inch
        try:
            anterior_rl_image = convert_cv2_to_rlimage(anterior_image_path, img_width, img_height)
            lateral_rl_image = convert_cv2_to_rlimage(lateral_image_path, img_width, img_height)
            images_table = Table([[anterior_rl_image, lateral_rl_image]],
                                 colWidths=[3.5 * inch, 3.5 * inch],
                                 hAlign='CENTER')
            story.append(images_table)
            story.append(Spacer(1, 6))
        except Exception as e:
            logging.error(f"Error converting images for report: {str(e)}")

        # Analysis results
        def create_analysis_table(title, results):
            data = [[Paragraph(title, styles['CustomHeading2'])]]
            for key, value in results.items():
                if isinstance(value, tuple) and len(value) == 3:
                    status, measurement, direction = value
                    if 'Severe' in status:
                        color = 'red'
                        bold = 'bold'
                    elif 'Mild' in status:
                        color = 'darkorange'
                        bold = 'bold'
                    else:
                        color = 'black'
                        bold = 'normal'

                    direction_color = 'black'
                    if 'Left' in direction or 'Varus' in direction or 'Externally' in direction:
                        direction_color = 'darkblue'
                    elif 'Right' in direction or 'Valgus' in direction or 'Internally' in direction:
                        direction_color = 'darkred'

                    if key == 'pelvic_tilt' and title == "Lateral View Analysis":
                        data.append([Paragraph(
                            f"<font color='black'>{key.replace('_', ' ').title()}:</font> "
                            f"<font color='red'><b>{status}</b></font> - {direction}",
                            styles['CustomBodyText'])])
                    elif key == 'forward_head':
                        data.append([Paragraph(
                            f"<font color='black'>{key.replace('_', ' ').title()}:</font> "
                            f"<font color='{color}'><b>{status}</b></font> "
                            f"({measurement:.1f}°) - <font color='darkblue'>{direction}</font>",
                            styles['CustomBodyText'])])
                    elif key == 'knee_angle':
                        data.append([Paragraph(
                            f"Knee Angle: <font color='{color}'><b>{status}</b></font> "
                            f"({measurement:.1f}°) - <font color='{direction_color}'>{direction}</font>",
                            styles['CustomBodyText'])])
                    else:
                        data.append([Paragraph(
                            f"<font color='black'>{key.replace('_', ' ').title()}:</font> "
                            f"<font color='{color}'><b>{status}</b></font> "
                            f"({measurement:.1f}°) - <font color='{direction_color}'>{direction}</font>",
                            styles['CustomBodyText'])])
                else:
                    logging.warning(f"Unexpected format for key {key}: {value}")
                    data.append([Paragraph(f"{key.replace('_', ' ').title()}: {value}", styles['CustomBodyText'])])
            return Table(data, colWidths=[3.3 * inch], style=TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]))

        analysis_table = Table([[create_analysis_table("Anterior View Analysis", anterior_results),
                                 create_analysis_table("Lateral View Analysis", lateral_results)]],
                               colWidths=[3.5 * inch, 3.5 * inch],
                               hAlign='CENTER')
        story.append(analysis_table)
        story.append(Spacer(1, 6))

        # Key issues
        story.append(Paragraph("Key Postural Issues", styles['CustomHeading2']))

        def get_issue_order(issue):
            if 'pelvic tilt' in issue.lower():
                return 0
            severity_order = {
                'Severe': 1,
                'Mild': 2,
                'Normal': 3
            }
            for severity in ['Severe', 'Mild', 'Normal']:
                if severity in issue:
                    return severity_order.get(severity, 4)
            return 4

        all_issues = []
        for view, results in [("Anterior View", anterior_results), ("Lateral View", lateral_results)]:
            for key, value in results.items():
                if isinstance(value, tuple) and len(value) == 3:
                    status, measurement, direction = value
                    if 'Normal' not in status and 'Aligned' not in direction:
                        if key == 'pelvic_tilt' and view == "Lateral View":
                            issue = f"{key.replace('_', ' ').title()}: {status} - {direction}"
                        elif key == 'knee_angle':
                            issue = f"Knee Angle: {status} ({measurement:.1f}°) - {direction}"
                        else:
                            issue = f"{key.replace('_', ' ').title()}: {status} ({measurement:.1f}°) - {direction}"
                        all_issues.append((issue, measurement))
                else:
                    logging.warning(f"Unexpected format for key {key}: {value}")

        all_issues.sort(key=lambda x: get_issue_order(x[0]))

        for i, (issue, _) in enumerate(all_issues, 1):
            parts = issue.split(':')
            key = parts[0].strip()
            value = ':'.join(parts[1:]).strip()

            if 'Severe' in value:
                color = 'red'
                bold = 'bold'
            elif 'Mild' in value:
                color = 'darkorange'
                bold = 'bold'
            else:
                color = 'black'
                bold = 'normal'

            direction = value.split('-')[-1].strip()
            direction_color = 'black'
            if 'Left' in direction or 'Varus' in direction or 'Externally' in direction:
                direction_color = 'darkblue'
            elif 'Right' in direction or 'Valgus' in direction or 'Internally' in direction:
                direction_color = 'darkred'

            if 'Knee Angle' in key:
                story.append(Paragraph(
                    f"{i}. <font color='black'>{key}:</font> "
                    f"<font color='{color}'><b>{value.split('-')[0].strip()}</b></font> - "
                    f"<font color='{direction_color}'>{direction}</font>",
                    styles['CustomBodyText']))
            elif 'pelvic tilt' in key.lower():
                story.append(Paragraph(
                    f"{i}. <font color='black'>{key}:</font> "
                    f"<font color='{color}'><b>{value.split('-')[0].strip()}</b></font> - "
                    f"<font color='{direction_color}'>{direction}</font>",
                    styles['CustomBodyText']))
            else:
                story.append(Paragraph(
                    f"{i}. <font color='black'>{key}:</font> "
                    f"<font color='{color}'><b>{value.split('-')[0].strip()}</b></font> - "
                    f"<font color='{direction_color}'>{direction}</font>",
                    styles['CustomBodyText']))

        if not all_issues:
            story.append(Paragraph("No significant postural issues detected.", styles['CustomBodyText']))
        story.append(Spacer(1, 6))

        # Recommended exercises
        story.append(Paragraph("Recommended Exercises", styles['CustomHeading2']))

        # Add exercises for detected issues
        for i, (issue, _) in enumerate(all_issues):
            issue_title = issue.split(':')[0].strip()

            exercise_title = f"{chr(65 + i)}. {key_to_title(issue_title)}"
            story.append(Paragraph(exercise_title, styles['CustomHeading3']))

            issue_exercises = INTERVENTION_CONTENT.get(issue_title, [])

            for idx, exercise in enumerate(issue_exercises, start=1):
                story.append(Paragraph(f"{exercise}", styles['CustomBodyText']))

                # Fetch exercise image from external URL
                # Assuming the images are hosted at https://your-cdn.com/exercises/{Issue_Name}/{idx}.jpg
                # Replace 'your-cdn.com' with your actual CDN or storage URL
                sanitized_issue = issue_title.replace(' ', '_')
                exercise_image_url = f"https://your-cdn.com/exercises/{sanitized_issue}/{idx}.jpg"  # Replace with actual URLs
                try:
                    response = requests.get(exercise_image_url)
                    response.raise_for_status()
                    exercise_image = PILImage.open(io.BytesIO(response.content))
                    exercise_byte_arr = io.BytesIO()
                    exercise_image.save(exercise_byte_arr, format='JPEG')
                    exercise_byte_arr.seek(0)
                    rl_image = RLImage(exercise_byte_arr, width=3 * cm, height=3 * cm, kind='proportional')
                    story.append(rl_image)
                    logging.info(f"Fetched exercise image: {exercise_image_url}")
                except Exception as e:
                    story.append(Paragraph("Exercise image not found.", styles['CustomBodyText']))
                    logging.warning(f"Image not found for {issue_title}: {exercise_image_url}")

            if not issue_exercises:
                story.append(Paragraph("No specific exercises found for this issue.", styles['CustomBodyText']))
                logging.warning(f"No exercises found for {issue_title}")

            story.append(Spacer(1, 6))

        # General recommendations
        story.append(Paragraph("General Recommendations", styles['CustomHeading2']))
        general_recs = INTERVENTION_CONTENT.get('General Recommendations', [])
        if general_recs:
            for rec in general_recs:
                story.append(Paragraph(f"{rec}", styles['CustomBodyText']))
        else:
            story.append(Paragraph("No general recommendations found.", styles['CustomBodyText']))

        # Build the PDF with custom background and frames
        def add_background_and_frames(canvas_obj, doc_obj):
            canvas_obj.saveState()
            canvas_obj.setFillColor(background_color)
            canvas_obj.rect(0, 0, doc_obj.pagesize[0], doc_obj.pagesize[1], fill=True, stroke=False)
            canvas_obj.setFillColor(frame_color)
            canvas_obj.roundRect(0.25 * inch, 0.25 * inch, doc_obj.pagesize[0] - 0.5 * inch,
                                 doc_obj.pagesize[1] - 0.5 * inch, 10, fill=True, stroke=False)
            canvas_obj.setFont('Helvetica', 9)
            canvas_obj.drawRightString(7.5 * inch, 0.75 * inch, f"Page {doc_obj.page}")
            canvas_obj.restoreState()

        doc.build(story, onFirstPage=add_background_and_frames, onLaterPages=add_background_and_frames)
        logging.info(f"Report generated: {report_path}")

def key_to_title(key):
    return ' '.join(word.capitalize() for word in key.replace('-', '_').split('_'))

# =============================
# Flask Routes
# =============================

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
            try:
                # Secure filenames
                anterior_filename = secure_filename(anterior_file.filename)
                lateral_filename = secure_filename(lateral_file.filename)

                # Save files to upload directory
                anterior_path = os.path.join(app.config['UPLOAD_PATH'], anterior_filename)
                lateral_path = os.path.join(app.config['UPLOAD_PATH'], lateral_filename)

                anterior_file.save(anterior_path)
                lateral_file.save(lateral_path)

                logging.info(f"Uploaded anterior image: {anterior_path}")
                logging.info(f"Uploaded lateral image: {lateral_path}")

                # Process images
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

                # Create temporary directory for report
                with tempfile.TemporaryDirectory() as tmpdirname:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    annotated_anterior_path = os.path.join(tmpdirname, f'annotated_anterior_{timestamp}.jpg')
                    annotated_lateral_path = os.path.join(tmpdirname, f'annotated_lateral_{timestamp}.jpg')

                    cv2.imwrite(annotated_anterior_path, annotated_anterior)
                    cv2.imwrite(annotated_lateral_path, annotated_lateral)

                    logging.info(f"Annotated anterior image saved: {annotated_anterior_path}")
                    logging.info(f"Annotated lateral image saved: {annotated_lateral_path}")

                    # Generate report in temporary directory
                    report_path = os.path.join(tmpdirname, f'posture_analysis_report_{timestamp}.pdf')
                    generate_report(anterior_results, lateral_results, annotated_anterior_path, annotated_lateral_path, report_path)

                    # Send the PDF as a downloadable file
                    return send_file(report_path, as_attachment=True, attachment_filename=f'posture_analysis_report_{timestamp}.pdf')

            except Exception as e:
                logging.error(f"An error occurred during processing: {str(e)}")
                flash('An error occurred during processing.')
                return redirect(request.url)
        else:
            flash('Allowed image types are -> png, jpg, jpeg')
            return redirect(request.url)

    return render_template('index.html')

# =============================
# Main Entry
# =============================

if __name__ == "__main__":
    # For local development only. Heroku uses the Procfile to run the app.
    app.run(debug=True)