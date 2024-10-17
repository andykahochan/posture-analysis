import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from datetime import datetime
import logging
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import tempfile
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload size

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Helper Functions

def compress_image(input_image, max_size_kb=30):
    """
    Compress the image to be under max_size_kb.
    Returns a BytesIO object of the compressed image.
    """
    try:
        img = Image.open(input_image)
        img_format = img.format  # Keep the original format

        # Initialize variables
        quality = 85
        resized = False

        img_io = BytesIO()

        while True:
            img_io.seek(0)
            if img_format == 'JPEG':
                img.save(img_io, format=img_format, quality=quality, optimize=True)
            else:
                img.save(img_io, format=img_format, optimize=True)
            size = img_io.tell()

            if size <= max_size_kb * 1024 or quality <= 20:
                break

            if img_format == 'JPEG':
                quality -= 5
            elif not resized:
                # Resize the image to reduce size
                img = img.resize((int(img.width * 0.9), int(img.height * 0.9)), Image.LANCZOS)
                resized = True
            else:
                break  # Cannot compress further

        img_io.seek(0)
        return img_io
    except Exception as e:
        logging.error(f"Error compressing image: {str(e)}")
        return None

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
            key = f'Head Tilt - {direction}'
            results[key] = ('Severe', head_tilt, direction)
        elif 2 <= head_tilt <= 3:
            direction = 'Right' if head_angle > 0 else 'Left'
            key = f'Head Tilt - {direction}'
            results[key] = ('Mild', head_tilt, direction)
        else:
            key = 'Head Tilt - Centered'
            results[key] = ('Normal', head_tilt, 'Centered')
    else:
        results['Head Tilt - Not detected'] = ('Not detected', 0, 'Unable to analyze')

    # Shoulder Horizontal Tilt
    if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
        left_shoulder = np.array(keypoints['left_shoulder'])
        right_shoulder = np.array(keypoints['right_shoulder'])
        shoulder_angle = np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0]))
        shoulder_tilt = 180 - abs(shoulder_angle)
        if shoulder_tilt > 3:
            direction = 'Right' if shoulder_angle > 0 else 'Left'
            key = f'Shoulder Tilt - {direction}'
            results[key] = ('Severe', shoulder_tilt, direction)
        elif 2 <= shoulder_tilt <= 3:
            direction = 'Right' if shoulder_angle > 0 else 'Left'
            key = f'Shoulder Tilt - {direction}'
            results[key] = ('Mild', shoulder_tilt, direction)
        else:
            key = 'Shoulder Tilt - Level'
            results[key] = ('Normal', shoulder_tilt, 'Level')
    else:
        results['Shoulder Tilt - Not detected'] = ('Not detected', 0, 'Unable to analyze')

    # Pelvic Horizontal Tilt
    if 'left_hip' in keypoints and 'right_hip' in keypoints:
        left_hip = np.array(keypoints['left_hip'])
        right_hip = np.array(keypoints['right_hip'])
        pelvic_angle = np.degrees(np.arctan2(right_hip[1] - left_hip[1], right_hip[0] - left_hip[0]))
        pelvic_tilt = 180 - abs(pelvic_angle)
        if pelvic_tilt > 3:
            direction = 'Right' if pelvic_angle > 0 else 'Left'
            key = f'Pelvic Tilt - {direction}'
            results[key] = ('Severe', pelvic_tilt, direction)
        elif 2 <= pelvic_tilt <= 3:
            direction = 'Right' if pelvic_angle > 0 else 'Left'
            key = f'Pelvic Tilt - {direction}'
            results[key] = ('Mild', pelvic_tilt, direction)
        else:
            key = 'Pelvic Tilt - Level'
            results[key] = ('Normal', pelvic_tilt, 'Level')
    else:
        results['Pelvic Tilt - Not detected'] = ('Not detected', 0, 'Unable to analyze')

    # Knee Valgus/Varus
    for side in ['Left', 'Right']:
        side_lower = side.lower()
        if f'{side_lower}_hip' in keypoints and f'{side_lower}_knee' in keypoints and f'{side_lower}_ankle' in keypoints:
            hip = np.array(keypoints[f'{side_lower}_hip'])
            knee = np.array(keypoints[f'{side_lower}_knee'])
            ankle = np.array(keypoints[f'{side_lower}_ankle'])

            hip_ankle_vector = ankle - hip
            hip_knee_vector = knee - hip
            cross_product = np.cross(hip_ankle_vector[:2], hip_knee_vector[:2])
            angle = abs(calculate_angle(hip, knee, ankle) - 180)
            knee_deviation = 'Valgus' if (cross_product > 0 and side == 'Left') or (cross_product < 0 and side == 'Right') else 'Varus'

            key = f'Knee - {knee_deviation}'
            if angle > 15:
                results[key] = ('Severe', angle, knee_deviation)
            elif 5 <= angle <= 15:
                results[key] = ('Mild', angle, knee_deviation)
            else:
                results[key] = ('Normal', angle, 'Aligned')
        else:
            key = f'Knee - {side}'
            results[key] = ('Not detected', 0, 'Unable to analyze')

    # Feet Rotation
    for side in ['Left', 'Right']:
        side_lower = side.lower()
        if f'{side_lower}_ankle' in keypoints and f'{side_lower}_foot_index' in keypoints:
            ankle = np.array(keypoints[f'{side_lower}_ankle'])
            toe = np.array(keypoints[f'{side_lower}_foot_index'])
            foot_angle = np.degrees(np.arctan2(toe[0] - ankle[0], ankle[1] - toe[1]))
            foot_angle = 180 - abs(foot_angle)

            if foot_angle > 30:
                condition = 'Externally Rotated'
                key = f'Foot Rotation - {condition}'
                results[key] = ('Severe', foot_angle, condition)
            elif 18 < foot_angle <= 30:
                condition = 'Externally Rotated'
                key = f'Foot Rotation - {condition}'
                results[key] = ('Mild', foot_angle, condition)
            elif 5 <= foot_angle <= 18:
                condition = 'Aligned'
                key = f'Foot Rotation - {condition}'
                results[key] = ('Normal', foot_angle, condition)
            elif 0 <= foot_angle < 5:
                condition = 'Internally Rotated'
                key = f'Foot Rotation - {condition}'
                results[key] = ('Mild', foot_angle, condition)
            else:
                condition = 'Internally Rotated'
                key = f'Foot Rotation - {condition}'
                results[key] = ('Severe', abs(foot_angle), condition)
        else:
            key = f'Foot Rotation - {side}'
            results[key] = ('Not detected', 0, 'Unable to analyze')

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
            key = 'Forward Head'
            results[key] = ('Severe', forward_head_distance, 'Forward')
        elif forward_head_distance > 2:
            key = 'Forward Head'
            results[key] = ('Mild', forward_head_distance, 'Forward')
        else:
            key = 'Forward Head'
            results[key] = ('Normal', forward_head_distance, 'Aligned')
    else:
        results['Forward Head'] = ('Not detected', 0, 'Unable to analyze')

    # Round Shoulders
    if 'right_shoulder' in keypoints and 'right_hip' in keypoints:
        shoulder = np.array(keypoints['right_shoulder'])
        hip = np.array(keypoints['right_hip'])
        shoulder_angle = np.degrees(np.arctan2(shoulder[0] - hip[0], hip[1] - shoulder[1]))
        if shoulder_angle > 30:
            key = 'Round Shoulders'
            results[key] = ('Severe', shoulder_angle, 'Rounded')
        elif shoulder_angle > 20:
            key = 'Round Shoulders'
            results[key] = ('Mild', shoulder_angle, 'Rounded')
        else:
            key = 'Round Shoulders'
            results[key] = ('Normal', shoulder_angle, 'Aligned')
    else:
        results['Round Shoulders'] = ('Not detected', 0, 'Unable to analyze')

    # Pelvic Tilt
    if 'right_hip' in keypoints and 'right_knee' in keypoints:
        hip = keypoints['right_hip']
        knee = keypoints['right_knee']
        ankle = keypoints.get('right_ankle', hip)  # Use hip if ankle not detected to avoid errors

        pelvic_angle = np.degrees(np.arctan2(knee[0] - hip[0], hip[1] - knee[1]))

        if pelvic_angle > 0:
            key = 'Pelvic Tilt - Posterior'
            results[key] = ('Posterior', pelvic_angle, 'Tilted')
        elif pelvic_angle < 0:
            key = 'Pelvic Tilt - Anterior'
            results[key] = ('Anterior', abs(pelvic_angle), 'Tilted')
        else:
            key = 'Pelvic Tilt - Neutral'
            results[key] = ('Normal', pelvic_angle, 'Neutral')
    else:
        results['Pelvic Tilt - Not detected'] = ('Not detected', 0, 'Unable to analyze')

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
        key = 'Knee Angle'
        results[key] = (status, deviation_angle, direction)
    else:
        results['Knee Angle'] = ('Not detected', 0, 'Unable to analyze')

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
            ('left_eye', 'right_eye', 'Head Tilt - Left', 'Head Tilt - Right'),
            ('left_shoulder', 'right_shoulder', 'Shoulder Tilt - Left', 'Shoulder Tilt - Right'),
            ('left_hip', 'right_hip', 'Pelvic Tilt - Left', 'Pelvic Tilt - Right'),
            ('left_shoulder', 'left_elbow', None, None),
            ('left_elbow', 'left_wrist', None, None),
            ('right_shoulder', 'right_elbow', None, None),
            ('right_elbow', 'right_wrist', None, None),
            ('left_shoulder', 'left_hip', None, None),
            ('right_shoulder', 'right_hip', None, None),
            ('left_hip', 'left_knee', 'Knee - Valgus', 'Knee - Varus'),
            ('right_hip', 'right_knee', 'Knee - Valgus', 'Knee - Varus'),
            ('left_knee', 'left_ankle', 'Knee - Valgus', 'Knee - Varus'),
            ('right_knee', 'right_ankle', 'Knee - Valgus', 'Knee - Varus'),
            ('left_ankle', 'left_foot_index', 'Foot Rotation - Externally Rotated', 'Foot Rotation - Internally Rotated'),
            ('right_ankle', 'right_foot_index', 'Foot Rotation - Externally Rotated', 'Foot Rotation - Internally Rotated')
        ]
    elif view == 'lateral':
        connections = [
            ('right_ear', 'right_shoulder', 'Forward Head', None),
            ('right_shoulder', 'right_hip', 'Round Shoulders', None),
            ('right_hip', 'right_knee', 'Pelvic Tilt - Anterior', 'Pelvic Tilt - Posterior'),
            ('right_knee', 'right_ankle', 'Knee Angle', None)
        ]

    # Determine if pelvic tilt is anterior or posterior in lateral view
    highlight_hip_knee = False
    if view == 'lateral':
        pelvic_tilt_anterior = analysis_results.get('Pelvic Tilt - Anterior')
        pelvic_tilt_posterior = analysis_results.get('Pelvic Tilt - Posterior')
        if (pelvic_tilt_anterior and pelvic_tilt_anterior[0] in ['Severe', 'Mild']) or \
           (pelvic_tilt_posterior and pelvic_tilt_posterior[0] in ['Severe', 'Mild']):
            highlight_hip_knee = True

    for connection in connections:
        start, end, key1, key2 = connection
        color = (0, 255, 0)  # Default green
        if key1 and key1 in analysis_results:
            status = analysis_results[key1][0]
            if 'Severe' in status:
                color = (0, 0, 255)  # Red if severe
            elif 'Mild' in status:
                color = (0, 165, 255)  # Orange if mild
        elif key2 and key2 in analysis_results:
            status = analysis_results[key2][0]
            if 'Severe' in status:
                color = (0, 0, 255)  # Red if severe
            elif 'Mild' in status:
                color = (0, 165, 255)  # Orange if mild

        # For lateral view, if 'Knee Angle' is abnormal, highlight in red
        if view == 'lateral' and highlight_hip_knee and (('Knee Angle' in key1) or ('Knee Angle' in key2)):
            color = (0, 0, 255)

        if start in keypoints and end in keypoints:
            draw_line(keypoints[start], keypoints[end], color)

    if view == 'anterior':
        if ('left_shoulder' in keypoints and 'right_shoulder' in keypoints and
            'left_hip' in keypoints and 'right_hip' in keypoints):
            mid_shoulder = tuple(((np.array(keypoints['left_shoulder']) + np.array(keypoints['right_shoulder'])) / 2).astype(int))
            mid_hip = tuple(((np.array(keypoints['left_hip']) + np.array(keypoints['right_hip'])) / 2).astype(int))
            if 'left_eye' in keypoints and 'right_eye' in keypoints:
                mid_eye = tuple(((np.array(keypoints['left_eye']) + np.array(keypoints['right_eye'])) / 2).astype(int))
                draw_line(mid_eye, mid_shoulder, (0, 255, 0))
            draw_line(mid_shoulder, mid_hip, (0, 255, 0))

    # For lateral view, if pelvic tilt is anterior or posterior, redraw the hip-knee line in red
    if view == 'lateral' and highlight_hip_knee:
        if 'right_hip' in keypoints and 'right_knee' in keypoints:
            hip = keypoints['right_hip']
            knee = keypoints['right_knee']
            draw_line(hip, knee, (0, 0, 255))  # Red color

    return annotated_image

def generate_report(anterior_results, lateral_results, anterior_image_path, lateral_image_path, logo_path):
    output_buffer = BytesIO()
    doc = SimpleDocTemplate(output_buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch,
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

    # Background and frame colors
    background_color = colors.Color(1, 0.9, 0.8)  # Light orange
    frame_color = colors.white

    # Add logo
    if os.path.exists(logo_path):
        logo = RLImage(logo_path, width=6 * inch, height=0.5 * inch, kind='proportional')
        story.append(logo)
        story.append(Spacer(1, 6))
    else:
        logging.warning(f"Logo file not found at path: {logo_path}")

    # Title
    story.append(Paragraph("姿勢分析報告", styles['CustomHeading1']))
    story.append(Spacer(1, 6))

    # Date
    story.append(Paragraph(f"日期: {datetime.now().strftime('%Y-%m-%d')}", styles['CustomBodyText']))
    story.append(Spacer(1, 6))

    # Images
    img_width = 3 * inch
    img_height = 4 * inch
    story.append(Table([[RLImage(anterior_image_path, width=img_width, height=img_height, kind='proportional'),
                         RLImage(lateral_image_path, width=img_width, height=img_height, kind='proportional')]],
                       colWidths=[3.5 * inch, 3.5 * inch],
                       hAlign='CENTER'))
    story.append(Spacer(1, 6))

    # Analysis results
    def create_analysis_table(title, results):
        data = [[Paragraph(title, styles['CustomHeading2'])]]
        for key, value in results.items():
            if isinstance(value, tuple) and len(value) == 3:
                status, measurement, direction = value
                condition = key  # e.g., 'Head Tilt - Left'
                if 'Not detected' in status:
                    content = f"{condition}: {status} - {direction}"
                else:
                    # Determine color based on status
                    if 'Severe' in status:
                        color = 'red'
                        bold = 'bold'
                    elif 'Mild' in status:
                        color = 'darkorange'
                        bold = 'bold'
                    else:
                        color = 'black'
                        bold = 'normal'

                    # Determine color for direction
                    direction_color = 'black'
                    if 'Left' in direction or 'Varus' in direction or 'Externally Rotated' in condition:
                        direction_color = 'darkblue'
                    elif 'Right' in direction or 'Valgus' in direction or 'Internally Rotated' in condition:
                        direction_color = 'darkred'
                    elif 'Anterior' in condition or 'Posterior' in condition:
                        direction_color = 'darkred'

                    if key == 'Knee Angle':
                        content = f"Knee Angle: <font color='{color}'><b>{status}</b></font> ({measurement:.1f}°) - <font color='{direction_color}'>{direction}</font>"
                    else:
                        content = f"<font color='black'>{condition}:</font> <font color='{color}'><b>{status}</b></font> ({measurement:.1f}°) - <font color='{direction_color}'>{direction}</font>"
                data.append([Paragraph(content, styles['CustomBodyText'])])
            else:
                logging.warning(f"Unexpected format for key {key}: {value}")
                data.append([Paragraph(f"{key.replace('_', ' ').title()}: {value}", styles['CustomBodyText'])])
        return Table(data, colWidths=[6.5 * inch], style=TableStyle([
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

    analysis_table = Table([[create_analysis_table("上視圖分析", anterior_results),
                             create_analysis_table("側視圖分析", lateral_results)]],
                           colWidths=[3.5 * inch, 3.5 * inch],
                           hAlign='CENTER')
    story.append(analysis_table)
    story.append(Spacer(1, 6))

    # Key issues
    story.append(Paragraph("主要姿勢問題", styles['CustomHeading2']))

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
    for view, results in [("上視圖", anterior_results), ("側視圖", lateral_results)]:
        for key, value in results.items():
            if isinstance(value, tuple) and len(value) == 3:
                status, measurement, direction = value
                if 'Normal' not in status and 'Aligned' not in direction and 'Neutral' not in status:
                    issue = key  # e.g., 'Head Tilt - Left'
                    all_issues.append((issue, measurement))
            else:
                logging.warning(f"Unexpected format for key {key}: {value}")

    all_issues.sort(key=lambda x: get_issue_order(x[0]))

    for i, (issue, _) in enumerate(all_issues, 1):
        # issue is like 'Head Tilt - Left'
        condition = issue  # Full condition name

        story.append(Paragraph(f"{i}. {condition}", styles['CustomBodyText']))

    if not all_issues:
        story.append(Paragraph("未檢測到顯著的姿勢問題。", styles['CustomBodyText']))
    story.append(Spacer(1, 6))

    # Recommended exercises
    story.append(Paragraph("推薦運動", styles['CustomHeading2']))

    # Read exercises from the Intervention.txt file
    exercises = {}
    try:
        intervention_path = os.path.join(os.getcwd(), 'Intervention.txt')
        with open(intervention_path, 'r', encoding='utf-8') as file:
            current_issue = None
            for line in file:
                line = line.strip()
                if line.endswith(':'):
                    current_issue = line[:-1].strip()
                    exercises[current_issue] = []
                elif line.startswith(('1.', '2.')) and current_issue:
                    exercises[current_issue].append(line)
    except FileNotFoundError:
        logging.error("Intervention.txt 文件未找到。")
    except Exception as e:
        logging.error(f"讀取 Intervention.txt 時出錯：{str(e)}")

    # Add exercises for detected issues
    for i, (issue, _) in enumerate(all_issues):
        exercise_title = f"{chr(65 + i)}. {key_to_title(issue)}"
        story.append(Paragraph(exercise_title, styles['CustomHeading3']))

        issue_exercises = exercises.get(issue, [])

        for idx, exercise in enumerate(issue_exercises, start=1):
            story.append(Paragraph(f"{exercise}", styles['CustomBodyText']))

            # Image path based on issue and exercise number
            image_path = os.path.join('Exercise', issue, f"{idx}.jpg")
            full_image_path = os.path.join(os.getcwd(), image_path)
            if os.path.exists(full_image_path):
                story.append(RLImage(full_image_path, width=3 * cm, height=3 * cm, kind='proportional'))
                logging.info(f"找到 {issue} 的圖片: {full_image_path}")
            else:
                story.append(Paragraph("未找到運動圖片。", styles['CustomBodyText']))
                logging.warning(f"未找到 {issue} 的圖片: {full_image_path}")

        if not issue_exercises:
            story.append(Paragraph("未找到針對此問題的具體運動。", styles['CustomBodyText']))
            logging.warning(f"未找到 {issue} 的運動。")

        story.append(Spacer(1, 6))

    # General recommendations
    story.append(Paragraph("一般建議", styles['CustomHeading2']))

    general_recs = exercises.get('General Recommendations', [])
    if general_recs:
        for rec in general_recs:
            story.append(Paragraph(f"{rec}", styles['CustomBodyText']))
    else:
        story.append(Paragraph("未找到一般建議。", styles['CustomBodyText']))

    # Build the PDF with custom background and frames
    def add_background_and_frames(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setFillColor(background_color)
        canvas_obj.rect(0, 0, doc_obj.pagesize[0], doc_obj.pagesize[1], fill=True, stroke=False)
        canvas_obj.setFillColor(frame_color)
        canvas_obj.roundRect(0.25 * inch, 0.25 * inch, doc_obj.pagesize[0] - 0.5 * inch,
                             doc_obj.pagesize[1] - 0.5 * inch, 10, fill=True, stroke=False)
        canvas_obj.setFont('Helvetica', 9)
        canvas_obj.drawRightString(7.5 * inch, 0.75 * inch, f"第 {doc_obj.page} 頁")
        canvas_obj.restoreState()

    doc.build(story, onFirstPage=add_background_and_frames, onLaterPages=add_background_and_frames)
    output_buffer.seek(0)
    return output_buffer

def key_to_title(key):
    return key  # 已經格式化為 'Head Tilt - Left' 等

# Route Definitions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if files are present
        if 'anterior_image' not in request.files or 'lateral_image' not in request.files:
            flash('請上傳兩張圖像（上視圖和側視圖）。')
            return redirect(request.url)

        anterior_file = request.files['anterior_image']
        lateral_file = request.files['lateral_image']

        if anterior_file.filename == '' or lateral_file.filename == '':
            flash('請上傳兩張圖像（上視圖和側視圖）。')
            return redirect(request.url)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Compress and save uploaded anterior image
                anterior_compressed = compress_image(anterior_file, max_size_kb=30)
                if anterior_compressed is None:
                    flash('上視圖圖像壓縮失敗。請嘗試使用較小的圖像。')
                    return redirect(request.url)

                anterior_path = os.path.join(temp_dir, 'anterior.jpg')
                with open(anterior_path, 'wb') as f:
                    f.write(anterior_compressed.read())

                # Compress and save uploaded lateral image
                lateral_compressed = compress_image(lateral_file, max_size_kb=30)
                if lateral_compressed is None:
                    flash('側視圖圖像壓縮失敗。請嘗試使用較小的圖像。')
                    return redirect(request.url)

                lateral_path = os.path.join(temp_dir, 'lateral.jpg')
                with open(lateral_path, 'wb') as f:
                    f.write(lateral_compressed.read())

                logging.info(f"壓縮後的上視圖圖像保存到 {anterior_path}")
                logging.info(f"壓縮後的側視圖圖像保存到 {lateral_path}")

                # Read images using OpenCV
                anterior_image = cv2.imread(anterior_path)
                lateral_image = cv2.imread(lateral_path)

                if anterior_image is None or lateral_image is None:
                    flash('錯誤：無法讀取上視圖或側視圖圖像。')
                    return redirect(request.url)

                # Detect keypoints
                anterior_keypoints = detect_keypoints(anterior_image)
                lateral_keypoints = detect_keypoints(lateral_image)

                # Analyze views
                anterior_results = analyze_anterior_view(anterior_keypoints, anterior_image.shape)
                lateral_results = analyze_lateral_view(lateral_keypoints, lateral_image.shape)

                # Draw landmarks and angles
                annotated_anterior = draw_landmarks_and_angles(
                    anterior_image,
                    anterior_keypoints or {},
                    'anterior',
                    anterior_results
                )
                annotated_lateral = draw_landmarks_and_angles(
                    lateral_image,
                    lateral_keypoints or {},
                    'lateral',
                    lateral_results
                )

                # Save annotated images to temp directory
                annotated_anterior_path = os.path.join(temp_dir, 'annotated_anterior.jpg')
                annotated_lateral_path = os.path.join(temp_dir, 'annotated_lateral.jpg')
                cv2.imwrite(annotated_anterior_path, annotated_anterior)
                cv2.imwrite(annotated_lateral_path, annotated_lateral)

                logging.info(f"註釋後的上視圖圖像保存到 {annotated_anterior_path}")
                logging.info(f"註釋後的側視圖圖像保存到 {annotated_lateral_path}")

                # Path to logo.jpg in static folder
                logo_path = os.path.join(os.getcwd(), 'static', 'logo.jpg')

                # Generate PDF report
                pdf_buffer = generate_report(
                    anterior_results,
                    lateral_results,
                    annotated_anterior_path,
                    annotated_lateral_path,
                    logo_path
                )

                logging.info("報告生成完成。")

                # Send the PDF as a downloadable file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"posture_analysis_report_{timestamp}.pdf"

                return send_file(
                    pdf_buffer,
                    as_attachment=True,
                    download_name=pdf_filename,
                    mimetype='application/pdf'
                )

        except Exception as e:
            logging.error(f"處理過程中發生錯誤：{str(e)}")
            logging.exception("詳細異常信息：")
            flash('處理您的圖像時發生錯誤。請確保圖片大小不超過10MB並重試。')
            return redirect(request.url)

    return render_template('index.html')

# Error handler for file size limit
@app.errorhandler(413)
def request_entity_too_large(error):
    flash('所上傳的檔案太大，請上傳小於10MB的圖像。')
    return redirect(request.url)

# Run the app (for local testing)
if __name__ == "__main__":
    app.run(debug=True)