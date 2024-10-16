import os
import uuid
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from datetime import datetime
import logging

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'Output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
LOGO_FILENAME = "logo.jpg"

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
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key') # Replace with a secure secret key

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
        asis = (asis[0], asis[1] - int(0.1 * abs(right_hip[1] - left_hip[1]))) # Adjust Y coordinate slightly upward
        keypoints['asis'] = asis

    return keypoints

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # Clamp to avoid numerical issues
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
    if deviation[0] > 0: # Assuming positive x is anterior
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
            direction = '右偏' if head_angle > 0 else '左偏' # Translated to Traditional Chinese
            results['head_tilt'] = ('嚴重', head_tilt, direction)
        elif 2 <= head_tilt <= 3:
            direction = '右偏' if head_angle > 0 else '左偏'
            results['head_tilt'] = ('輕微', head_tilt, direction)
        else:
            results['head_tilt'] = ('正常', head_tilt, '居中')
    else:
        results['head_tilt'] = ('未檢測到', 0, '無法分析')

    # Shoulder Horizontal Tilt
    if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
        left_shoulder = np.array(keypoints['left_shoulder'])
        right_shoulder = np.array(keypoints['right_shoulder'])
        shoulder_angle = np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0]))
        shoulder_tilt = 180 - abs(shoulder_angle)
        if shoulder_tilt > 3:
            direction = '右偏' if shoulder_angle > 0 else '左偏'
            results['shoulder_tilt'] = ('嚴重', shoulder_tilt, direction)
        elif 2 <= shoulder_tilt <= 3:
            direction = '右偏' if shoulder_angle > 0 else '左偏'
            results['shoulder_tilt'] = ('輕微', shoulder_tilt, direction)
        else:
            results['shoulder_tilt'] = ('正常', shoulder_tilt, '水平')
    else:
        results['shoulder_tilt'] = ('未檢測到', 0, '無法分析')

    # Pelvic Horizontal Tilt
    if 'left_hip' in keypoints and 'right_hip' in keypoints:
        left_hip = np.array(keypoints['left_hip'])
        right_hip = np.array(keypoints['right_hip'])
        pelvic_angle = np.degrees(np.arctan2(right_hip[1] - left_hip[1], right_hip[0] - left_hip[0]))
        pelvic_tilt = 180 - abs(pelvic_angle)
        if pelvic_tilt > 3:
            direction = '右偏' if pelvic_angle > 0 else '左偏'
            results['pelvic_tilt'] = ('嚴重', pelvic_tilt, direction)
        elif 2 <= pelvic_tilt <= 3:
            direction = '右偏' if pelvic_angle > 0 else '左偏'
            results['pelvic_tilt'] = ('輕微', pelvic_tilt, direction)
        else:
            results['pelvic_tilt'] = ('正常', pelvic_tilt, '水平')
    else:
        results['pelvic_tilt'] = ('未檢測到', 0, '無法分析')

    # Knee Valgus/Varus
    for side in ['left', 'right']:
        if f'{side}_hip' in keypoints and f'{side}_knee' in keypoints and f'{side}_ankle' in keypoints:
            hip = np.array(keypoints[f'{side}_hip'])
            knee = np.array(keypoints[f'{side}_knee'])
            ankle = np.array(keypoints[f'{side}_ankle'])

            knee_deviation_angle, deviation_direction = calculate_knee_deviation(hip, knee, ankle)

            # Determine severity based on the deviation angle
            if deviation_direction == 'Flexed':
                if knee_deviation_angle > 10:
                    status = '嚴重'
                elif 5 < knee_deviation_angle <= 10:
                    status = '輕微'
                else:
                    status = '正常'
            elif deviation_direction == 'Hyperextended':
                if knee_deviation_angle > 10:
                    status = '嚴重'
                elif 5 < knee_deviation_angle <= 10:
                    status = '輕微'
                else:
                    status = '正常'
            else:
                status = '正常'

            # Format the result
            key = f'{side}_knee'
            results[key] = (status, knee_deviation_angle, deviation_direction)
        else:
            key = f'{side}_knee'
            results[key] = ('未檢測到', 0, '無法分析')

    # Feet Rotation
    for side in ['left', 'right']:
        if f'{side}_ankle' in keypoints and f'{side}_foot_index' in keypoints:
            ankle = np.array(keypoints[f'{side}_ankle'])
            toe = np.array(keypoints[f'{side}_foot_index'])
            foot_angle = np.degrees(np.arctan2(toe[0] - ankle[0], ankle[1] - toe[1]))
            foot_angle = 180 - abs(foot_angle)

            if foot_angle > 30:
                rotation = '嚴重外旋'
                status = '嚴重'
            elif 18 < foot_angle <= 30:
                rotation = '輕微外旋'
                status = '輕微'
            elif 5 <= foot_angle <= 18:
                rotation = '正常'
                status = '正常'
            elif 0 <= foot_angle < 5:
                rotation = '輕微內旋'
                status = '輕微'
            else:
                rotation = '嚴重內旋'
                status = '嚴重'

            key = f'{side}_foot_rotation'
            results[key] = (status, foot_angle, rotation)
        else:
            key = f'{side}_foot_rotation'
            results[key] = ('未檢測到', 0, '無法分析')

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
        forward_head_distance = (ear[0] - shoulder[0]) / width * 100 # Convert to percentage of image width
        if forward_head_distance > 5: # Threshold (adjust as needed)
            results['forward_head'] = ('嚴重', forward_head_distance, '前傾')
        elif forward_head_distance > 2:
            results['forward_head'] = ('輕微', forward_head_distance, '前傾')
        else:
            results['forward_head'] = ('正常', forward_head_distance, '居中')
    else:
        results['forward_head'] = ('未檢測到', 0, '無法分析')

    # Round Shoulders
    if 'right_shoulder' in keypoints and 'right_hip' in keypoints:
        shoulder = np.array(keypoints['right_shoulder'])
        hip = np.array(keypoints['right_hip'])
        shoulder_angle = np.degrees(np.arctan2(shoulder[0] - hip[0], hip[1] - shoulder[1]))
        if shoulder_angle > 30:
            results['round_shoulders'] = ('嚴重', shoulder_angle, '圓肩')
        elif shoulder_angle > 20:
            results['round_shoulders'] = ('輕微', shoulder_angle, '圓肩')
        else:
            results['round_shoulders'] = ('正常', shoulder_angle, '居中')
    else:
        results['round_shoulders'] = ('未檢測到', 0, '無法分析')

    # Pelvic Tilt
    if 'right_hip' in keypoints and 'right_knee' in keypoints:
        hip = keypoints['right_hip']
        knee = keypoints['right_knee']
        ankle = keypoints.get('right_ankle', hip) # Use hip if ankle not detected to avoid errors

        pelvic_angle = np.degrees(np.arctan2(knee[0] - hip[0], hip[1] - knee[1]))

        if pelvic_angle > 0:
            results['pelvic_tilt'] = ('後傾', pelvic_angle, '傾斜')
        elif pelvic_angle < 0:
            results['pelvic_tilt'] = ('前傾', abs(pelvic_angle), '傾斜')
        else:
            results['pelvic_tilt'] = ('正常', pelvic_angle, '中立')
    else:
        results['pelvic_tilt'] = ('未檢測到', 0, '無法分析')

    # Knee Flexion/Hyperextension (Right Lateral View)
    if 'right_hip' in keypoints and 'right_knee' in keypoints and 'right_ankle' in keypoints:
        hip = keypoints['right_hip']
        knee = keypoints['right_knee']
        ankle = keypoints['right_ankle']
        deviation_angle, direction = calculate_knee_deviation(hip, knee, ankle)

        # Determine severity based on the deviation angle
        if direction == 'Flexed':
            if deviation_angle > 10:
                status = '嚴重'
            elif 5 < deviation_angle <= 10:
                status = '輕微'
            else:
                status = '正常'
        elif direction == 'Hyperextended':
            if deviation_angle > 10:
                status = '嚴重'
            elif 5 < deviation_angle <= 10:
                status = '輕微'
            else:
                status = '正常'

        # Format the result
        results['knee_angle'] = (status, deviation_angle, direction)
    else:
        results['knee_angle'] = ('未檢測到', 0, '無法分析')

    return results

def draw_landmarks_and_angles(image, keypoints, view, analysis_results):
    annotated_image = image.copy()
    height, width, _ = annotated_image.shape

    # Draw vertical line for perfect alignment
    cv2.line(annotated_image, (width // 2, 0), (width // 2, height), (255, 0, 0), 2) # Blue vertical line

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
        color = (0, 255, 0) # Default green
        if analysis_key and analysis_key in analysis_results:
            status = analysis_results[analysis_key][0]
            if '嚴重' in status:
                color = (0, 0, 255) # Red if severe
            elif '輕微' in status:
                color = (0, 165, 255) # Orange if mild
        draw_line(keypoints[start], keypoints[end], color)

    if view == 'anterior':
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints and 'left_hip' in keypoints and 'right_hip' in keypoints:
            mid_shoulder = tuple(((np.array(keypoints['left_shoulder']) + np.array(keypoints['right_shoulder'])) / 2).astype(int))
            mid_hip = tuple(((np.array(keypoints['left_hip']) + np.array(keypoints['right_hip'])) / 2).astype(int))
            if 'left_eye' in keypoints and 'right_eye' in keypoints:
                mid_eye = tuple(((np.array(keypoints['left_eye']) + np.array(keypoints['right_eye'])) / 2).astype(int))
                draw_line(mid_eye, mid_shoulder, (0, 255, 0))
            draw_line(mid_shoulder, mid_hip, (0, 255, 0))

    return annotated_image

def generate_report(anterior_results, lateral_results, anterior_image_path, lateral_image_path):
    output_dir = 'Output'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"posture_analysis_report_{timestamp}.pdf")
    doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch,
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
    background_color = colors.Color(1, 0.9, 0.8) # Light orange
    frame_color = colors.white

    # Add logo
    logo_path = os.path.join('static', "logo.jpg")
    if os.path.exists(logo_path):
        logo = ReportImage(logo_path, width=6 * inch, height=0.5 * inch, kind='proportional')
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
    story.append(Table([[ReportImage(anterior_image_path, width=img_width, height=img_height, kind='proportional'),
                         ReportImage(lateral_image_path, width=img_width, height=img_height, kind='proportional')]],
                       colWidths=[3.5 * inch, 3.5 * inch],
                       hAlign='CENTER'))
    story.append(Spacer(1, 6))

    # Analysis results
    def create_analysis_table(title, results):
        data = [[Paragraph(title, styles['CustomHeading2'])]]
        for key, value in results.items():
            if isinstance(value, tuple) and len(value) == 3:
                status, measurement, direction = value
                if '嚴重' in status:
                    color = 'red'
                    bold = 'bold'
                elif '輕微' in status:
                    color = 'darkorange'
                    bold = 'bold'
                else:
                    color = 'black'
                    bold = 'normal'

                direction_color = 'black'
                if '右偏' in direction or '彎曲' in direction or '外旋' in direction:
                    direction_color = 'darkblue'
                elif '左偏' in direction or '內旋' in direction or '過度伸展' in direction:
                    direction_color = 'darkred'

                if key == 'pelvic_tilt' and title == "側視圖分析":
                    data.append([Paragraph(
                        f"<font color='black'>{key.replace('_', ' ').title()}:</font> <font color='red'><b>{status}</b></font> - {direction}",
                        styles['CustomBodyText'])])
                elif key == 'forward_head':
                    data.append([Paragraph(
                        f"<font color='black'>{key.replace('_', ' ').title()}:</font> <font color='{color}'><b>{status}</b></font> ({measurement:.1f}%) - <font color='darkblue'>{direction}</font>",
                        styles['CustomBodyText'])])
                elif key == 'knee_angle':
                    data.append([Paragraph(
                        f"Knee Angle: <font color='{color}'><b>{status}</b></font> ({measurement:.1f}°) - <font color='{direction_color}'>{direction}</font>",
                        styles['CustomBodyText'])])
                else:
                    data.append([Paragraph(
                        f"<font color='black'>{key.replace('_', ' ').title()}:</font> <font color='{color}'><b>{status}</b></font> ({measurement:.1f}°) - <font color='{direction_color}'>{direction}</font>",
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

    analysis_table = Table([[create_analysis_table("正面視圖分析", anterior_results),
                             create_analysis_table("側視圖分析", lateral_results)]],
                           colWidths=[3.5 * inch, 3.5 * inch],
                           hAlign='CENTER')
    story.append(analysis_table)
    story.append(Spacer(1, 6))

    # Key issues
    story.append(Paragraph("主要姿態問題", styles['CustomHeading2']))

    def get_issue_order(issue):
        if 'pelvic tilt' in issue.lower() or '骨盆傾斜' in issue.lower():
            return 0
        severity_order = {
            '嚴重': 1,
            '輕微': 2,
            '正常': 3
        }
        for severity in ['嚴重', '輕微', '正常']:
            if severity in issue:
                return severity_order.get(severity, 4)
        return 4

    all_issues = []
    for view, results in [("正面視圖", anterior_results), ("側視圖", lateral_results)]:
        for key, value in results.items():
            if isinstance(value, tuple) and len(value) == 3:
                status, measurement, direction = value
                if '正常' not in status and '居中' not in direction and 'Neutral' not in direction:
                    if key == 'pelvic_tilt' and view == "側視圖":
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

        if '嚴重' in value:
            color = 'red'
            bold = 'bold'
        elif '輕微' in value:
            color = 'darkorange'
            bold = 'bold'
        else:
            color = 'black'
            bold = 'normal'

        direction = value.split('-')[-1].strip()
        direction_color = 'black'
        if '右偏' in direction or '彎曲' in direction or '外旋' in direction:
            direction_color = 'darkblue'
        elif '左偏' in direction or '內旋' in direction or '過度伸展' in direction:
            direction_color = 'darkred'

        if 'Knee Angle' in key:
            story.append(Paragraph(
                f"{i}. <font color='black'>{key}:</font> <font color='{color}'><b>{value.split('-')[0].strip()}</b></font> - <font color='{direction_color}'>{direction}</font>",
                styles['CustomBodyText']))
        elif 'pelvic tilt' in key.lower() or '骨盆傾斜' in key.lower():
            story.append(Paragraph(
                f"{i}. <font color='black'>{key}:</font> <font color='{color}'><b>{value.split('-')[0].strip()}</b></font> - <font color='{direction_color}'>{direction}</font>",
                styles['CustomBodyText']))
        else:
            story.append(Paragraph(
                f"{i}. <font color='black'>{key}:</font> <font color='{color}'><b>{value.split('-')[0].strip()}</b></font> - <font color='{direction_color}'>{direction}</font>",
                styles['CustomBodyText']))

    if not all_issues:
        story.append(Paragraph("沒有檢測到顯著的姿態問題。", styles['CustomBodyText']))
    story.append(Spacer(1, 6))

    # Recommended exercises
    story.append(Paragraph("推薦運動", styles['CustomHeading2']))

    # Read exercises from the Intervention.txt file
    exercises = {}
    try:
        with open('Intervention.txt', 'r', encoding='utf-8') as file:
            current_issue = None
            for line in file:
                line = line.strip()
                if line.endswith(':'):
                    current_issue = line[:-1].strip()
                    exercises[current_issue] = []
                elif line.startswith(('1.', '2.', '3.', '4.', '5.')) and current_issue:
                    exercises[current_issue].append(line)
    except FileNotFoundError:
        logging.error("Intervention.txt file not found.")
    except Exception as e:
        logging.error(f"Error reading Intervention.txt: {str(e)}")

    # Add exercises for detected issues
    for i, (issue, _) in enumerate(all_issues):
        issue_title = issue.split(':')[0].strip()

        exercise_title = f"{chr(65 + i)}. {key_to_title(issue_title)}"
        story.append(Paragraph(exercise_title, styles['CustomHeading3']))

        issue_exercises = exercises.get(issue_title, [])

        for idx, exercise in enumerate(issue_exercises, start=1):
            story.append(Paragraph(f"{exercise}", styles['CustomBodyText']))

            image_path = os.path.join('Exercise', issue_title, f"{idx}.jpg")
            if os.path.exists(image_path):
                story.append(ReportImage(image_path, width=3 * cm, height=3 * cm, kind='proportional'))
                logging.info(f"Found image for {issue_title}: {image_path}")
            else:
                story.append(Paragraph("運動圖片未找到。", styles['CustomBodyText']))
                logging.warning(f"Image not found for {issue_title}: {image_path}")

        if not issue_exercises:
            story.append(Paragraph("未找到針對此問題的具體運動。", styles['CustomBodyText']))
            logging.warning(f"No exercises found for {issue_title}")

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
        canvas_obj.drawRightString(7.5 * inch, 0.75 * inch, f"Page {doc_obj.page}")
        canvas_obj.restoreState()

    doc.build(story, onFirstPage=add_background_and_frames, onLaterPages=add_background_and_frames)
    logging.info(f"Report generated: {output_path}")


def key_to_title(key):
    # Translate keys to Traditional Chinese titles
    translation = {
        'pelvic tilt': '骨盆傾斜',
        'knee angle': '膝蓋角度',
        'forward_head': '前傾頭部',
        'head tilt': '頭部傾斜',
        'shoulder tilt': '肩膀傾斜',
        'round shoulders': '圓肩',
        'feet rotation': '腳部旋轉'
        # Add more translations as needed
    }
    return translation.get(key.lower(), key.replace('-', ' ').title())


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the files
        if 'anterior_image' not in request.files or 'lateral_image' not in request.files:
            flash('未上傳任何文件。')
            return redirect(request.url)

        anterior_file = request.files['anterior_image']
        lateral_file = request.files['lateral_image']

        if anterior_file.filename == '' or lateral_file.filename == '':
            flash('未選擇文件。')
            return redirect(request.url)

        if anterior_file and allowed_file(anterior_file.filename) and lateral_file and allowed_file(
                lateral_file.filename):
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
                    flash("無法讀取一個或兩個圖像。")
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
                generate_report(anterior_results, lateral_results, annotated_anterior_path, annotated_lateral_path)

                report_filename = f'posture_analysis_report_{timestamp}.pdf'
                report_path = os.path.join(app.config['OUTPUT_FOLDER'], report_filename)

                logging.info(f"Report generated: {report_path}")

                # Cleanup uploaded and annotated images
                os.remove(anterior_path)
                os.remove(lateral_path)
                os.remove(annotated_anterior_path)
                os.remove(annotated_lateral_path)

                return redirect(url_for('download_file', filename=report_filename))
            except Exception as e:
                logging.error(f"An error occurred during processing: {str(e)}")
                flash("處理期間發生錯誤。請稍後再試。")
                return redirect(request.url)
        else:
            flash('允許上傳的文件類型為 png, jpg, jpeg。')
            return redirect(request.url)

    return render_template('index.html', logo=LOGO_FILENAME)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


if __name__ == "__main__":
    # Run the application on all interfaces
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))