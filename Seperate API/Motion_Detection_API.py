from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from ultralytics import YOLO
import yaml
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the 'uploads' directory exists
os.makedirs('uploads', exist_ok=True)

# Load class names from data.yaml
def load_class_names(yaml_file):
    with open(yaml_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded['names']

# LSTM model for motion detection
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Extract features from bounding boxes
def extract_features_from_boxes(boxes):
    features = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        features.append([center_x, center_y, width, height])
    return np.array(features)

# Check if required personnel are near the ICU patient
def check_proximity(boxes, class_ids, personnel_classes, icu_patient_class_id, threshold=50):
    icu_patient_boxes = [boxes[i] for i in range(len(class_ids)) if class_ids[i] == icu_patient_class_id]
    personnel_boxes = [boxes[i] for i in range(len(class_ids)) if class_ids[i] in personnel_classes]
    for icu_box in icu_patient_boxes:
        icu_x, icu_y, icu_w, icu_h = icu_box
        for p_box in personnel_boxes:
            p_x, p_y, p_w, p_h = p_box
            if abs(icu_x - p_x) < threshold and abs(icu_y - p_y) < threshold:
                return True
    return False

# Draw bounding boxes on the frame
def draw_bounding_boxes(frame, boxes, confidences, class_ids, classes):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Process video
def process_video(video_path):
    yaml_file = "data.yaml"
    model_weights = "best.pt"
    logger.info(f"Loading class names from: {yaml_file}")
    class_names = load_class_names(yaml_file)
    personnel_classes = [class_names.index(name) for name in ['Doctor', 'Nurse', 'Staff']]
    icu_patient_class_id = class_names.index('ICU_Patient')

    logger.info(f"Loading YOLO model with weights: {model_weights}")
    yolo_model = YOLO(model_weights)  # Load YOLOv8 model with custom weights

    cap = cv2.VideoCapture(video_path)
    lstm_model = create_lstm_model((10, 4))  # Assuming sequence length of 10 and 4 features per box

    sequence_length = 10
    feature_sequence = []
    frame_count = 0

    results_to_return = []
    frame_confidences = []

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video_path = os.path.join("uploads", "output_detected.mp4")

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Perform object detection
        results = yolo_model(frame)
        logger.info(f"Processed frame {frame_count} with YOLO")

        # Extract bounding boxes, confidences, and class IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Assuming the first element contains the bounding boxes
        confidences = results[0].boxes.conf.cpu().numpy()  # Assuming the first element contains the confidences
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Assuming the first element contains the class IDs

        draw_bounding_boxes(frame, boxes, confidences, class_ids, class_names)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

        features = extract_features_from_boxes(boxes)
        if features.shape[0] < 10:  # Ensure the sequence length is consistent
            padding = np.zeros((10 - features.shape[0], 4))
            features = np.concatenate((features, padding), axis=0)
        else:
            features = features[:10, :]

        # Append each feature vector individually to the sequence
        for feature in features:
            feature_sequence.append(feature)
            if len(feature_sequence) > sequence_length:
                feature_sequence.pop(0)

        if len(feature_sequence) == sequence_length:
            feature_sequence_np = np.array(feature_sequence)
            feature_sequence_np = feature_sequence_np.reshape((1, sequence_length, 4))  # Reshape to match model input
            motion_detected = lstm_model.predict(feature_sequence_np)[0][0]
            logger.info(f"Motion detection result for frame {frame_count}: {motion_detected}")

            if motion_detected > 0.5 and not check_proximity(boxes, class_ids, personnel_classes, icu_patient_class_id):
                logger.info(f"Motion detected on frame: {frame_count}")
                logger.info("The doctor should visit as soon as possible")
                # Save frame to file
                frame_file = f"uploads/frame_{frame_count}.jpg"
                cv2.imwrite(frame_file, frame)
                frame_confidences.append((frame_count, frame_file, max(confidences)))  # Store frame and confidence

    cap.release()
    out.release()  # Release the video writer

    # Sort frames by confidence and return the top 10
    frame_confidences.sort(key=lambda x: x[2], reverse=True)
    top_10_results = frame_confidences[:10]

    return [(frame[0], frame[1]) for frame in top_10_results], output_video_path

@app.route('/')
def index():
    return "Welcome to the Motion Detection API. Use /upload to upload a video and /process to process it."

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    video_path = os.path.join("uploads", video.filename)
    video.save(video_path)
    logger.info(f'Video uploaded successfully: {video_path}')

    return jsonify({'message': 'Video uploaded successfully', 'video_path': video_path}), 200

@app.route('/process', methods=['POST'])
def process_uploaded_video():
    data = request.json
    video_path = data.get('video_path')
    yaml_file = "data.yaml"
    model_weights = "best.pt"

    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Invalid video file path'}), 400
    if not yaml_file or not os.path.exists(yaml_file):
        return jsonify({'error': 'Invalid YAML file path'}), 400
    if not model_weights or not os.path.exists(model_weights):
        return jsonify({'error': 'Invalid model weights file path'}), 400

    results, output_video_path = process_video(video_path)

    return jsonify({
        'message': 'Processing complete',
        'results': [
            {'frame_number': frame_number, 'frame_image': frame_image}
            for frame_number, frame_image in results
        ],
        'output_video_path': output_video_path
    }), 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join("uploads", filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=9000)
