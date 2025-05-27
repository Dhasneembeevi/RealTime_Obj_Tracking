import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tempfile
import shutil
import time

# Load model
model = joblib.load("violence_pose_model.pkl")

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Extract pose keypoints
def get_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        flat = [coord for point in keypoints for coord in point]
        return flat, results.pose_landmarks
    return None, None


# Make prediction
def predict(frame):
    keypoints, landmarks = get_pose_landmarks(frame)
    if keypoints:
        input_data = np.array(keypoints).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        return prediction, landmarks
    return None, None


# Streamlit UI
st.title("🎥 Real-Time Violence Detection using Pose Estimation")

option = st.radio("Choose Input Method", ("Upload Video", "Webcam"))

cap = None  # Define cap in global scope

if option == "Upload Video":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file is not None:
        # Use a temporary file to store uploaded video
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        shutil.copyfileobj(video_file, temp_video)
        cap = cv2.VideoCapture(temp_video.name)

elif option == "Webcam":
    cap = cv2.VideoCapture(0)

# Start button only shown if video or webcam is selected
if cap:
    run = st.button("Start Detection")
    frame_placeholder = st.empty()
    label_placeholder = st.empty()

    if run:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("Video ended or could not read frame.")
                break

            frame = cv2.resize(frame, (640, 480))
            prediction, landmarks = predict(frame)

            if landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

            # Labeling
            if prediction == 1:
                label = "Violent"
                color = (0, 0, 255)
            elif prediction == 0:
                label = "Normal"
                color = (0, 255, 0)
            else:
                label = "No Pose Detected"
                color = (200, 200, 200)

            cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            label_placeholder.markdown(f"**Status:** {label}")

            time.sleep(0.03)  # Delay to control refresh rate

        cap.release()
