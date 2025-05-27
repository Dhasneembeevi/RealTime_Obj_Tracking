import os
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import concurrent.futures
import joblib
import mediapipe as mp
import random

st.set_page_config(layout="wide")

model1 = YOLO("yolov8s.pt")
model2 = YOLO("gunKnife.pt")
model3 = YOLO("fire.pt")
model4 = joblib.load("violence_pose_model.pkl")

CONF_THRESHOLD = 0.6
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

COLOR_MAP = {}
DEFAULT_COLOR = (255, 255, 255)


def get_color(label):
    if label not in COLOR_MAP:
        random.seed(hash(label) % 10000)
        color = tuple(random.randint(0, 150) for _ in range(3))
        COLOR_MAP[label] = color
    return COLOR_MAP[label]


def run_model(model, frame):
    return model(frame)[0]


def get_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        flat = [coord for point in keypoints for coord in point]
        return flat, results.pose_landmarks
    return None, None


def predict_violence(frame):
    keypoints, landmarks = get_pose_landmarks(frame)
    if keypoints:
        input_data = np.array(keypoints).reshape(1, -1)
        prediction = model4.predict(input_data)[0]
        return prediction, landmarks
    return None, None


def detect_objects_parallel(frame):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future1 = executor.submit(run_model, model1, frame)
        future2 = executor.submit(run_model, model2, frame)
        future3 = executor.submit(run_model, model3, frame)

        results1 = future1.result()
        results2 = future2.result()
        results3 = future3.result()

    detections = []

    for results, model in [(results1, model1), (results2, model2), (results3, model3)]:
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls_id]
            if conf >= CONF_THRESHOLD:
                detections.append((box.xyxy[0].cpu().numpy(), conf, label))

    annotated_frame = frame.copy()
    for xyxy, conf, label in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        color = get_color(label)

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated_frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            2,
        )
    return annotated_frame


def main():
    st.title("🔍 Real-Time Detection")
    photo_captured = False
    output_dir = os.path.join(os.getcwd(), "captured_violent_photos")
    os.makedirs(output_dir, exist_ok=True)

    option = st.radio("Choose input source:", ["Webcam", "Upload Video"])
    cap = None
    video_path = None

    if option == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            cap = cv2.VideoCapture(video_path)

    if cap and cap.isOpened():
        run = st.button("▶ Start Detection")

        col1, col2 = st.columns([1, 2])
        pose_placeholder = col1.empty()
        status_placeholder = col1.empty()
        obj_placeholder = col2.empty()

        if run:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("End of stream or failed to read frame.")
                    break

                frame = cv2.resize(frame, (640, 480))
                annotated_obj = detect_objects_parallel(frame)
                prediction, landmarks = predict_violence(frame)

                annotated_pose = frame.copy()
                label = "No Pose Detected"
                color = (200, 200, 200)

                if landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_pose, landmarks, mp_pose.POSE_CONNECTIONS
                    )

                    if prediction == 1:
                        label = "Violent"
                        color = (0, 0, 255)

                        if not photo_captured:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"violent_frame_{timestamp}.jpg"
                            file_path = os.path.join(output_dir, filename)
                            if cv2.imwrite(file_path, frame):
                                photo_captured = True
                                st.success(
                                    f"Violent activity detected. Photo saved as: {filename}"
                                )
                            else:
                                st.error("Failed to save photo!")

                    elif prediction == 0:
                        label = "Normal"
                        color = (0, 255, 0)

                cv2.putText(
                    annotated_pose,
                    label,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    2,
                )

                pose_placeholder.image(
                    cv2.cvtColor(annotated_pose, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    caption="Pose & Violence Detection",
                )
                status_placeholder.markdown(f"### **Status:** `{label}`")
                obj_placeholder.image(
                    cv2.cvtColor(annotated_obj, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    caption="Object Detection",
                )

                time.sleep(1 / 30)

            cap.release()
        else:
            st.info("Click 'Start Detection' to begin.")


if __name__ == "__main__":
    main()
