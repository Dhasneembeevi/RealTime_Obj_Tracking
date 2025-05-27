# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import tempfile
# import time

# # Load YOLOv5 model once
# model = YOLO("yolov8s.pt")


# def detect_objects(frame):
#     results = model(frame)
#     annotated_frame = results[0].plot()
#     return annotated_frame


# def main():
#     st.title("Real-Time Object Detection with YOLOv8")
#     st.write(model.names)

#     # Select input mode
#     option = st.radio("Choose input source:", ["Webcam", "Upload Video"])

#     if option == "Webcam":
#         run_webcam()

#     elif option == "Upload Video":
#         uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
#         if uploaded_file is not None:
#             tfile = tempfile.NamedTemporaryFile(delete=False)
#             tfile.write(uploaded_file.read())
#             run_video(tfile.name)


# def run_webcam():
#     stframe = st.empty()
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         st.error("Cannot open webcam")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("Failed to grab frame")
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         annotated_frame = detect_objects(frame)

#         stframe.image(annotated_frame, channels="RGB")

#         # Stop button to exit loop
#         if st.button("Stop Webcam"):
#             break

#     cap.release()


# def run_video(path):
#     cap = cv2.VideoCapture(path)
#     stframe = st.empty()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         annotated_frame = detect_objects(frame)
#         stframe.image(annotated_frame, channels="RGB")

#         # Add a small delay to simulate real-time playback speed
#         time.sleep(1 / 30)

#     cap.release()


# if __name__ == "__main__":
#     main()

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import concurrent.futures

model1 = YOLO("yolov8s.pt")
model2 = YOLO("gunKnife.pt")
model3 = YOLO("fire.pt")
CONF_THRESHOLD = 0.5


def run_model(model, frame):
    return model(frame)[0]


def detect_objects_parallel(frame):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future1 = executor.submit(run_model, model1, frame)
        future2 = executor.submit(run_model, model2, frame)
        future3 = executor.submit(run_model, model3, frame)

        results1 = future1.result()
        results2 = future2.result()
        results3 = future3.result()

    detections = []

    # Model 1
    for box in results1.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model1.names[cls_id]
        if conf >= CONF_THRESHOLD:
            detections.append((box.xyxy[0].cpu().numpy(), conf, label))

    # Model 2
    for box in results2.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model2.names[cls_id]
        if conf >= CONF_THRESHOLD:
            detections.append((box.xyxy[0].cpu().numpy(), conf, label))

    for box in results3.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model3.names[cls_id]
        if conf >= CONF_THRESHOLD:
            detections.append((box.xyxy[0].cpu().numpy(), conf, label))

    annotated_frame = frame.copy()
    for xyxy, conf, label in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )

    return annotated_frame


def main():
    st.title("Real-Time Object Detection")
    option = st.radio("Choose input source:", ["Webcam", "Upload Video"])

    if option == "Webcam":
        run_webcam()
    elif option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            run_video(tfile.name)


def run_webcam():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open webcam")
        return

    stop = st.button("Stop Webcam")

    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = detect_objects_parallel(frame)
        stframe.image(annotated_frame, channels="RGB")

    cap.release()


def run_video(path):
    cap = cv2.VideoCapture(path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = detect_objects_parallel(frame)
        stframe.image(annotated_frame, channels="RGB")
        time.sleep(1 / 30)

    cap.release()


if __name__ == "__main__":
    main()
