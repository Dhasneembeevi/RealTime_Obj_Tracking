# 🔍 Real-Time Object & Violence Detection App

A real-time detection system built using **Streamlit**, **YOLOv8**, and **MediaPipe**, capable of identifying:

- Common objects (YOLOv8s)
- **Weapons** (Gun/Knife) – Custom YOLOv8 model
- **Fire hazards** – Custom YOLOv8 model
- **Violent behavior** – Pose-based classification using a custom-trained Random Forest Classifier model

---

## 🚀 Features

- 🔄 Real-time detection using **Webcam**
- 📤 Support for **uploaded video files**
- ⚡ Parallel processing using **ThreadPoolExecutor**
- 🤸 Pose estimation via **MediaPipe**
- 🧠 Violence detection using a trained **Pose Classification Model**
- 📸 Auto-capture and save of violent frames

---

## 🖥️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
streamlit run combine.py
