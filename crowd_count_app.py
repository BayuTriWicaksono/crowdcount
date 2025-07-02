import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
import tempfile

st.title("Crowd Counting Web App with YOLOv12")

# Load model
model = YOLO("yolo12n.pt")

# Upload video
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Simpan video ke file temporer
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Buka video
    cap = cv2.VideoCapture(video_path)
    frame_window = st.empty()

    ptime = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi orang (class 0)
        results = model.predict(frame, conf=0.5, iou=0.5, classes=[0])

        person_count = 0
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id == 0:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    conf = math.ceil(box.conf[0] * 100) / 100
                    label = f"Person: {conf}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hitung FPS
        ctime = cv2.getTickCount() / cv2.getTickFrequency()
        fps_now = 1 / (ctime - ptime) if (ctime - ptime) != 0 else 0
        ptime = ctime

        # Tulis info di frame
        cv2.putText(frame, f"People: {person_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps_now)}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Tampilkan ke Streamlit
        frame_window.image(frame, channels="BGR")

    cap.release()
