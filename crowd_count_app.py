import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
import tempfile
from PIL import Image

st.title("Crowd Counting Web App with YOLOv12")

# Load model YOLO
model = YOLO("yolo12n.pt")

# Upload video
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Simpan video ke file sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_window = st.empty()
    progress_bar = st.progress(0)
    total_person_count = 0

    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi objek
        results = model.predict(frame, conf=0.5, iou=0.5, classes=[0])
        person_count = 0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id == 0:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = math.ceil(box.conf[0] * 100) / 100
                    label = f"Person: {conf}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), 0, 0.5, (0, 255, 0), 2)

        total_person_count += person_count

        # Konversi frame ke RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # Tampilkan frame ke Streamlit
        frame_window.image(img_pil)

        current_frame += 1
        progress_bar.progress(current_frame / total_frames)

    cap.release()

    st.success(f"Proses selesai. Total deteksi orang: {total_person_count}")
