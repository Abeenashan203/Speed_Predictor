# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import joblib
import os
import time
from skimage.feature import hog

# 🚀 Page Layout & Title Setup
st.set_page_config(page_title="VEHICLE SPEED DETECTOR", layout="wide")

# ==========================================
# 🌌 TWO-VIDEO BACKGROUND PLAYLIST ENGINE 🎬
# ==========================================
# Streaming directly using the GitHub Raw content delivery engine
url_1 = "https://raw.githubusercontent.com/Abeenashan203/Speed_Predictor/main/car_video1.mp4"
url_2 = "https://raw.githubusercontent.com/Abeenashan203/Speed_Predictor/main/car_video2.mp4"

playlist_html = f"""
<style>
/* 1. Anchors the background video to fill the viewport seamlessly */
#bgVideo {{
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  object-fit: cover;
  z-index: -1;
  opacity: 0.25;
}}

/* 2. Global Transparency Reset for Streamlit layouts and custom widgets */
.stApp, 
.main, 
block-container, 
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stVerticalBlock"] > div,
div[data-testid="stMarkdownContainer"] p,
.stRadio,
div[role="radiogroup"],
div[data-baseweb="base-input"],
[data-testid="stFileUploader"] {{
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border-color: transparent !important;
}}

/* Removes default top/bottom block padding */
[data-testid="stAppViewContainer"] {{
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}}
</style>

<video id="bgVideo" autoplay muted playsinline>
  <source id="videoSource" src="{url_1}" type="video/mp4">
  Your browser does not support HTML5 video.
</video>

<script>
const video = document.getElementById('bgVideo');
const source = document.getElementById('videoSource');
const playlist = ["{url_1}", "{url_2}"];
let currentVideoIndex = 0;

video.addEventListener('ended', function() {{
    currentVideoIndex = (currentVideoIndex + 1) % playlist.length;
    source.src = playlist[currentVideoIndex];
    video.load();
    video.play().catch(error => console.log("Playback loop logs:", error));
}});
</script>
"""
st.markdown(playlist_html, unsafe_allow_html=True)


# ==========================================
# ⚡ DASHBOARD HEADERS & SIDEBAR 📊
# ==========================================
st.title("🚗 Vehicle Speed Estimator & Analytics System 📈")
st.markdown("-----")

st.sidebar.header("🎛️ System Configuration ⚙️")
SPEED_LIMIT = st.sidebar.slider("🚨 Speed Limit (km/h)", min_value=30, max_value=120, value=60)
PPM = st.sidebar.number_input("📏 Pixels Per Meter (PPM Calibration)", min_value=1.0, max_value=50.0, value=10.0)

# Load Models
@st.cache_resource
def load_models():
    model_path = "vehicle_svm_detector.joblib"
    scaler_path = "svm_scaler.joblib"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

svm_model, svm_scaler = load_models()

if svm_model is None:
    st.error("🛑 'vehicle_svm_detector.joblib' & 'svm_scaler.joblib' were not found in the repository root directory.")
    st.stop()


# ==========================================
# 📹 INPUT SELECTION (FILE VS LIVE CAMERA) 📸
# ==========================================
source_type = st.radio("🛠️ Select Video Input Source:", ["📁 Upload Video File", "📷 Live Web Camera Feed"], horizontal=True)

uploaded_file = None
camera_file = None

if source_type == "📁 Upload Video File":
    uploaded_file = st.file_uploader("📤 Upload Traffic Video File (MP4, AVI) 🎞️", type=["MP4", "AVI"])
else:
    camera_file = st.camera_input("📸 Capture/Stream Live Traffic Footage")

active_video = uploaded_file if source_type == "📁 Upload Video File" else camera_file


# ==========================================
# 🧠 IMAGE PROCESSING FUNCTIONS
# ==========================================
def stabilize_frame(prev_frame, curr_frame):
    if prev_frame is None: return curr_frame
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=30)
    if pts is None: return curr_frame
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
    idx = np.where(status == 1)[0]
    if len(idx) > 10:
        M, _ = cv2.estimateAffinePartial2D(pts[idx], next_pts[idx])
        if M is not None:
            return cv2.warpAffine(curr_frame, M, (curr_frame.shape[1], curr_frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    return curr_frame

def classify_vehicle(cropped_img):
    if cropped_img is None or cropped_img.size == 0: return None
    resized = cv2.resize(cropped_img, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    scaled_feat = svm_scaler.transform([hog_feat])
    return svm_model.predict(scaled_feat)[0]


class CentroidTracker:
    def __init__(self):
        self.next_id = 1
        self.objects = {}
        self.previous_objects = {}
        self.speeds = {}
        self.classes = {}

    def update(self, rects, frame, fps):
        current_rect_centroids = []
        for (x, y, w, h) in rects:
            current_rect_centroids.append((int(x + w / 2), int(y + h / 2), x, y, w, h))

        new_objects = {}
        for (cx, cy, x, y, w, h) in current_rect_centroids:
            matched_id, min_dist = None, 50
            for obj_id, old_cx_cy in self.objects.items():
                dist = np.hypot(cx - old_cx_cy[0], cy - old_cx_cy[1])
                if dist < min_dist:
                    min_dist, matched_id = dist, obj_id

            if matched_id is not None:
                new_objects[matched_id] = (cx, cy)
                self.previous_objects[matched_id] = self.objects[matched_id]
                prev_cx, prev_cy = self.previous_objects[matched_id]
                
                # Math Analytics
                speed_kmh = int(((np.hypot(cx - prev_cx, cy - prev_cy) / PPM) / (1.0 / fps)) * 3.6)
                if matched_id in self.speeds:
                    self.speeds[matched_id] = int(self.speeds[matched_id] * 0.7 + speed_kmh * 0.3)
                else:
                    self.speeds[matched_id] = speed_kmh
            else:
                cropped = frame[y:y+h, x:x+w]
                v_class = classify_vehicle(cropped)
                if v_class is not None:
                    new_objects[self.next_id] = (cx, cy)
                    self.classes[self.next_id] = v_class
                    self.next_id += 1
        self.objects = new_objects
        return self.objects


# ==========================================
# 🎬 RUNTIME PROCESSING LOOP
# ==========================================
if active_video is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(active_video.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    tracker = CentroidTracker()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎥 Real-time Video Analysis 🔍")
        video_placeholder = st.empty()
    with col2:
        st.subheader("💥 Speed Limit Violations Log 📋")
        log_placeholder = st.empty()

    violation_log = []
    prev_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        stabilized = stabilize_frame(prev_frame, frame)
        prev_frame = stabilized.copy()

        mask = object_detector.apply(stabilized)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 400:
                rects.append(cv2.boundingRect(cnt))

        tracked_objs = tracker.update(rects, stabilized, fps)

        for obj_id, (cx, cy) in tracked_objs.items():
            v_class = tracker.classes.get(obj_id, "Detecting...")
            speed = tracker.speeds.get(obj_id, 0)

            for (x, y, w, h) in rects:
                if abs(cx - (x + w / 2)) < 30 and abs(cy - (y + h / 2)) < 30:
                    color = (0, 0, 255) if speed > SPEED_LIMIT else (0, 255, 0)
                    cv2.rectangle(stabilized, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(stabilized, f"ID:{obj_id} {v_class} | {speed} km/h", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if speed > SPEED_LIMIT:
                        log_entry = {"⚠️ Vehicle ID": obj_id, "🚙 Type": v_class, "💨 Speed": f"{speed} km/h", "⚡ Status": "🔥 OVER SPEED"}
                        if log_entry not in violation_log:
                            violation_log.append(log_entry)
                            with col2:
                                log_placeholder.dataframe(violation_log, use_container_width=True)

        frame_rgb = cv2.cvtColor(stabilized, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        time.sleep(1 / fps)

    cap.release()
    st.success("🎉 Video Speed Prediction Completed! ✨ 🚀")
