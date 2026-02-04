import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO
import requests

# Load YOLO model
model = YOLO("tomato.pt")  # YOUR MODEL PATH

# IP address - UPDATE THIS
ESP_IP = "192.168.1.100"  # YOUR ESP32 IP

def send_esp_command(color):
    try:
        if color == "green":
            requests.get(f"http://{ESP_IP}/green", timeout=0.5)
        else:
            requests.get(f"http://{ESP_IP}/red", timeout=0.5)
    except requests.exceptions.RequestException:
        print("Unable to reach ESP device.")

# Streamlit configurations
st.set_page_config(page_title="Smart Tomato Detection", layout="wide")

# Custom CSS - DARK THEME + GLOWING LIGHTS (EXACT ambulance style)
st.markdown("""
<style>
    .stApp, .main { background-color: #121212; color: white; }
    h1, h2, h3 { color: #1de9b6; text-align: center; font-weight: 700; }
    .paragraph { color: #cccccc; font-size: 1.1rem; line-height: 1.6; }

    .light-green { font-size: 4rem; text-align: center; animation: glowG 1s infinite alternate; }
    .light-red { font-size: 4rem; text-align: center; animation: glowR 1s infinite alternate; }

    @keyframes glowG { from {text-shadow: 0 0 5px #00ff00;} to {text-shadow: 0 0 25px #00ff00;} }
    @keyframes glowR { from {text-shadow: 0 0 5px #ff0000;} to {text-shadow: 0 0 25px #ff0000;} }
</style>
""", unsafe_allow_html=True)

st.title("🌱 Smart Tomato Plant Detection System")

left, right = st.columns([2, 1])

with right:
    st.subheader("📘 Project Overview")
    st.markdown("""
    <div class="paragraph">
    This AI-powered system detects TOMATO PLANTS in real-time using YOLO deep learning.  
    When a tomato plant is detected:  
    <br><br>
    • 🟢 GREEN LED turns ON  
    • 🔴 RED LED turns OFF  
    <br><br>
    Works with **live camera feed**, **uploaded videos**, and **images**.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("⚙ Input Source")
    source = st.radio("Choose input type:", ["Live Camera", "Upload Image", "Upload Video"])

with left:
    st.subheader("📡 Live Detection Feed")
    frame_box = st.empty()
    light_box = st.empty()
    fps_box = st.empty()
    confidence_box = st.empty()

def process_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, conf=0.4)  # Tomato plant confidence

    tomato_plant = False
    conf_value = 0

    for r in results:
        for box in r.boxes:
            c = box.conf[0].item()
            # ANY detection = tomato plant (your model trained on tomatoes)
            tomato_plant = True
            conf_value = max(conf_value, round(c * 100, 1))

    annotated = results[0].plot()
    return annotated, tomato_plant, conf_value

if source == "Live Camera":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open camera.")
        st.stop()

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Frame error")
            break

        start = time.time()
        annotated, tomato_detected, conf_val = process_image(frame)
        end = time.time()

        fps = round(1 / (end - start), 1)

        frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", width=640)

        fps_box.markdown(f"### ⚡ FPS: **{fps}**")
        confidence_box.markdown(f"### 📊 Confidence: **{conf_val}%**")

        # Send command to ESP32 - EXACT ambulance logic
        if tomato_detected:
            light_box.markdown("<div class='light-green'>🟢</div>", unsafe_allow_html=True)
            send_esp_command("green")
        else:
            light_box.markdown("<div class='light-red'>🔴</div>", unsafe_allow_html=True)
            send_esp_command("red")

        time.sleep(0.03)

elif source == "Upload Image":
    uploaded = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        annotated, tomato_detected, conf_val = process_image(img)

        frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", width=640)
        confidence_box.markdown(f"### 📊 Confidence: **{conf_val}%**")

        if tomato_detected:
            light_box.markdown("<div class='light-green'>🟢</div>", unsafe_allow_html=True)
            send_esp_command("green")
        else:
            light_box.markdown("<div class='light-red'>🔴</div>", unsafe_allow_html=True)
            send_esp_command("red")

elif source == "Upload Video":
    uploaded = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded:
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded.read())

        cap = cv2.VideoCapture(tfile)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            annotated, tomato_detected, conf_val = process_image(frame)
            end = time.time()

            fps = round(1 / (end - start), 1)

            frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", width=640)
            fps_box.markdown(f"### ⚡ FPS: **{fps}**")
            confidence_box.markdown(f"### 📊 Confidence: **{conf_val}%**")

            if tomato_detected:
                light_box.markdown("<div class='light-green'>🟢</div>", unsafe_allow_html=True)
                send_esp_command("green")
            else:
                light_box.markdown("<div class='light-red'>🔴</div>", unsafe_allow_html=True)
                send_esp_command("red")

            time.sleep(0.03)

        cap.release()
