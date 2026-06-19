import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import threading

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Sign Language AI", page_icon="🤟", layout="wide")
st.title("🤟 Sign Language Communication Dashboard")

# ---------------- MODEL ----------------
model = pickle.load(open("gesture_model.pkl", "rb"))

# ---------------- SAFE SPEECH ----------------
def speak(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except:
        st.toast(f"🔊 {text}")

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
draw = mp.solutions.drawing_utils

# ---------------- SESSION STATE ----------------
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if "cap" not in st.session_state:
    st.session_state.cap = None

if "current_gesture" not in st.session_state:
    st.session_state.current_gesture = "None"

if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = ""

if "prediction_buffer" not in st.session_state:
    st.session_state.prediction_buffer = []

# ---------------- MODE ----------------
mode = st.radio("Select Mode", ["Gesture", "Voice", "Text"])

# =====================================================
# ================== GESTURE MODE ======================
# =====================================================
if mode == "Gesture":

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📷 Live Camera Feed")
        frame_window = st.empty()

    with col2:
        start = st.button("📷 Start Camera")
        stop = st.button("🛑 Stop Camera")

        gesture_box = st.empty()
        status_box = st.empty()

    # START CAMERA
    if start:
        st.session_state.camera_on = True
        st.session_state.cap = cv2.VideoCapture(0)

    # STOP CAMERA
    if stop:
        st.session_state.camera_on = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None

    # ---------------- CAMERA PROCESSING (SAFE LOOP) ----------------
    if st.session_state.camera_on and st.session_state.cap is not None:

        cap = st.session_state.cap

        for _ in range(200):  # limited loop (Streamlit-safe)

            if not st.session_state.camera_on:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            prediction = "None"

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:

                    data = []
                    for lm in hand.landmark:
                        data.extend([lm.x, lm.y])

                    if len(data) == 42:  # safety check
                        prediction = model.predict([data])[0]

                    draw.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )

            # ---------------- STABILITY FILTER ----------------
            st.session_state.prediction_buffer.append(prediction)

            if len(st.session_state.prediction_buffer) > 5:
                st.session_state.prediction_buffer.pop(0)

            stable = max(
                set(st.session_state.prediction_buffer),
                key=st.session_state.prediction_buffer.count
            )

            st.session_state.current_gesture = stable

            # ---------------- SPEECH ----------------
            if stable not in ["None", ""]:
                if stable != st.session_state.last_spoken:
                    speak_async(stable)
                    st.session_state.last_spoken = stable

            # ---------------- UI UPDATE ----------------
            frame_window.image(frame, channels="BGR")
            gesture_box.metric("Detected Gesture", stable)
            status_box.info("Camera Running...")

            time.sleep(0.03)

        status_box.warning("Camera Stopped")

# =====================================================
# ================== VOICE MODE =======================
# =====================================================
elif mode == "Voice":

    st.subheader("🎤 Speech to Text (Upload Audio Only)")

    audio_file = st.file_uploader("Upload WAV file", type=["wav"])

    if audio_file:
        import speech_recognition as sr

        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)

            text = recognizer.recognize_google(audio)
            st.success("You said: " + text)

        except:
            st.error("Speech recognition failed")

# =====================================================
# ================== TEXT MODE =========================
# =====================================================
elif mode == "Text":

    st.subheader("⌨️ Text to Speech")

    text_input = st.text_input("Enter text")

    if st.button("Speak Text"):
        speak_async(text_input)
