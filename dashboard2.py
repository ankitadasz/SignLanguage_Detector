import streamlit as st
import cv2
import mediapipe as mp
import pickle
import speech_recognition as sr
import time
import threading
import pyttsx3
import numpy as np
from collections import Counter

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Sign Language AI", page_icon="🤟", layout="wide")
st.title("🤟 Sign Language Communication Dashboard")

# ---------------- MODEL ----------------
model = pickle.load(open("gesture_model.pkl", "rb"))

# ---------------- SAFE TTS ----------------
def create_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    return engine

def speak(text):
    if text.strip() == "":
        return
    try:
        engine = create_engine()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except:
        pass

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils

# ---------------- SESSION ----------------
if "sentence" not in st.session_state:
    st.session_state.sentence = []

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if "cap" not in st.session_state:
    st.session_state.cap = None

if "history" not in st.session_state:
    st.session_state.history = []

if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = ""

if "prediction_buffer" not in st.session_state:
    st.session_state.prediction_buffer = []

if "current_gesture" not in st.session_state:
    st.session_state.current_gesture = ""

# ---------------- MODE ----------------
mode = st.radio("Select Mode", ["Gesture", "Voice", "Text"])

# =====================================================
# ================== GESTURE MODE ======================
# =====================================================
if mode == "Gesture":

    col1, col2 = st.columns([3,1])

    with col1:
        st.subheader("📷 Live Camera Feed")
        FRAME_WINDOW = st.empty()

    with col2:
        # ✅ ORDER FIXED
        if st.button("📷 Start Camera"):
            st.session_state.camera_on = True
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.last_spoken = ""

        if st.button("🛑 Stop Camera"):
            st.session_state.camera_on = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None

        if st.button("➕ Add Gesture"):
            g = st.session_state.current_gesture
            if g not in ["NONE", ""]:
                st.session_state.sentence.append(g)

        # ✅ NEW SPACE BUTTON
        if st.button("␣ Space"):
            st.session_state.sentence.append(" ")

        gesture_display = st.empty()

    # -------- CAMERA LOOP --------
    if st.session_state.camera_on:

        cap = st.session_state.cap

        while st.session_state.camera_on:

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            prediction = "NONE"

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:

                    x_list, y_list, z_list = [], [], []

                    for lm in hand.landmark:
                        x_list.append(lm.x)
                        y_list.append(lm.y)
                        z_list.append(lm.z)

                    min_x, min_y, min_z = min(x_list), min(y_list), min(z_list)

                    row = []
                    for lm in hand.landmark:
                        row.append(lm.x - min_x)
                        row.append(lm.y - min_y)
                        row.append(lm.z - min_z)

                    max_val = max(row)
                    if max_val != 0:
                        row = [i / max_val for i in row]

                    prediction = model.predict([row])[0]

                    draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                    cv2.putText(frame, prediction, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
            else:
                prediction = "NONE"
                st.session_state.prediction_buffer.clear()

            # -------- STABILITY FILTER --------
            st.session_state.prediction_buffer.append(prediction)
            if len(st.session_state.prediction_buffer) > 4:
                st.session_state.prediction_buffer.pop(0)

            stable_prediction = Counter(st.session_state.prediction_buffer).most_common(1)[0][0]
            st.session_state.current_gesture = stable_prediction

            # -------- SPEAK --------
            if stable_prediction != "NONE":
                if stable_prediction != st.session_state.last_spoken:
                    speak_async(stable_prediction)
                    st.session_state.last_spoken = stable_prediction

            # -------- HISTORY --------
            if stable_prediction != "NONE":
                if len(st.session_state.history) == 0 or \
                   st.session_state.history[-1] != stable_prediction:
                    st.session_state.history.append(stable_prediction)

            st.session_state.history = st.session_state.history[-5:]

            FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
            gesture_display.metric("Gesture", stable_prediction)

            time.sleep(0.03)

            if not st.session_state.camera_on:
                break

    # -------- SENTENCE --------
    st.subheader("📝 Sentence")
    st.write("".join(st.session_state.sentence))

    colA, colB, colC = st.columns(3)

    with colA:
        if st.button("🔊 Speak Sentence"):
            sentence = "".join(st.session_state.sentence)
            if sentence.strip() != "":
                speak_async(sentence)

    with colB:
        if st.button("🗑 Clear Sentence"):
            st.session_state.sentence.clear()

    with colC:
        if st.button("⬅ Delete Last"):
            if len(st.session_state.sentence) > 0:
                st.session_state.sentence.pop()

# =====================================================
# ================== VOICE MODE ========================
# =====================================================
elif mode == "Voice":

    st.subheader("🎤 Speech to Text")

    if st.button("Start Listening"):
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            st.write("Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
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