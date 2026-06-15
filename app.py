import streamlit as st
import cv2
import mediapipe as mp
import pickle
import speech_recognition as sr
import time
import threading

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Sign Language AI", page_icon="🤟", layout="wide")
st.title("🤟 Sign Language Communication Dashboard")

# ---------------- MODEL ----------------
model = pickle.load(open("gesture_model.pkl", "rb"))

# ---------------- SAFE TTS (CLOUD FRIENDLY) ----------------
def speak(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except:
        # fallback for Streamlit Cloud
        st.toast(f"🔊 {text}")

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw = mp.solutions.drawing_utils

# ---------------- SESSION STATE ----------------
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

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📷 Live Camera Feed")
        FRAME_WINDOW = st.empty()

    with col2:
        if st.button("📷 Start Camera"):
            st.session_state.camera_on = True
            st.session_state.cap = cv2.VideoCapture(0)

        if st.button("🛑 Stop Camera"):
            st.session_state.camera_on = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None

        if st.button("➕ Add Gesture"):
            g = st.session_state.current_gesture
            if g not in ["None", ""]:
                if g == "SPACE":
                    st.session_state.sentence.append(" ")
                else:
                    st.session_state.sentence.append(g)

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

            prediction = "None"

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    data = []
                    for lm in hand.landmark:
                        data.extend([lm.x, lm.y])

                    prediction = model.predict([data])[0]
                    draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # stability filter
            st.session_state.prediction_buffer.append(prediction)
            if len(st.session_state.prediction_buffer) > 5:
                st.session_state.prediction_buffer.pop(0)

            stable_prediction = max(
                set(st.session_state.prediction_buffer),
                key=st.session_state.prediction_buffer.count
            )

            st.session_state.current_gesture = stable_prediction

            # speech (safe)
            if stable_prediction not in ["None", ""]:
                if stable_prediction != st.session_state.last_spoken:
                    speak_async(stable_prediction)
                    st.session_state.last_spoken = stable_prediction

            FRAME_WINDOW.image(frame, channels="BGR")
            gesture_display.metric("Gesture", stable_prediction)

            time.sleep(0.03)

# =====================================================
# ================== VOICE MODE ========================
# =====================================================
elif mode == "Voice":

    st.subheader("🎤 Speech to Text (Upload Audio Only)")

    audio_file = st.file_uploader("Upload WAV file", type=["wav"])

    if audio_file:
        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)

            text = recognizer.recognize_google(audio)
            st.success("You said: " + text)

        except Exception as e:
            st.error("Speech recognition failed")

# =====================================================
# ================== TEXT MODE =========================
# =====================================================
elif mode == "Text":

    st.subheader("⌨️ Text to Speech")

    text_input = st.text_input("Enter text")

    if st.button("Speak Text"):
        speak_async(text_input)
