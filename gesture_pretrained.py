import cv2
import mediapipe as mp
import pyttsx3
import time

# ---------------- TEXT TO SPEECH ----------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# ---------------- MEDIAPIPE HANDS (PRETRAINED) ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA (LAPTOP) ----------------
cap = cv2.VideoCapture(0)

last_gesture = ""
last_time = 0

def speak(text):
    engine.say(text)
    engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = ""

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        # Finger states (OPEN = True)
        index_open  = lm[8].y  < lm[6].y
        middle_open = lm[12].y < lm[10].y
        ring_open   = lm[16].y < lm[14].y
        pinky_open  = lm[20].y < lm[18].y

        # ---------------- GESTURE LOGIC ----------------
        if index_open and middle_open and ring_open and pinky_open:
            gesture = "STOP"
        elif not index_open and not middle_open and not ring_open and not pinky_open:
            gesture = "FIST"
        elif index_open and not middle_open and not ring_open and not pinky_open:
            gesture = "YES"
        elif index_open and middle_open and not ring_open and not pinky_open:
            gesture = "HELLO"

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    # ---------------- SPEECH CONTROL ----------------
    if gesture and gesture != last_gesture and time.time() - last_time > 1.5:
        print("Detected:", gesture)
        speak(gesture)
        last_gesture = gesture
        last_time = time.time()

    # ---------------- DISPLAY ----------------
    cv2.putText(
        frame,
        gesture,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Pretrained Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
