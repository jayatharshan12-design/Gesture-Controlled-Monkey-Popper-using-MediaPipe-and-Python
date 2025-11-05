import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

monkey_near = cv2.imread("monkey1.png", cv2.IMREAD_UNCHANGED)
monkey_far = cv2.imread("monkey2.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

def overlay_transparent(background, overlay):
    overlay_resized = cv2.resize(overlay, (background.shape[1], background.shape[0]))
    if overlay_resized.shape[2] == 4:
        alpha = overlay_resized[:, :, 3] / 255.0
        for c in range(3):
            background[:, :, c] = (
                background[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
            )
    else:
        background[:] = overlay_resized
    return background

last_state = None
last_change_time = 0
stable_duration = 0.5

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_state = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape
            face_x, face_y = w // 2, h // 2
            index_x = int(landmarks[8].x * w)
            index_y = int(landmarks[8].y * h)
            distance = np.sqrt((index_x - face_x) ** 2 + (index_y - face_y) ** 2)

            if distance < 150:
                current_state = "near"
            elif distance > 250:
                current_state = "far"

    now = time.time()
    if current_state != last_state:
        last_change_time = now
    elif current_state and (now - last_change_time) > stable_duration:
        if current_state == "near":
            frame = overlay_transparent(frame, monkey_near)
        elif current_state == "far":
            frame = overlay_transparent(frame, monkey_far)

    last_state = current_state

    cv2.imshow("🐒 Monkey Popper", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
