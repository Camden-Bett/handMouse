import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# environment variables
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    success, img = cap.read()

    # display a window of the current webcam footage each frame
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    hands = mp_hands.Hands()

    rgbFrame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgbFrame)

    if results.multi_hand_landmarks:
        # Grab first (only) set of hand landmarks
        lm = results.multi_hand_landmarks[0]

        # Isolate index fingertip
        tip = lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        # Get coords
        tip_x, tip_y, tip_z = tip.x, tip.y, tip.z

        print(f"X: {tip_x} | Y: {tip_y} | Z: {tip_z}")


