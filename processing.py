import mediapipe as mp

hand = mp.solutions.hands.Hand()

rgbFrame = "test"

if results.multi_hand_landmarks:
    # Grab first (only) set of hand landmarks
    lm = results.multi_hand_landmarks[0]

    # Isolate index fingertip
    tip = lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

    # Get coords
    tip_x, tip_y, tip_z = tip.x, tip.y, tip.z

    print(f"X: {tip_x} | Y: {tip_y} | Z: {tip_z}")


