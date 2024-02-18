# AI Virtual Mouse Project
# Camden Bettencourt, Nathan VanSickle, and Aidan Limle

# place necessary imports here
import mediapipe as mp
import cv2
import pyautogui
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# environment variables
wCam, hCam = 640, 480
screenX, screenY = tuple(element/2 for element in pyautogui.size()) # center of screen
# debug show resolution print(f"screen dimensions: {screenX}x{screenY}")
clickThreshold = 0.05 # threshold for click gesture
sensitivity = 3.2 # controls how fast mouse moves

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

stopProgram = False

def frameAnalysis():
    global stopProgram
    while not stopProgram:
        success, img = cap.read()

        # display a window of the current webcam footage each frame
        img = cv2.flip(img, 1)

        hands = mp_hands.Hands()

        rgbFrame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(rgbFrame)

        if results.multi_hand_landmarks:
            # Grab first (only) set of hand landmarks
            lm = results.multi_hand_landmarks[0]

            # Isolate index fingertip and middle fingertip
            ttip = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
            iftip = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            mftip = lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            rftip = lm.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ptip = lm.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # debug show fingertip coordinates relative position print(f"X: {iftip_x} | Y: {iftip_y} | Z: {iftip_z}")
            cv2.circle(img, (int(ptip.x * img.shape[1]), int(ptip.y * img.shape[0])), 5, (0, 255, 0), -1)
            cv2.circle(img, (int(iftip.x * img.shape[1]), int(iftip.y * img.shape[0])), 5, (255, 0, 0), -1)
            cv2.circle(img, (int(mftip.x * img.shape[1]), int(mftip.y * img.shape[0])), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(ttip.x * img.shape[1]), int(ttip.y * img.shape[0])), 5, (255, 255, 255), -1)
            cv2.circle(img, (int(rftip.x * img.shape[1]), int(rftip.y * img.shape[0])), 5, (255, 165, 0), -1)

            cv2.line(img, (int(iftip.x * img.shape[1]), int(iftip.y * img.shape[0])), (int(ttip.x * img.shape[1]), int(ttip.y * img.shape[0])), (0, 255, 0), 5)
            cv2.line(img, (int(mftip.x * img.shape[1]), int(mftip.y * img.shape[0])), (int(ttip.x * img.shape[1]), int(ttip.y * img.shape[0])), (0, 255, 0), 5)
            cv2.line(img, (int(rftip.x * img.shape[1]), int(rftip.y * img.shape[0])), (int(ttip.x * img.shape[1]), int(ttip.y * img.shape[0])), (0, 0, 255), 5)

            cv2.imshow("Image", img)
            cv2.waitKey(1)

            # Move mouse cursor to current fingertip position
            fingerX = screenX + screenX * (ptip.x - 0.5) * sensitivity
            fingerY = screenY + screenY * (ptip.y - 0.5) * sensitivity

            # Pad mouse cursor to edges
            screenUpper = screenY * 2
            screenRight = screenX * 2
            fingerX = fingerX if fingerX > 3 else 3
            fingerY = fingerY if fingerY > 3 else 3
            fingerX = fingerX if fingerX < screenRight - 3 else screenRight - 3
            fingerY = fingerY if fingerY < screenUpper - 3 else screenUpper - 3

            # debug show fingertip coordinates by screen resolution 
            # print(f"X: {fingerX} | Y: {fingerY}")
            pyautogui.moveTo(fingerX, fingerY)

            # calculate distance from each finger to the thumb
            leftDistance = ((iftip.x - ttip.x)**2 + (iftip.y - ttip.y)**2)**0.5
            rightDistance = ((mftip.x - ttip.x)**2 + (mftip.y - ttip.y)**2)**0.5
            quitDistance = ((rftip.x - ttip.x)**2 + (rftip.y - ttip.y)**2)**0.5

            # click if finger is touching thumb
            if leftDistance < clickThreshold:
                pyautogui.leftClick()
            elif rightDistance < clickThreshold:
                pyautogui.rightClick()
            elif quitDistance < clickThreshold:
                stopProgram = True

frameAnalysis()

cv2.destroyAllWindows()
