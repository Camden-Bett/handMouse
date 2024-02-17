import mediapipe as mp
import cv2
import threading
import keyboard
import pyautogui
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# environment variables
wCam, hCam = 640, 480
screenX, screenY = pyautogui.size() # dimensions of screen
# debug show resolution print(f"screen dimensions: {screenX}x{screenY}")

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

stopProgram = False

# Key Listener thread
def keyListener():
    global stopProgram
    
    keyboard.wait("q")
    stopProgram = True

keythread = threading.Thread(target=keyListener)
keythread.start()

def frameAnalysis():
    while not stopProgram:
        success, img = cap.read()

        # display a window of the current webcam footage each frame
        img = cv2.flip(img, 1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        hands = mp_hands.Hands()

        rgbFrame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(rgbFrame)

        if results.multi_hand_landmarks:
            # Grab first (only) set of hand landmarks
            lm = results.multi_hand_landmarks[0]

            # Isolate index fingertip and middle fingertip
            iftip = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            mftip = lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Get coords
            iftip_x, iftip_y, iftip_z = iftip.x, iftip.y, iftip.z

            # debug show fingertip coordinates relative position print(f"X: {iftip_x} | Y: {iftip_y} | Z: {iftip_z}")
            
            # Move mouse cursor to current fingertip position
            fingerX = screenX * iftip_x
            fingerY = screenY * iftip_y
            # debug show fingertip coordinates by screen resolution 
            # print(f"X: {fingerX} | Y: {fingerY}")
            pyautogui.moveTo(fingerX, fingerY)

framethread = threading.Thread(target=frameAnalysis)
framethread.start()

keythread.join()
framethread.join()

cv2.destroyAllWindows()


