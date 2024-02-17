import cv2

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