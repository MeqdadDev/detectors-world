"""Count raised fingers on one hand using landmark positions from locate().

Press q to quit.
"""

import cv2 as cv
# from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

from detectors_world import create_detector

FINGER_TIPS = [8, 12, 16, 20]

# To use landmark names instead of numbers, uncomment the import above and this
# list, then use the second "raised" line below in place of the first.
# FINGERS = [
#     (HandLandmark.INDEX_FINGER_TIP, HandLandmark.INDEX_FINGER_PIP),
#     (HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_PIP),
#     (HandLandmark.RING_FINGER_TIP, HandLandmark.RING_FINGER_PIP),
#     (HandLandmark.PINKY_TIP, HandLandmark.PINKY_PIP),
# ]

cap = cv.VideoCapture(0)
with create_detector("hand", max_hands=1) as hand:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv.flip(frame, 1)
        hand.detect(frame)
        landmarks = hand.locate(frame, draw=False)
        if landmarks:
            # A finger is raised when its tip is above the joint two points below it (smaller y).
            raised = sum(landmarks[tip][2] < landmarks[tip - 2][2] for tip in FINGER_TIPS)
            # raised = sum(landmarks[tip][2] < landmarks[joint][2] for tip, joint in FINGERS)
            cv.putText(frame, str(raised), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)
        cv.imshow("Finger Counter", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv.destroyAllWindows()
