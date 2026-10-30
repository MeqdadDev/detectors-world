"""Count raised fingers on one hand using landmark positions from locate().

Press q to quit.
"""

import math

import cv2 as cv
# from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

from detectors_world import create_detector

FINGER_TIPS = [8, 12, 16, 20]
THUMB_TIP = 4
THUMB_JOINT = 3
PINKY_BASE = 17

# To use landmark names instead of numbers, uncomment the import above and these
# lines, then use the second "raised" line below in place of the first.
# THUMB_TIP = HandLandmark.THUMB_TIP
# THUMB_JOINT = HandLandmark.THUMB_IP
# PINKY_BASE = HandLandmark.PINKY_MCP
# FINGERS = [
#     (HandLandmark.INDEX_FINGER_TIP, HandLandmark.INDEX_FINGER_PIP),
#     (HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_PIP),
#     (HandLandmark.RING_FINGER_TIP, HandLandmark.RING_FINGER_PIP),
#     (HandLandmark.PINKY_TIP, HandLandmark.PINKY_PIP),
# ]


def distance_to_pinky(landmarks, point):
    return math.dist(landmarks[point][1:], landmarks[PINKY_BASE][1:])


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
            # The thumb moves sideways: it is out when its tip is farther from the pinky than its joint.
            raised += distance_to_pinky(landmarks, THUMB_TIP) > distance_to_pinky(landmarks, THUMB_JOINT)
            cv.putText(frame, str(raised), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)
        cv.imshow("Finger Counter", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv.destroyAllWindows()
