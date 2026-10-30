"""Measure the distance between the thumb and index fingertips.

The bar fills as the fingers open, which makes a simple volume or brightness control.
Press q to quit.
"""

import math

import cv2 as cv

from detectors_world import create_detector

THUMB_TIP = 4
INDEX_TIP = 8
CLOSED, OPEN = 30, 220

cap = cv.VideoCapture(0)
with create_detector("hand", max_hands=1) as hand:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv.flip(frame, 1)
        hands = hand.detect(frame, draw=False)
        if hands:
            thumb, index = hands[0][THUMB_TIP], hands[0][INDEX_TIP]
            distance = math.dist(thumb, index)
            level = min(max((distance - CLOSED) / (OPEN - CLOSED), 0), 1)

            cv.line(frame, thumb, index, (0, 255, 0), 3)
            cv.rectangle(frame, (20, 40), (50, 240), (255, 255, 255), 2)
            cv.rectangle(frame, (20, 240 - int(200 * level)), (50, 240), (0, 255, 0), cv.FILLED)
            cv.putText(frame, f"{level:.0%}", (15, 275), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.imshow("Pinch Distance", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv.destroyAllWindows()
