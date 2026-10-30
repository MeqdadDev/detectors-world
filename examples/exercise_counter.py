"""Count arm curls from the angle at the elbow.

Face the camera so your whole arm is visible. Press q to quit.
"""

import math

import cv2 as cv

from detectors_world import create_detector

SHOULDER, ELBOW, WRIST = 11, 13, 15
BENT, STRAIGHT = 50, 150


def angle(a, b, c):
    """The angle at point b, in degrees."""
    first = math.atan2(a[1] - b[1], a[0] - b[0])
    second = math.atan2(c[1] - b[1], c[0] - b[0])
    degrees = abs(math.degrees(first - second))
    return 360 - degrees if degrees > 180 else degrees


count = 0
was_bent = False

cap = cv.VideoCapture(0)
with create_detector("pose") as pose:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        bodies = pose.detect(frame)
        if bodies:
            body = bodies[0]
            elbow_angle = angle(body[SHOULDER], body[ELBOW], body[WRIST])
            if elbow_angle < BENT:
                was_bent = True
            elif elbow_angle > STRAIGHT and was_bent:
                was_bent = False
                count += 1
            cv.putText(frame, f"{elbow_angle:.0f} deg", body[ELBOW], cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, f"Reps: {count}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        cv.imshow("Exercise Counter", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv.destroyAllWindows()
