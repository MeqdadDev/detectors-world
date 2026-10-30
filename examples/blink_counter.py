"""Count eye blinks from how open the eyes are.

An eye's openness is its height divided by its width. Press q to quit.
"""

import math

import cv2 as cv

from detectors_world import create_detector

# Two corners of the eye, then its upper and lower lid.
RIGHT_EYE = (33, 133, 159, 145)
LEFT_EYE = (362, 263, 386, 374)
CLOSED = 0.2


def openness(face, eye):
    corner_a, corner_b, upper, lower = eye
    return math.dist(face[upper], face[lower]) / math.dist(face[corner_a], face[corner_b])


blinks = 0
eyes_closed = False

cap = cv.VideoCapture(0)
with create_detector("face_mesh", max_faces=1) as face_mesh:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        faces = face_mesh.detect(frame, draw=False)
        if faces:
            average = (openness(faces[0], RIGHT_EYE) + openness(faces[0], LEFT_EYE)) / 2
            closed_now = average < CLOSED
            if closed_now and not eyes_closed:
                blinks += 1
            eyes_closed = closed_now
        cv.putText(frame, f"Blinks: {blinks}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        cv.imshow("Blink Counter", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv.destroyAllWindows()
