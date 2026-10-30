"""Blur every face in the webcam image.

Press q to quit.
"""

import cv2 as cv

from detectors_world import create_detector

cap = cv.VideoCapture(0)
with create_detector("face") as face:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        for found in face.detect(frame, draw=False):
            # Boxes can start outside the image when a face is near the edge.
            x, y = max(found.x, 0), max(found.y, 0)
            region = frame[y : found.y + found.height, x : found.x + found.width]
            if region.size:
                region[:] = cv.GaussianBlur(region, (51, 51), 0)
        cv.imshow("Face Blur", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv.destroyAllWindows()
