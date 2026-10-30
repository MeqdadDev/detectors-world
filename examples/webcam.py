"""Run any detector on your webcam.

    python examples/webcam.py hand
    python examples/webcam.py pose --camera 1

Press q to quit.
"""

import argparse

import cv2 as cv

from detectors_world import create_detector

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("detector", choices=["hand", "pose", "face", "face_mesh"])
parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
args = parser.parse_args()

cap = cv.VideoCapture(args.camera)
with create_detector(args.detector) as detector:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv.flip(frame, 1)
        found = detector.detect(frame)
        cv.putText(frame, f"{args.detector}: {len(found)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Detectors World", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv.destroyAllWindows()
