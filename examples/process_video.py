"""Run a detector on a video file and save a copy with the drawings.

    python examples/process_video.py pose dance.mp4
    python examples/process_video.py hand clip.mp4 --output tracked.mp4
"""

import argparse

import cv2 as cv

from detectors_world import create_detector

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("detector", choices=["hand", "pose", "face", "face_mesh"])
parser.add_argument("video", help="path to the video")
parser.add_argument("--output", default="result.mp4", help="where to save the result (default: result.mp4)")
args = parser.parse_args()

cap = cv.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"Could not open the video: {args.video}")

fps = cap.get(cv.CAP_PROP_FPS) or 30
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
writer = cv.VideoWriter(args.output, cv.VideoWriter_fourcc(*"mp4v"), fps, size)

frames = 0
with create_detector(args.detector) as detector:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detector.detect(frame)
        writer.write(frame)
        frames += 1

cap.release()
writer.release()
print(f"Processed {frames} frames, saved to {args.output}")
