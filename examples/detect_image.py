"""Run a detector on a photo and save the result.

    python examples/detect_image.py hand photo.jpg
    python examples/detect_image.py face group.jpg --output faces.jpg
"""

import argparse

import cv2 as cv

from detectors_world import create_detector

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("detector", choices=["hand", "pose", "face", "face_mesh"])
parser.add_argument("image", help="path to the photo")
parser.add_argument("--output", default="result.jpg", help="where to save the result (default: result.jpg)")
args = parser.parse_args()

frame = cv.imread(args.image)
if frame is None:
    raise SystemExit(f"Could not read the image: {args.image}")

with create_detector(args.detector, static_image=True) as detector:
    found = detector.detect(frame)

cv.imwrite(args.output, frame)
print(f"Found {len(found)}, saved to {args.output}")
