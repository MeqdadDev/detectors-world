# Photos and Video Files

Detectors work on any image, not only a webcam.

## Photos

Read the image with OpenCV, and set `static_image=True` so that each image is treated on its own:

```python
import cv2 as cv
from detectors_world import create_detector

frame = cv.imread("photo.jpg")

with create_detector("hand", static_image=True) as hand:
    hands = hand.detect(frame)

cv.imwrite("result.jpg", frame)
print(f"Found {len(hands)} hands")
```

`cv.imread()` returns `None` when it can't read a file. Passing that to a detector raises a `ValueError`.

For a complete program with command line options, see [Photo Detection](../examples/photo.md).

## A folder of photos

Reuse one detector, because creating a detector loads its model each time:

```python
from pathlib import Path

with create_detector("face", static_image=True) as face:
    for path in Path("photos").glob("*.jpg"):
        frame = cv.imread(str(path))
        print(path.name, len(face.detect(frame, draw=False)))
```

## Video files

Open the file with `cv.VideoCapture`, and leave `static_image` at its default so that the detector follows things between frames:

```python
cap = cv.VideoCapture("clip.mp4")

with create_detector("pose") as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bodies = pose.detect(frame)

cap.release()
```

The [Video File Processing](../examples/video.md) example also saves the result to a new file.

## When to use static_image

| Your frames | Use |
|---|---|
| Webcam, or a video file | `static_image=False` (default) |
| Unrelated photos | `static_image=True` |

In video mode, the model follows what it found in the previous frame, which is smoother and faster. On unrelated photos that gets in the way, so `static_image=True` searches each image from scratch.

## Detecting several times on one frame

Each `detect()` call runs the model again. Call it once per frame and reuse the result, rather than calling it twice:

```python
hands = hand.detect(frame)
first = hand.locate(frame, hand=0)
second = hand.locate(frame, hand=1)
```
