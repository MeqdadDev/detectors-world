# Quick Start

This page takes you from an empty file to a live hand tracker.

## Your first detector

Save this as `hand_demo.py`:

```python
import cv2 as cv
from detectors_world import create_detector

cap = cv.VideoCapture(0)

with create_detector("hand") as hand:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hands = hand.detect(frame)
        cv.imshow("Hand Detection", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv.destroyAllWindows()
```

Run it:

```bash
python hand_demo.py
```

Hold your hand up to the camera and the landmarks appear on it. To quit, click the video window and press ++q++.

## What each part does

| Code | Purpose |
|---|---|
| `cv.VideoCapture(0)` | Opens the first camera. |
| `create_detector("hand")` | Creates the hand detector and loads its model. |
| `with ... as hand:` | Releases the model when the block ends. |
| `cap.read()` | Reads one frame. `ok` is `False` if that failed. |
| `hand.detect(frame)` | Finds hands, draws them on `frame`, and returns them. |
| `cv.imshow(...)` | Shows the frame in a window. |
| `cv.waitKey(1)` | Refreshes the window and reads the keyboard. |

## Use what it found

`detect()` returns a list with one item per hand, so you can act on the result. Landmark 8 is the tip of the index finger:

```python
hands = hand.detect(frame)

for found in hands:
    x, y = found[8]
    print(found.label, "index fingertip at", x, y)
```

The [hand landmark map](../detectors/hand.md#landmark-map) shows what every number means.

## Try another detector

Change the name, and the same program runs a different detector:

```python
detector = create_detector("pose")       # or "face", or "face_mesh"
```

## Pass options

Options go straight into `create_detector()`:

```python
hand = create_detector("hand", max_hands=1, detection_confidence=0.7)
```

Each detector's page lists the options it accepts.

## Next steps

- Learn how results work in [Core Concepts](concepts.md).
- See what each detector offers in [Detectors](../detectors/index.md).
- Build something from the [Examples](../examples/index.md).
