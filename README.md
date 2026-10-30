# Detectors World
[![PyPI](https://img.shields.io/pypi/v/detectors-world)](https://pypi.org/project/detectors-world/)
[![Python](https://img.shields.io/pypi/pyversions/detectors-world)](https://pypi.org/project/detectors-world/)
[![Tests](https://github.com/MeqdadDev/detectors-world/actions/workflows/tests.yml/badge.svg)](https://github.com/MeqdadDev/detectors-world/actions/workflows/tests.yml)
[![Downloads](https://static.pepy.tech/badge/detectors-world)](https://pepy.tech/project/detectors-world)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

<p align="center">
<img alt="Detectors World Logo" src="https://raw.githubusercontent.com/MeqdadDev/detectors-world/main/Detectors-World-Logo.png" width="50%">
</p>

Detectors World is a Python package for building computer vision applications in an easy and handy way.
It wraps Google's [MediaPipe](https://ai.google.dev/edge/mediapipe) models behind a small, friendly API that works directly with OpenCV images.

Developed by [@MeqdadDev](https://www.github.com/MeqdadDev)

## Available Detectors

| Name | What it finds | `detect()` returns |
|---|---|---|
| `"hand"` | Hands, 21 landmarks each, plus left/right | `list[Landmarks]` |
| `"pose"` | Bodies, 33 landmarks each | `list[Landmarks]` |
| `"face"` | Face boxes and 6 key points | `list[FaceBox]` |
| `"face_mesh"` | Faces, 478 landmarks each | `list[Landmarks]` |

## Installation

```bash
pip install detectors-world
```

Requires Python 3.10 or newer on Windows, Linux, or macOS (Apple Silicon).
Models are downloaded automatically the first time each detector is created and cached for later runs.

## Quick Start

```python
import cv2 as cv
from detectors_world import create_detector

cap = cv.VideoCapture(0)

with create_detector("hand") as hand:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hands = hand.detect(frame)  # draws on frame and returns what it found
        cv.imshow("Hand Detection", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

cap.release()
cv.destroyAllWindows()
```

Using the detector in a `with` block releases the model when you are done. You can also call `hand.close()` yourself.

## Hand Detector ✋

<p align="center">
<img src="https://raw.githubusercontent.com/MeqdadDev/detectors-world/main/assets/hand-detector.gif" width="50%">
</p>

```python
hand = create_detector("hand", max_hands=2, detection_confidence=0.5)

hands = hand.detect(frame)
for h in hands:
    print(h.label, h.points[8])  # "Left"/"Right" and the index fingertip (x, y)

landmarks = hand.locate(frame, hand=0)  # [[id, x, y], ...] for the first hand, [] if none
```

Landmark numbers follow MediaPipe's [hand landmark map](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models): 0 is the wrist, 4 the thumb tip, 8 the index fingertip, and so on.
See [examples/finger_counter.py](examples/finger_counter.py) for a complete program that uses them.

## Pose Detector 💪

<p align="center">
<img src="https://raw.githubusercontent.com/MeqdadDev/detectors-world/main/assets/pose-detection.gif" width="50%">
</p>

```python
pose = create_detector("pose", model="lite")  # "lite", "full" or "heavy"

bodies = pose.detect(frame)
landmarks = pose.locate(frame)  # [[id, x, y], ...] for 33 body landmarks
```

## Face Detector 😃

<p align="center">
<img src="https://raw.githubusercontent.com/MeqdadDev/detectors-world/main/assets/face-detection.gif" width="50%">
</p>

```python
face = create_detector("face")

faces = face.detect(frame, draw=False)
for f in faces:
    print(f.box, f"{f.score:.0%}")  # (x, y, width, height) and confidence

boxes = face.locate(frame)  # draws framed boxes with the score: [[id, (x, y, w, h), score], ...]
```

## Face Mesh Detector 🤖

<p align="center">
<img src="https://raw.githubusercontent.com/MeqdadDev/detectors-world/main/assets/facemesh-detection.gif" width="50%">
</p>

```python
face_mesh = create_detector("face_mesh", max_faces=1)

faces = face_mesh.detect(frame)  # draws eyes, brows, lips and face outline
landmarks = face_mesh.locate(frame, show_ids=True)  # writes each landmark's number on the frame
```

## Options

Every detector accepts these options in `create_detector()`:

| Option | Default | Meaning |
|---|---|---|
| `detection_confidence` | `0.5` | Minimum confidence to report something new |
| `tracking_confidence` | `0.5` | Minimum confidence to keep tracking between frames (not used by `"face"`) |
| `static_image` | `False` | Set to `True` when processing unrelated photos instead of video frames |
| `model_path` | `None` | Use a model file you already downloaded, for offline use |

`detect()` and `locate()` take `draw=False` to skip drawing, and `color=(b, g, r)` to change the drawing color.

Downloaded models are stored in `%LOCALAPPDATA%\detectors_world` on Windows and `~/.cache/detectors_world` elsewhere.
Set the `DETECTORS_WORLD_CACHE` environment variable to use another folder.

## Examples

```bash
python examples/webcam.py hand
python examples/webcam.py face_mesh --camera 1
python examples/finger_counter.py
```

## Upgrading from 0.x

Code written for 0.x keeps working and shows a `DeprecationWarning` that names the new spelling:

| 0.x | 1.x |
|---|---|
| `DetectorCreator().getDetector("hand")` | `create_detector("hand")` |
| `drawOnHand=`, `drawOnFace=`, `drawOnPose=` | `draw=` |
| `handsMax=`, `facesMax=` | `max_hands=`, `max_faces=` |
| `detectionConfidence=`, `trackingConfidence=` | `detection_confidence=`, `tracking_confidence=` |
| `drawingColor=`, `drawingSize=` | `color=`, `radius=` |
| `locate(..., handsNumber=1)` (counts from 1) | `locate(..., hand=0)` (counts from 0) |

The old names will be removed in 2.0. See [CHANGELOG.md](CHANGELOG.md) for the full list of changes.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

To set up a development environment and run the tests:

```bash
pip install -e ".[test]"
pytest
```

## 🔗 Find me on
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/meqdad-darwish/)

[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/MeqdadDev)
