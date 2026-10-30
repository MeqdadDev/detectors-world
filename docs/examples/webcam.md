# Webcam Viewer

Runs any detector on your webcam and shows how many things it found. This is the quickest way to try each detector.

## Run it

```bash
python examples/webcam.py hand
python examples/webcam.py pose
python examples/webcam.py face
python examples/webcam.py face_mesh --camera 1
```

## Code

```python title="examples/webcam.py"
--8<-- "examples/webcam.py"
```

## How it works

- `create_detector(args.detector)` lets the same loop work with every detector.
- `cv.flip(frame, 1)` makes the image behave like a mirror. If you use hand labels, read [Left and right](../detectors/hand.md#left-and-right).
- `len(found)` works for all detectors, because `detect()` always returns a list.
