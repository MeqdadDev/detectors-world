# Upgrading from 0.x

Version 1.0 was rebuilt because MediaPipe removed the API that 0.x depended on. Most 0.x code keeps running, with warnings that tell you what to change.

## Step 1: replace OpenCV

0.x installed `opencv-python`, while 1.x uses the `opencv-contrib-python` build that MediaPipe requires. Remove the old one, so that only one copy of `cv2` is installed:

```bash
pip uninstall opencv-python
pip install --upgrade --force-reinstall detectors-world
```

## Step 2: run your code

Old names and positional arguments still work, and show a `DeprecationWarning` naming the new spelling. They will be removed in 2.0.

To see warnings from every file in your project, not only the main script, run Python with `-W default`:

```bash
python -W default your_script.py
```

## Step 3: check return values

These changed, and need an update in your code:

| Method | 0.x returned | 1.x returns |
|---|---|---|
| `detect()` | The frame | The results |
| `face.locate()` | `(frame, boxes)` when drawing | `boxes` |
| `face_mesh.locate()` | `(frame, all_faces)` | One face as `[[id, x, y], ...]` |

`detect()` still draws on the frame you pass in, so remove the assignment:

=== "Before"

    ```python
    img = hand.detect(img)
    ```

=== "After"

    ```python
    hand.detect(img)
    ```

## Name changes

| 0.x | 1.x |
|---|---|
| `DetectorCreator().getDetector("hand")` | `create_detector("hand")` |
| `sourceFrame=` | `frame=` |
| `drawOnHand=`, `drawOnFace=`, `drawOnPose=` | `draw=` |
| `handsMax=`, `facesMax=` | `max_hands=`, `max_faces=` |
| `detectionConfidence=`, `trackingConfidence=` | `detection_confidence=`, `tracking_confidence=` |
| `drawingColor=`, `drawingSize=` | `color=`, `radius=` |
| `drawingThickness=`, `drawingCircleRadius=` | `thickness=`, `radius=` |
| `locate(handsNumber=1)`, counting from 1 | `locate(hand=0)`, counting from 0 |

The `upperBody` and `faceDistanceLevel` options have no replacement. Passing them shows a warning and has no effect.

## Behavior changes

- `locate()` needs a `detect()` call first, and raises a clear `RuntimeError` if there wasn't one.
- `hand.locate()` returns `[]` when the requested hand isn't visible. It used to fall back to the first hand.
- `face_mesh.locate()` only draws when you ask it to.
- Settings such as `handsMax` were ignored in 0.x, and now take effect.
- Frames are now tracked as video, which is smoother. Use `static_image=True` for unrelated photos.

See the [changelog](../project/changelog.md) for the full list.
