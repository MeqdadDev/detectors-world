# Speed and Accuracy

These adjustments help when a detector runs too slowly, or misses things.

## Make it faster

### Use smaller frames

Most of the time goes into the model, and smaller frames make your whole program lighter. Reduce the frame before detecting:

```python
small = cv.resize(frame, (640, 360))
hand.detect(small)
```

Results are positions in the frame you passed in, so draw on that same small frame.

### Look for less

Search only for what you need:

```python
hand = create_detector("hand", max_hands=1)
face_mesh = create_detector("face_mesh", max_faces=1)
```

### Choose a lighter detector or model

- Use [Face](../detectors/face.md) instead of Face Mesh if boxes are enough.
- Use `model="lite"` for [Pose](../detectors/pose.md#choosing-a-model).

### Skip drawing

Drawing adds a little time for each frame, most of all for Face Mesh:

```python
faces = face_mesh.detect(frame, draw=False)
```

### Create one detector

Creating a detector loads a model, which is slow. Create it once before your loop, not inside it.

## Make it more accurate

### Improve the image

Detection depends on what the camera sees. Good, even light helps most, followed by a plain background and keeping the whole hand, face or body in view.

### Use a larger model

```python
pose = create_detector("pose", model="heavy")
```

### Tune the confidence

| Problem | Try |
|---|---|
| False detections | Raise `detection_confidence`, for example to `0.7` |
| Misses real ones | Lower `detection_confidence`, for example to `0.3` |
| Tracking keeps getting lost | Lower `tracking_confidence` |

```python
hand = create_detector("hand", detection_confidence=0.7)
```

## Small movements

Landmarks shift slightly from frame to frame, even when nothing moves. If a project reacts to tiny changes, give it a margin. For example, the [Exercise Counter](../examples/exercise-counter.md) waits for clearly bent and clearly straight positions, instead of using a single angle.
