# Core Concepts

All four detectors work the same way, so what you learn here applies to each of them.

## Creating a detector

`create_detector()` takes a name and any options:

```python
from detectors_world import create_detector

hand = create_detector("hand", max_hands=1)
```

You can also use the classes directly, which gives you the same object:

```python
from detectors_world import Hand

hand = Hand(max_hands=1)
```

| Name | Class | Also accepted |
|---|---|---|
| `"hand"` | [`Hand`](../reference/api.md#detectors_world.Hand) | `"hands"` |
| `"pose"` | [`Pose`](../reference/api.md#detectors_world.Pose) | |
| `"face"` | [`Face`](../reference/api.md#detectors_world.Face) | |
| `"face_mesh"` | [`FaceMesh`](../reference/api.md#detectors_world.FaceMesh) | `"facemesh"`, `"face-mesh"` |

Names ignore capital letters.

## detect() and locate()

Every detector has two main methods:

- **`detect(frame)`** runs the model on a frame. Call it once for every frame.
- **`locate(frame)`** reads the latest `detect()` result and gives it to you as plain lists, drawing markers if you ask.

Calling `locate()` before `detect()` raises a `RuntimeError`, because there is nothing to read yet.

```python
hands = hand.detect(frame)       # objects with names: hands[0].points, hands[0].label
landmarks = hand.locate(frame)   # plain lists: [[id, x, y], ...]
```

Use whichever you find easier. `detect()` gives every hand at once, and `locate()` gives one hand as numbered rows.

## Frames

A frame is the NumPy image that OpenCV gives you, from `cap.read()` or `cv.imread()`. Detectors accept three kinds:

| Frame | Shape |
|---|---|
| BGR color, the OpenCV default | `(height, width, 3)` |
| BGRA color | `(height, width, 4)` |
| Grayscale | `(height, width)` |

Frames must be 8-bit (`dtype` `uint8`). You don't need to convert to RGB, because detectors do that themselves.

## Drawing

`detect()` draws on the frame you pass in, so your frame changes and you don't need to reassign it:

```python
hand.detect(frame)               # draws, then frame is ready to show
hand.detect(frame, draw=False)   # only detects
```

See the [Drawing guide](../guides/drawing.md) for colors and sizes.

## Results

`detect()` always returns a list, with one item for each thing found. An empty list means nothing was found:

```python
hands = hand.detect(frame)

if hands:
    print(f"Found {len(hands)} hands")
```

### Landmarks

The hand, pose and face mesh detectors return `Landmarks` objects:

| Attribute | Meaning |
|---|---|
| `points` | `(x, y)` pixel position of each landmark |
| `normalized` | `(x, y, z)` for each landmark, with `x` and `y` from 0.0 to 1.0 across the image |
| `label` | `"Left"` or `"Right"` for hands, otherwise `None` |
| `score` | Confidence of `label`, otherwise `None` |

You can index a `Landmarks` object directly and take its length:

```python
first = hands[0]

first[8] == first.points[8]   # True
len(first)                    # 21 for a hand
```

Pixel positions can fall slightly outside the image when part of a hand or body is out of view.

### FaceBox

The face detector returns `FaceBox` objects:

| Attribute | Meaning |
|---|---|
| `x`, `y` | Top-left corner in pixels |
| `width`, `height` | Size in pixels |
| `score` | Confidence from 0.0 to 1.0 |
| `keypoints` | `(x, y)` pixel positions of the 6 key points |
| `box` | The box as `(x, y, width, height)` |

## Releasing the model

A detector holds a loaded model until you release it. A `with` block does this for you:

```python
with create_detector("hand") as hand:
    hands = hand.detect(frame)
```

If you can't use a `with` block, call `close()`:

```python
hand = create_detector("hand")
hands = hand.detect(frame)
hand.close()
```

A closed detector can't detect anymore, and trying raises a `RuntimeError`.

## Video or still images

By default, detectors treat frames as a video, following things from one frame to the next. For unrelated photos, pass `static_image=True`:

```python
hand = create_detector("hand", static_image=True)
```

The [Photos and Video Files guide](../guides/images-and-video.md) covers this in more detail.
