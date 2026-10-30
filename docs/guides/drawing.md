# Drawing

Detectors draw directly on the frame you pass in. This page covers how to control that, and how to draw your own.

## Turn drawing on or off

Both `detect()` and `locate()` take `draw`:

```python
hand.detect(frame)               # draws
hand.detect(frame, draw=False)   # only detects

hand.locate(frame)               # draws dots
hand.locate(frame, draw=False)   # only returns positions
```

`face_mesh.locate()` is the exception: it doesn't draw unless you pass `draw=True`.

## Colors

Colors follow OpenCV's order, `(blue, green, red)`, with each value from 0 to 255:

| Color | Value |
|---|---|
| Red | `(0, 0, 255)` |
| Green | `(0, 255, 0)` |
| Blue | `(255, 0, 0)` |
| Yellow | `(0, 255, 255)` |
| White | `(255, 255, 255)` |

## What each method accepts

| Method | Options |
|---|---|
| `hand.detect()`, `pose.detect()` | `draw`, `color` (lines), `point_color` (dots) |
| `hand.locate()`, `pose.locate()` | `draw`, `color`, `radius` |
| `face.detect()` | `draw`, `color` (box) |
| `face.locate()` | `draw`, `color` |
| `face_mesh.detect()` | `draw`, `color`, `thickness`, `radius` |
| `face_mesh.locate()` | `draw`, `show_ids`, `color`, `radius` |

```python
hand.detect(frame, color=(255, 255, 255), point_color=(0, 255, 255))
hand.locate(frame, color=(255, 0, 0), radius=4)
```

## Keep the original frame

Drawing changes the frame. To keep a clean copy, draw on a copy:

```python
preview = frame.copy()
hand.detect(preview)
```

## Draw your own

Results are plain pixel positions, so any OpenCV drawing function works with them:

```python
hands = hand.detect(frame, draw=False)

for found in hands:
    cv.circle(frame, found[8], 12, (0, 255, 255), cv.FILLED)
    cv.line(frame, found[4], found[8], (0, 255, 0), 3)
```

## Show text next to a landmark

```python
for found in hand.detect(frame):
    cv.putText(frame, found.label, found[0], cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```
