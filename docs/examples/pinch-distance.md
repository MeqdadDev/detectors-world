# Pinch Distance

Measures the distance between your thumb and index fingertips and shows it as a level from 0 to 100%. This makes a simple volume or brightness control.

## Run it

```bash
python examples/pinch_distance.py
```

## Code

```python title="examples/pinch_distance.py"
--8<-- "examples/pinch_distance.py"
```

## How it works

- `hands[0][THUMB_TIP]` gives a landmark's `(x, y)` position directly.
- `math.dist()` calculates the straight-line distance in pixels.
- `level` turns that distance into a value from 0 to 1, using `CLOSED` and `OPEN` as the range.

## Try next

- The distance in pixels changes as your hand moves toward the camera. Divide it by the distance between landmarks 0 and 9 to get a value that doesn't change.
- Adjust `CLOSED` and `OPEN` to suit your camera.
