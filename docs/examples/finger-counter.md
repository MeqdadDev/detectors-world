# Finger Counter

Shows how many fingers are raised, from 0 to 5, on either hand. Hold your hand upright with your palm toward the camera.

## Run it

```bash
python examples/finger_counter.py
```

## Code

```python title="examples/finger_counter.py"
--8<-- "examples/finger_counter.py"
```

## How it works

Fingertips are landmarks 8, 12, 16 and 20, and each middle joint is two numbers below its tip, as the [hand landmark map](../detectors/hand.md#landmark-map) shows.

In images, `y` grows downward, so a raised fingertip has a smaller `y` than its joint. Each comparison gives `True` or `False`, and `sum()` counts the `True` results.

The thumb moves sideways, so height doesn't help. When it's out, its tip (4) is farther from the base of the pinky (17) than its joint (3) is. When it's folded across the palm, the tip is closer. This works for left and right hands, and on mirrored images.

## Landmark names

The numbers can be replaced with names. The comments in the code show how: uncomment the import, the three thumb and pinky names and the `FINGERS` list, then use the second `raised` line in place of the first.

## Try next

- Count both hands with `max_hands=2`, using `hand.locate(frame, hand=1)` for the second one.
