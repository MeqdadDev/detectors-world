# Exercise Counter

Counts arm curls by measuring the angle at your left elbow. Stand so that your whole arm is visible.

## Run it

```bash
python examples/exercise_counter.py
```

## Code

```python title="examples/exercise_counter.py"
--8<-- "examples/exercise_counter.py"
```

## How it works

- Landmarks 11, 13 and 15 are the left shoulder, elbow and wrist, as the [pose landmark map](../detectors/pose.md#landmark-map) shows.
- `angle()` calculates the angle at the middle point. A straight arm is close to 180 degrees.
- A repetition counts when the arm goes below `BENT` and then above `STRAIGHT`. Using two limits stops small shakes from counting as extra repetitions.

## Try next

- Count the right arm with landmarks 12, 14 and 16.
- Count squats with the angle at the knee, using landmarks 23, 25 and 27.
