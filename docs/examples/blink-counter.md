# Blink Counter

Counts your blinks by measuring how open your eyes are.

## Run it

```bash
python examples/blink_counter.py
```

## Code

```python title="examples/blink_counter.py"
--8<-- "examples/blink_counter.py"
```

## How it works

- Each eye has two corner landmarks and two eyelid landmarks, listed on the [face mesh page](../detectors/face-mesh.md#landmark-map).
- `openness()` divides the eye's height by its width. Using a ratio means the result stays the same as you move closer or farther.
- `eyes_closed` makes sure one blink counts once, however many frames it lasts.

## Try next

- `CLOSED` works for most people, but adjust it if blinks are missed or counted by mistake.
- Print the `average` value while you blink to see the range for your eyes.
