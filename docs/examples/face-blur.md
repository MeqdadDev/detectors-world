# Face Blur

Blurs every face in the webcam image.

## Run it

```bash
python examples/face_blur.py
```

## Code

```python title="examples/face_blur.py"
--8<-- "examples/face_blur.py"
```

## How it works

- `draw=False` keeps the boxes off the image, and the program uses the positions itself.
- A box can start outside the image near an edge, so `max()` limits it before slicing the frame.
- `cv.GaussianBlur` blurs just that region, and the result is written back into the frame.

## Try next

- Increase the `(51, 51)` blur size for a stronger effect. Both numbers must be odd.
- Replace the blur with a solid box using `cv.rectangle`.
