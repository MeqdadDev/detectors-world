# Photo Detection

Runs a detector on a photo and saves a copy with the drawings.

## Run it

```bash
python examples/detect_image.py hand photo.jpg
python examples/detect_image.py face group.jpg --output faces.jpg
```

## Code

```python title="examples/detect_image.py"
--8<-- "examples/detect_image.py"
```

## How it works

- `static_image=True` treats the photo on its own, as there are no earlier frames to follow.
- `cv.imread()` returns `None` when it can't read a file, so the program checks that first.
- `detect()` draws on `frame`, so saving `frame` saves the result.
