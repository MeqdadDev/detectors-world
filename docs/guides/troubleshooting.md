# Troubleshooting

## Installation

### `pip install` picks an old version, or fails

Detectors World 1.x needs Python 3.10 or newer on Windows, Linux, or macOS on Apple Silicon. Check your version:

```bash
python --version
```

On older Python versions or Intel Macs, pip can't install 1.x.

### `AttributeError: module 'mediapipe' has no attribute 'solutions'`

You have version 0.0.2, which doesn't work with current MediaPipe. Upgrade by following [Upgrading from 0.x](upgrading.md).

### `libGLESv2.so.2: cannot open shared object file`

Your Linux system is missing OpenGL libraries, which is common on servers and in Docker. [Install them](../getting-started/installation.md#linux-servers-and-docker).

### `import cv2` fails, or OpenCV behaves strangely

You may have two OpenCV packages installed. Check:

```bash
pip list
```

If both `opencv-python` and `opencv-contrib-python` appear, follow the steps in [Upgrading from 0.x](upgrading.md#step-1-replace-opencv).

## Camera

### `ValueError: frame is None`

The camera returned no image. Check that:

- No other program is using the camera, such as a video call.
- Your program has camera permission in your system's privacy settings.
- The camera number is right. Try `cv.VideoCapture(1)`.

Always check `ok` from `cap.read()` before using the frame:

```python
ok, frame = cap.read()
if not ok:
    print("Could not read from the camera")
```

### The camera is slow to open on Windows

Try the DirectShow backend:

```python
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
```

### The window doesn't respond to the keyboard

Click the video window first, so that it has focus. `cv.waitKey()` only reads keys pressed in an OpenCV window.

### The window freezes or stays empty

`cv.imshow()` needs a `cv.waitKey()` call after it to refresh the window.

## Detection

### Nothing is detected

- Check the lighting, and keep the whole hand, face or body in view.
- For Face, stay within about 2 meters of the camera.
- Lower `detection_confidence`.
- Make sure the frame isn't damaged, by showing it with `cv.imshow`.

### Left and right hands are swapped

The frame is probably mirrored. See [Left and right](../detectors/hand.md#left-and-right).

### `RuntimeError: Call detect(frame) before locate(frame)`

`locate()` reads the result of `detect()`, so call `detect()` first, once for every frame.

### `RuntimeError: This detector is closed`

The detector was used after its `with` block ended, or after `close()`. Keep your loop inside the `with` block.

### `ValueError: frame must be an 8-bit image`

Your image holds a different number type, often after calculations. Convert it:

```python
frame = frame.astype("uint8")
```

If its values run from 0.0 to 1.0, scale them first:

```python
frame = (frame * 255).astype("uint8")
```

### `TypeError: frame must be a NumPy array`

Detectors take NumPy images from OpenCV. Convert a PIL image, for example, and reorder its colors:

```python
import numpy as np

frame = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
```

## Models

### `Could not download the model`

The computer couldn't reach the download server. Check your connection, firewall and proxy. The error message shows the address. You can also download the file on another computer and follow [Use models offline](models.md#use-models-offline).

### `FileNotFoundError: Model file not found`

The path passed to `model_path` doesn't point to a file. Check the spelling and the folder you are running from.

## Log messages

Lines starting with `INFO`, `WARNING` or `W0000` come from MediaPipe when a model loads. They are normal.

## Still stuck?

[Open an issue](https://github.com/MeqdadDev/detectors-world/issues) with your Python version, operating system and the full error message.
