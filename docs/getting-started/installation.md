# Installation

## Requirements

- Python 3.10 or newer
- Windows, Linux, or macOS on Apple Silicon (Intel Macs are not supported)
- A webcam, if you want to run the live examples

## Install

We recommend a virtual environment, so the package and its dependencies stay separate from your other projects.

=== "Windows"

    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    pip install detectors-world
    ```

=== "Linux and macOS"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install detectors-world
    ```

This also installs MediaPipe, NumPy and OpenCV, which you can import as `cv2`.

## Check the installation

```bash
python -c "import detectors_world; print(detectors_world.__version__)"
```

This prints the installed version.

## First run

Each detector downloads its model the first time you create it, so your first run needs an internet connection. Later runs use the saved copy. See [Models and Offline Use](../guides/models.md) to control where models are stored.

MediaPipe prints a few `INFO` and `WARNING` lines when a model loads. These are normal and can be ignored.

## Linux servers and Docker

Systems without a desktop can be missing the OpenGL libraries that MediaPipe and OpenCV need. If you see `libGLESv2.so.2: cannot open shared object file`, install them on Debian or Ubuntu:

```bash
sudo apt install libgl1 libegl1 libgles2
```

## Next step

Continue to the [Quick Start](quickstart.md).
