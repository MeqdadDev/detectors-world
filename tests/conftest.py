import urllib.request
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

# Sample images published by the MediaPipe project, fetched once and cached locally.
IMAGES = {
    "hands": "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg",
    "person": "https://storage.googleapis.com/mediapipe-assets/business-person.png",
    "pose": "https://storage.googleapis.com/mediapipe-assets/pose.jpg",
}
IMAGE_DIR = Path(__file__).parent / ".cache"


@pytest.fixture(scope="session")
def image():
    def load(name: str) -> np.ndarray:
        url = IMAGES[name]
        path = IMAGE_DIR / url.rsplit("/", 1)[-1]
        if not path.is_file():
            IMAGE_DIR.mkdir(exist_ok=True)
            try:
                urllib.request.urlretrieve(url, path)
            except OSError as err:
                pytest.skip(f"could not download sample image {url}: {err}")
        frame = cv.imread(str(path))
        assert frame is not None
        return frame

    return load


@pytest.fixture
def blank():
    return np.zeros((480, 640, 3), dtype=np.uint8)
