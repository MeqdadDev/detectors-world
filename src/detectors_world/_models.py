from __future__ import annotations

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

_BASE_URL = "https://storage.googleapis.com/mediapipe-models"

MODELS: dict[str, str] = {
    "hand_landmarker.task": "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "blaze_face_short_range.tflite": "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "face_landmarker.task": "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def cache_dir() -> Path:
    """Folder where downloaded model files are kept.

    Set the ``DETECTORS_WORLD_CACHE`` environment variable to use another folder.
    """
    override = os.environ.get("DETECTORS_WORLD_CACHE")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "detectors_world"


def resolve_model(filename: str, model_path: str | os.PathLike[str] | None = None) -> str:
    """Return a local path to a model file, downloading it on first use."""
    if model_path is not None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return str(path)

    target = cache_dir() / filename
    if target.is_file():
        return str(target)

    url = f"{_BASE_URL}/{MODELS[filename]}"
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"detectors_world: downloading {filename} (first run only)...", file=sys.stderr)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as tmp, urllib.request.urlopen(url, timeout=60) as response:
            while chunk := response.read(1 << 16):
                tmp.write(chunk)
        os.replace(tmp_name, target)
    except OSError as err:
        Path(tmp_name).unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the model {filename} from {url}: {err}\n"
            f"Download it manually and pass model_path='path/to/{filename}', "
            f"or place it in {target.parent}."
        ) from err
    return str(target)
