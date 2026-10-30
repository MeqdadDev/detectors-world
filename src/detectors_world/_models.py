from __future__ import annotations

import hashlib
import http.client
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

_BASE_URL = "https://storage.googleapis.com/mediapipe-models"

# Pinned to version 1 with hashes: the "latest" URLs can change without notice.
MODELS: dict[str, tuple[str, str]] = {
    "hand_landmarker.task": (
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    "pose_landmarker_lite.task": (
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
    "pose_landmarker_full.task": (
        "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
    ),
    "pose_landmarker_heavy.task": (
        "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
    ),
    "blaze_face_short_range.tflite": (
        "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    ),
    "face_landmarker.task": (
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
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


def _is_intact(filename: str, data: bytes) -> bool:
    return hashlib.sha256(data).hexdigest() == MODELS[filename][1]


def _download(filename: str, target: Path) -> bytes:
    url = f"{_BASE_URL}/{MODELS[filename][0]}"
    advice = (
        f"Download it manually and pass model_path='path/to/{filename}', "
        f"or place it in {target.parent}."
    )
    print(f"detectors_world: downloading {filename} (first run only)...", file=sys.stderr)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except (OSError, http.client.HTTPException) as err:
        raise RuntimeError(f"Could not download the model {filename} from {url}: {err}\n{advice}") from err
    if not _is_intact(filename, data):
        raise RuntimeError(f"The download of {filename} from {url} is damaged or incomplete.\n{advice}")
    return data


def _store(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        os.replace(tmp_name, target)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def load_model(filename: str, model_path: str | os.PathLike[str] | None = None) -> bytes:
    """Return a model's contents, downloading it on first use.

    Bytes rather than a path are handed to MediaPipe because it cannot open
    paths with non-ASCII characters on Windows, such as a user folder name.
    """
    if model_path is not None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path.read_bytes()

    target = cache_dir() / filename
    if target.is_file():
        data = target.read_bytes()
        if _is_intact(filename, data):
            return data

    data = _download(filename, target)
    _store(target, data)
    return data
