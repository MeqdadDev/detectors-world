"""Detectors World: hand, pose, face and face mesh detection in a few lines of Python.

Example:
    >>> import cv2 as cv
    >>> from detectors_world import create_detector
    >>> cap = cv.VideoCapture(0)
    >>> with create_detector("hand") as hand:
    ...     while cv.waitKey(1) & 0xFF != ord("q"):
    ...         ok, frame = cap.read()
    ...         hands = hand.detect(frame)
    ...         cv.imshow("Hand Detection", frame)
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, overload

from ._models import cache_dir
from .detectors import Detector, Face, FaceMesh, Hand, Pose
from .results import FaceBox, Landmarks

__version__ = "1.0.0"
__author__ = "Meqdad Darwish"

__all__ = [
    "Detector",
    "DetectorCreator",
    "Face",
    "FaceBox",
    "FaceMesh",
    "Hand",
    "Landmarks",
    "Pose",
    "cache_dir",
    "create_detector",
]

_DETECTORS: dict[str, type[Detector]] = {
    "hand": Hand,
    "hands": Hand,
    "pose": Pose,
    "face": Face,
    "facemesh": FaceMesh,
    "face_mesh": FaceMesh,
}


@overload
def create_detector(kind: Literal["hand", "hands"], **options: Any) -> Hand: ...
@overload
def create_detector(kind: Literal["pose"], **options: Any) -> Pose: ...
@overload
def create_detector(kind: Literal["face"], **options: Any) -> Face: ...
@overload
def create_detector(kind: Literal["facemesh", "face_mesh"], **options: Any) -> FaceMesh: ...
@overload
def create_detector(kind: str, **options: Any) -> Detector: ...
def create_detector(kind: str, **options: Any) -> Detector:
    """Create a detector by name.

    Args:
        kind: ``"hand"``, ``"pose"``, ``"face"`` or ``"face_mesh"``.
        **options: Passed to the detector class, for example ``max_hands=1``.
            See :class:`Hand`, :class:`Pose`, :class:`Face` and :class:`FaceMesh`.

    Raises:
        ValueError: If ``kind`` is not a known detector.
    """
    key = kind.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _DETECTORS:
        names = ", ".join(repr(name) for name in ("hand", "pose", "face", "face_mesh"))
        raise ValueError(f"Unknown detector {kind!r}. Choose one of: {names}.")
    return _DETECTORS[key](**options)


class DetectorCreator:
    """Deprecated: use :func:`create_detector` instead. Kept so 0.x code keeps running."""

    def __init__(self) -> None:
        warnings.warn(
            "DetectorCreator is deprecated, use detectors_world.create_detector() instead. "
            "It will be removed in detectors_world 2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._detectors: dict[str, Detector] = {}

    def getDetector(self, detectorType: str) -> Detector:  # noqa: N802, N803
        key = detectorType.strip().lower()
        if key not in self._detectors:
            self._detectors[key] = create_detector(key)
        return self._detectors[key]
