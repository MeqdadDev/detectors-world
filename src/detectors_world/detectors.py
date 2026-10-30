from __future__ import annotations

import functools
import os
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple, TypeVar, cast

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ._models import load_model
from .results import FaceBox, Landmarks

Color = tuple[int, int, int]
PathLike = str | os.PathLike[str]

_F = TypeVar("_F", bound=Callable[..., Any])
_D = TypeVar("_D", bound="Detector")

_POSE_MODELS = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}


class _Renamed(NamedTuple):
    name: str
    convert: Callable[[Any], Any]
    note: str


_LegacyRule = str | _Renamed | None


def _deprecated(old: str, new: str, note: str = "") -> str:
    detail = f" ({note})" if note else ""
    return f"'{old}' is deprecated, use '{new}' instead{detail}. It will be removed in detectors_world 2.0."


def _accepts_legacy(names: Mapping[str, _LegacyRule]) -> Callable[[_F], _F]:
    """Accept 0.x keyword and positional arguments, translating them with a DeprecationWarning.

    Args:
        names: Each 0.x parameter, in its 0.x positional order, mapped to its new
            name, to a ``_Renamed`` rule, or to ``None`` if it no longer has an effect.
    """

    def decorate(func: _F) -> _F:
        leading = func.__code__.co_argcount - 1

        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if leading and "sourceFrame" in kwargs:
                warnings.warn(_deprecated("sourceFrame", "frame"), DeprecationWarning, stacklevel=2)
                args = (kwargs.pop("sourceFrame"), *args)
            extra = args[leading:]
            if len(extra) > len(names):
                raise TypeError(f"{func.__qualname__}() got too many positional arguments")
            supplied = dict(zip(names, extra))
            supplied.update({old: kwargs.pop(old) for old in names if old in kwargs})
            for old, value in supplied.items():
                rule = names[old]
                if rule is None:
                    message = f"'{old}' has no effect since detectors_world 1.0 and will be removed in 2.0."
                elif isinstance(rule, _Renamed):
                    message = _deprecated(old, rule.name, rule.note)
                    kwargs[rule.name] = rule.convert(value)
                else:
                    message = _deprecated(old, rule)
                    kwargs[rule] = value
                warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(self, *args[:leading], **kwargs)

        return cast(_F, wrapper)

    return decorate


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError(
            "frame is None. If it came from cv.VideoCapture.read(), the camera returned no image: "
            "check the camera index and that no other program is using the camera."
        )
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be a NumPy array from OpenCV, got {type(frame).__name__}.")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame must be an 8-bit image (dtype uint8), got dtype {frame.dtype}.")
    if frame.ndim == 2:
        return cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv.cvtColor(frame, cv.COLOR_BGRA2RGB)
    raise ValueError(f"frame must be a grayscale, BGR or BGRA image, got shape {frame.shape}.")


def _to_pixels(points: Iterable[Any], shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    height, width = shape[:2]
    return tuple((int(point.x * width), int(point.y * height)) for point in points)


def _to_landmarks(
    landmarks: Sequence[Any], shape: tuple[int, ...], label: str | None = None, score: float | None = None
) -> Landmarks:
    return Landmarks(
        points=_to_pixels(landmarks, shape),
        normalized=tuple((lm.x, lm.y, lm.z) for lm in landmarks),
        label=label,
        score=score,
    )


def _draw_connections(
    frame: np.ndarray, points: Sequence[tuple[int, int]], connections: Iterable[Any], color: Color, thickness: int
) -> None:
    for connection in connections:
        cv.line(frame, points[connection.start], points[connection.end], color, thickness, cv.LINE_AA)


def _draw_points(
    frame: np.ndarray, points: Iterable[tuple[int, int]], color: Color, radius: int, thickness: int = cv.FILLED
) -> None:
    for point in points:
        cv.circle(frame, point, radius, color, thickness, cv.LINE_AA)


class Detector(ABC):
    """Base class for all detectors.

    Detectors hold a loaded model. Use them as a context manager, or call
    :meth:`close` when you are done, to release it.
    """

    _last: list[Any] | None = None

    def __init__(self, model: bytes, static_image: bool) -> None:
        self._static_image = static_image
        self._last_timestamp_ms = -1
        self._closed = False
        self._task = self._create_task(
            BaseOptions(model_asset_buffer=model),
            vision.RunningMode.IMAGE if static_image else vision.RunningMode.VIDEO,
        )

    @abstractmethod
    def _create_task(self, base_options: BaseOptions, running_mode: vision.RunningMode) -> Any: ...

    @abstractmethod
    def detect(self, frame: np.ndarray, *, draw: bool = True) -> list[Any]:
        """Run the model on ``frame`` (a BGR image from OpenCV) and return what was found."""

    @abstractmethod
    def locate(self, frame: np.ndarray) -> list[Any]:
        """Return pixel positions from the latest :meth:`detect` call."""

    def _run(self, frame: np.ndarray) -> Any:
        if self._closed:
            raise RuntimeError("This detector is closed. Create a new one to keep detecting.")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=_to_rgb(frame))
        if self._static_image:
            return self._task.detect(image)
        # VIDEO mode rejects timestamps that do not strictly increase.
        timestamp_ms = max(self._last_timestamp_ms + 1, int(time.monotonic() * 1000))
        self._last_timestamp_ms = timestamp_ms
        return self._task.detect_for_video(image, timestamp_ms)

    def _detected(self) -> list[Any]:
        if self._last is None:
            raise RuntimeError("Call detect(frame) before locate(frame), once for every frame.")
        return self._last

    def close(self) -> None:
        """Release the model. The detector cannot be used afterwards."""
        if not self._closed:
            self._closed = True
            self._task.close()

    def __enter__(self: _D) -> _D:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _LandmarkDetector(Detector):
    _connections: Sequence[Any]

    def _draw_skeletons(self, frame: np.ndarray, color: Color, point_color: Color) -> None:
        for found in self._detected():
            _draw_connections(frame, found.points, self._connections, color, 2)
            _draw_points(frame, found.points, point_color, 3)

    def _locate(self, frame: np.ndarray, index: int, draw: bool, color: Color, radius: int) -> list[list[int]]:
        detected = self._detected()
        if not 0 <= index < len(detected):
            return []
        found = detected[index]
        if draw:
            _draw_points(frame, found.points, color, radius)
        return [[i, x, y] for i, (x, y) in enumerate(found.points)]


class Hand(_LandmarkDetector):
    """Finds hands and their 21 landmarks (wrist, knuckles and fingertips).

    Args:
        max_hands: Largest number of hands to look for.
        detection_confidence: Minimum confidence (0.0 to 1.0) to report a new hand.
        tracking_confidence: Minimum confidence (0.0 to 1.0) to keep tracking a hand between frames.
        static_image: ``True`` for unrelated still images, ``False`` (default) for video frames.
        model_path: Use a local ``hand_landmarker.task`` file instead of downloading one.
    """

    _connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    @_accepts_legacy(
        {
            "handsMax": "max_hands",
            "detectionConfidence": "detection_confidence",
            "trackingConfidence": "tracking_confidence",
        }
    )
    def __init__(
        self,
        *,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        static_image: bool = False,
        model_path: PathLike | None = None,
    ) -> None:
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        super().__init__(load_model("hand_landmarker.task", model_path), static_image)

    def _create_task(self, base_options: BaseOptions, running_mode: vision.RunningMode) -> Any:
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.detection_confidence,
                min_hand_presence_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
        )

    @_accepts_legacy({"drawOnHand": "draw"})
    def detect(
        self,
        frame: np.ndarray,
        *,
        draw: bool = True,
        color: Color = (224, 224, 224),
        point_color: Color = (0, 0, 255),
    ) -> list[Landmarks]:
        """Find hands in ``frame`` and optionally draw them on it.

        Returns:
            One :class:`Landmarks` per hand, with ``label`` set to ``"Left"`` or ``"Right"``.
            An empty list when no hand is visible. Labels name the person's own hands,
            and MediaPipe swaps them on a mirrored frame such as ``cv.flip(frame, 1)``.
        """
        result = self._run(frame)
        self._last = [
            _to_landmarks(lms, frame.shape, handed[0].category_name, handed[0].score)
            for lms, handed in zip(result.hand_landmarks, result.handedness)
        ]
        if draw:
            self._draw_skeletons(frame, color, point_color)
        return self._last

    @_accepts_legacy(
        {
            "handsNumber": _Renamed("hand", lambda number: max(int(number) - 1, 0), "it counts from 0"),
            "drawOnHand": "draw",
            "drawingColor": "color",
            "drawingSize": "radius",
        }
    )
    def locate(
        self,
        frame: np.ndarray,
        *,
        hand: int = 0,
        draw: bool = True,
        color: Color = (0, 255, 0),
        radius: int = 6,
    ) -> list[list[int]]:
        """Pixel positions of one hand's landmarks from the latest :meth:`detect` call.

        Args:
            frame: The image to draw on.
            hand: Which hand to return, starting at 0.
            draw: Draw a dot on each landmark.
            color: Dot color in BGR.
            radius: Dot radius in pixels.

        Returns:
            ``[[id, x, y], ...]`` for the 21 landmarks, or ``[]`` if that hand is not visible.
        """
        return self._locate(frame, hand, draw, color, radius)


class Pose(_LandmarkDetector):
    """Finds a body and its 33 landmarks (face, shoulders, arms, hips, legs).

    Args:
        model: ``"lite"`` (fastest), ``"full"`` (default) or ``"heavy"`` (most accurate).
        max_poses: Largest number of people to look for.
        detection_confidence: Minimum confidence (0.0 to 1.0) to report a new body.
        tracking_confidence: Minimum confidence (0.0 to 1.0) to keep tracking a body between frames.
        static_image: ``True`` for unrelated still images, ``False`` (default) for video frames.
        model_path: Use a local ``pose_landmarker_*.task`` file instead of downloading one.
    """

    _connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    @_accepts_legacy(
        {
            "upperBody": None,
            "detectionConfidence": "detection_confidence",
            "trackingConfidence": "tracking_confidence",
        }
    )
    def __init__(
        self,
        *,
        model: str = "full",
        max_poses: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        static_image: bool = False,
        model_path: PathLike | None = None,
    ) -> None:
        if model not in _POSE_MODELS:
            raise ValueError(f"model must be one of {', '.join(map(repr, _POSE_MODELS))}, got {model!r}.")
        self.model = model
        self.max_poses = max_poses
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        super().__init__(load_model(_POSE_MODELS[model], model_path), static_image)

    def _create_task(self, base_options: BaseOptions, running_mode: vision.RunningMode) -> Any:
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_poses=self.max_poses,
                min_pose_detection_confidence=self.detection_confidence,
                min_pose_presence_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
        )

    @_accepts_legacy({"drawOnPose": "draw"})
    def detect(
        self,
        frame: np.ndarray,
        *,
        draw: bool = True,
        color: Color = (224, 224, 224),
        point_color: Color = (0, 0, 255),
    ) -> list[Landmarks]:
        """Find bodies in ``frame`` and optionally draw them on it.

        Returns:
            One :class:`Landmarks` per body, or an empty list when nobody is visible.
        """
        result = self._run(frame)
        self._last = [_to_landmarks(lms, frame.shape) for lms in result.pose_landmarks]
        if draw:
            self._draw_skeletons(frame, color, point_color)
        return self._last

    @_accepts_legacy({"drawOnPose": "draw", "drawingColor": "color", "drawingSize": "radius"})
    def locate(
        self,
        frame: np.ndarray,
        *,
        pose: int = 0,
        draw: bool = True,
        color: Color = (0, 255, 0),
        radius: int = 6,
    ) -> list[list[int]]:
        """Pixel positions of one body's landmarks from the latest :meth:`detect` call.

        Returns:
            ``[[id, x, y], ...]`` for the 33 landmarks, or ``[]`` if nobody is visible.
        """
        return self._locate(frame, pose, draw, color, radius)


class FaceMesh(_LandmarkDetector):
    """Finds faces and their 478 mesh landmarks (including the irises).

    Args:
        max_faces: Largest number of faces to look for.
        detection_confidence: Minimum confidence (0.0 to 1.0) to report a new face.
        tracking_confidence: Minimum confidence (0.0 to 1.0) to keep tracking a face between frames.
        static_image: ``True`` for unrelated still images, ``False`` (default) for video frames.
        model_path: Use a local ``face_landmarker.task`` file instead of downloading one.
    """

    _connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS

    @_accepts_legacy(
        {
            "facesMax": "max_faces",
            "detectionConfidence": "detection_confidence",
            "trackingConfidence": "tracking_confidence",
        }
    )
    def __init__(
        self,
        *,
        max_faces: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        static_image: bool = False,
        model_path: PathLike | None = None,
    ) -> None:
        self.max_faces = max_faces
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        super().__init__(load_model("face_landmarker.task", model_path), static_image)

    def _create_task(self, base_options: BaseOptions, running_mode: vision.RunningMode) -> Any:
        return vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_faces=self.max_faces,
                min_face_detection_confidence=self.detection_confidence,
                min_face_presence_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
        )

    @_accepts_legacy(
        {
            "drawOnFace": "draw",
            "drawingColor": "color",
            "drawingThickness": "thickness",
            "drawingCircleRadius": "radius",
        }
    )
    def detect(
        self,
        frame: np.ndarray,
        *,
        draw: bool = True,
        color: Color = (224, 224, 224),
        thickness: int = 1,
        radius: int = 2,
    ) -> list[Landmarks]:
        """Find faces in ``frame`` and optionally draw the mesh on them.

        Args:
            frame: A BGR image from OpenCV.
            draw: Draw a ring on each landmark and the face outlines (eyes, brows, lips, face oval).
            color: Drawing color in BGR.
            thickness: Line thickness in pixels.
            radius: Landmark ring radius in pixels.

        Returns:
            One :class:`Landmarks` per face, or an empty list when no face is visible.
        """
        result = self._run(frame)
        self._last = [_to_landmarks(lms, frame.shape) for lms in result.face_landmarks]
        if draw:
            for face in self._last:
                _draw_connections(frame, face.points, self._connections, color, thickness)
                _draw_points(frame, face.points, color, radius, thickness)
        return self._last

    @_accepts_legacy({})
    def locate(
        self,
        frame: np.ndarray,
        *,
        face: int = 0,
        draw: bool = False,
        show_ids: bool = False,
        color: Color = (0, 255, 0),
        radius: int = 1,
    ) -> list[list[int]]:
        """Pixel positions of one face's landmarks from the latest :meth:`detect` call.

        Args:
            frame: The image to draw on.
            face: Which face to return, starting at 0.
            draw: Draw a dot on each landmark.
            show_ids: Write each landmark's number next to it, to find the ids you need.
            color: Dot and text color in BGR.
            radius: Dot radius in pixels.

        Returns:
            ``[[id, x, y], ...]`` for the 478 landmarks, or ``[]`` if that face is not visible.
        """
        landmarks = self._locate(frame, face, draw, color, radius)
        if show_ids:
            for i, x, y in landmarks:
                cv.putText(frame, str(i), (x, y), cv.FONT_HERSHEY_PLAIN, 0.6, color, 1)
        return landmarks


class Face(Detector):
    """Finds faces and returns a box around each one (faces within about 2 m of the camera).

    Args:
        detection_confidence: Minimum confidence (0.0 to 1.0) to report a face.
        static_image: ``True`` for unrelated still images, ``False`` (default) for video frames.
        model_path: Use a local ``blaze_face_short_range.tflite`` file instead of downloading one.
    """

    @_accepts_legacy({"detectionConfidence": "detection_confidence", "faceDistanceLevel": None})
    def __init__(
        self,
        *,
        detection_confidence: float = 0.6,
        static_image: bool = False,
        model_path: PathLike | None = None,
    ) -> None:
        self.detection_confidence = detection_confidence
        super().__init__(load_model("blaze_face_short_range.tflite", model_path), static_image)

    def _create_task(self, base_options: BaseOptions, running_mode: vision.RunningMode) -> Any:
        return vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=running_mode,
                min_detection_confidence=self.detection_confidence,
            )
        )

    @_accepts_legacy({"drawOnFace": "draw"})
    def detect(self, frame: np.ndarray, *, draw: bool = True, color: Color = (0, 255, 0)) -> list[FaceBox]:
        """Find faces in ``frame`` and optionally draw a box and key points on each.

        Returns:
            One :class:`FaceBox` per face, or an empty list when no face is visible.
        """
        result = self._run(frame)
        self._last = [
            FaceBox(
                x=d.bounding_box.origin_x,
                y=d.bounding_box.origin_y,
                width=d.bounding_box.width,
                height=d.bounding_box.height,
                score=d.categories[0].score,
                keypoints=_to_pixels(d.keypoints, frame.shape),
            )
            for d in result.detections
        ]
        if draw:
            for face in self._last:
                cv.rectangle(frame, face.box, color, 2)
                _draw_points(frame, face.keypoints, (0, 0, 255), 3)
        return self._last

    @_accepts_legacy({"drawOnFace": "draw", "drawingColor": "color"})
    def locate(self, frame: np.ndarray, *, draw: bool = True, color: Color = (0, 255, 0)) -> list[list[Any]]:
        """Face boxes from the latest :meth:`detect` call.

        Args:
            frame: The image to draw on.
            draw: Draw a framed box and the confidence percentage on each face.
            color: Drawing color in BGR.

        Returns:
            ``[[id, (x, y, width, height), score], ...]``, or ``[]`` when no face is visible.
        """
        boxes = []
        for i, face in enumerate(self._detected()):
            boxes.append([i, face.box, face.score])
            if draw:
                _draw_corner_box(frame, face.box, color)
                cv.putText(
                    frame, f"{int(face.score * 100)}%", (face.x, face.y - 20), cv.FONT_HERSHEY_PLAIN, 2, color, 2
                )
        return boxes


def _draw_corner_box(frame: np.ndarray, box: tuple[int, int, int, int], color: Color) -> None:
    x, y, w, h = box
    x2, y2 = x + w, y + h
    cv.rectangle(frame, box, color, 1)
    for cx, cy, dx, dy in ((x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)):
        cv.line(frame, (cx, cy), (cx + dx * 30, cy), color, 5)
        cv.line(frame, (cx, cy), (cx, cy + dy * 30), color, 5)
