from __future__ import annotations

import os
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ._models import resolve_model
from .results import FaceBox, Landmarks

Color = tuple[int, int, int]
PathLike = str | os.PathLike[str]

_POSE_MODELS = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}


def _translate_legacy(
    legacy: dict[str, Any], mapping: dict[str, str | None], owner: str
) -> dict[str, Any]:
    """Map 0.x camelCase keyword arguments to their new names, warning once per use."""
    translated = {}
    for old, value in legacy.items():
        if old not in mapping:
            raise TypeError(f"{owner}() got an unexpected keyword argument '{old}'")
        new = mapping[old]
        if new is None:
            message = f"'{old}' has no effect since detectors_world 1.0 and will be removed in 2.0."
        else:
            message = f"'{old}' is deprecated, use '{new}' instead. It will be removed in detectors_world 2.0."
            translated[new] = value
        warnings.warn(message, DeprecationWarning, stacklevel=3)
    return translated


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError(
            "frame is None. If it came from cv.VideoCapture.read(), the camera returned no image: "
            "check the camera index and that no other program is using the camera."
        )
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        raise ValueError(f"frame must be an 8-bit image (dtype uint8), got dtype {frame.dtype}.")
    if frame.ndim == 2:
        return cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv.cvtColor(frame, cv.COLOR_BGRA2RGB)
    raise ValueError(f"frame must be a grayscale, BGR or BGRA image, got shape {frame.shape}.")


def _to_landmarks(
    landmarks: Sequence[Any], shape: tuple[int, ...], label: str | None = None, score: float | None = None
) -> Landmarks:
    height, width = shape[:2]
    return Landmarks(
        points=tuple((int(lm.x * width), int(lm.y * height)) for lm in landmarks),
        normalized=tuple((lm.x, lm.y, lm.z) for lm in landmarks),
        label=label,
        score=score,
    )


def _draw_skeleton(
    frame: np.ndarray,
    points: Sequence[tuple[int, int]],
    connections: Iterable[Any],
    line_color: Color,
    point_color: Color | None,
    thickness: int,
    radius: int,
) -> None:
    for connection in connections:
        cv.line(frame, points[connection.start], points[connection.end], line_color, thickness, cv.LINE_AA)
    if point_color is not None:
        for point in points:
            cv.circle(frame, point, radius, point_color, cv.FILLED, cv.LINE_AA)


def _draw_points(frame: np.ndarray, points: Iterable[tuple[int, int]], color: Color, radius: int) -> None:
    for point in points:
        cv.circle(frame, point, radius, color, cv.FILLED, cv.LINE_AA)


def _indexed(landmarks: Landmarks) -> list[list[int]]:
    return [[i, x, y] for i, (x, y) in enumerate(landmarks.points)]


class Detector(ABC):
    """Base class for all detectors.

    Detectors hold a loaded model. Use them as a context manager, or call
    :meth:`close` when you are done, to release it.
    """

    def __init__(self, static_image: bool) -> None:
        self._static_image = static_image
        self._last_timestamp_ms = -1
        self._task = self._create_task(vision.RunningMode.IMAGE if static_image else vision.RunningMode.VIDEO)

    @abstractmethod
    def _create_task(self, running_mode: vision.RunningMode) -> Any: ...

    @abstractmethod
    def detect(self, frame: np.ndarray, *, draw: bool = True) -> list[Any]:
        """Run the model on ``frame`` (a BGR image from OpenCV) and return what was found."""

    @abstractmethod
    def locate(self, frame: np.ndarray) -> list[Any]:
        """Return pixel positions from the latest :meth:`detect` call."""

    def _run(self, frame: np.ndarray) -> Any:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=_to_rgb(frame))
        if self._static_image:
            return self._task.detect(image)
        # VIDEO mode rejects timestamps that do not strictly increase.
        timestamp_ms = max(self._last_timestamp_ms + 1, int(time.monotonic() * 1000))
        self._last_timestamp_ms = timestamp_ms
        return self._task.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        """Release the model. The detector cannot be used afterwards."""
        self._task.close()

    def __enter__(self) -> Detector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _LandmarkDetector(Detector):
    _connections: Sequence[Any] = ()
    _owner = "Detector"
    _last: list[Landmarks] | None = None

    def _pick(self, frame: np.ndarray, index: int) -> Landmarks | None:
        if self._last is None:
            self.detect(frame, draw=False)
        assert self._last is not None
        if not 0 <= index < len(self._last):
            return None
        return self._last[index]

    def _locate(self, frame: np.ndarray, index: int, draw: bool, color: Color, radius: int) -> list[list[int]]:
        found = self._pick(frame, index)
        if found is None:
            return []
        if draw:
            _draw_points(frame, found.points, color, radius)
        return _indexed(found)


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

    def __init__(
        self,
        *,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        static_image: bool = False,
        model_path: PathLike | None = None,
        **legacy: Any,
    ) -> None:
        options = {
            "max_hands": max_hands,
            "detection_confidence": detection_confidence,
            "tracking_confidence": tracking_confidence,
        }
        options.update(
            _translate_legacy(
                legacy,
                {
                    "handsMax": "max_hands",
                    "detectionConfidence": "detection_confidence",
                    "trackingConfidence": "tracking_confidence",
                },
                "Hand",
            )
        )
        self.max_hands = options["max_hands"]
        self.detection_confidence = options["detection_confidence"]
        self.tracking_confidence = options["tracking_confidence"]
        self._model_path = resolve_model("hand_landmarker.task", model_path)
        super().__init__(static_image)

    def _create_task(self, running_mode: vision.RunningMode) -> Any:
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._model_path),
                running_mode=running_mode,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.detection_confidence,
                min_hand_presence_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
        )

    def detect(
        self,
        frame: np.ndarray,
        *,
        draw: bool = True,
        color: Color = (224, 224, 224),
        point_color: Color = (0, 0, 255),
        **legacy: Any,
    ) -> list[Landmarks]:
        """Find hands in ``frame`` and optionally draw them on it.

        Returns:
            One :class:`Landmarks` per hand, with ``label`` set to ``"Left"`` or ``"Right"``.
            An empty list when no hand is visible.
        """
        draw = _translate_legacy(legacy, {"drawOnHand": "draw"}, "Hand.detect").get("draw", draw)
        result = self._run(frame)
        self._last = [
            _to_landmarks(lms, frame.shape, handed[0].category_name, handed[0].score)
            for lms, handed in zip(result.hand_landmarks, result.handedness)
        ]
        if draw:
            for hand in self._last:
                _draw_skeleton(frame, hand.points, self._connections, color, point_color, 2, 3)
        return self._last

    def locate(
        self,
        frame: np.ndarray,
        *,
        hand: int = 0,
        draw: bool = True,
        color: Color = (0, 255, 0),
        radius: int = 6,
        **legacy: Any,
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
        if "handsNumber" in legacy:
            hand = max(int(legacy["handsNumber"]) - 1, 0)
            legacy = {k: v for k, v in legacy.items() if k != "handsNumber"}
            warnings.warn(
                "'handsNumber' is deprecated, use 'hand' instead (it counts from 0). "
                "It will be removed in detectors_world 2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        options = _translate_legacy(
            legacy,
            {"drawOnHand": "draw", "drawingColor": "color", "drawingSize": "radius"},
            "Hand.locate",
        )
        return self._locate(
            frame, hand, options.get("draw", draw), options.get("color", color), options.get("radius", radius)
        )


class Pose(_LandmarkDetector):
    """Finds a body and its 33 landmarks (face, shoulders, arms, hips, legs).

    Args:
        model: ``"lite"`` (fastest), ``"full"`` or ``"heavy"`` (most accurate).
        max_poses: Largest number of people to look for.
        detection_confidence: Minimum confidence (0.0 to 1.0) to report a new body.
        tracking_confidence: Minimum confidence (0.0 to 1.0) to keep tracking a body between frames.
        static_image: ``True`` for unrelated still images, ``False`` (default) for video frames.
        model_path: Use a local ``pose_landmarker_*.task`` file instead of downloading one.
    """

    _connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    def __init__(
        self,
        *,
        model: str = "lite",
        max_poses: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        static_image: bool = False,
        model_path: PathLike | None = None,
        **legacy: Any,
    ) -> None:
        if model not in _POSE_MODELS:
            raise ValueError(f"model must be one of {', '.join(map(repr, _POSE_MODELS))}, got {model!r}.")
        options = {"detection_confidence": detection_confidence, "tracking_confidence": tracking_confidence}
        options.update(
            _translate_legacy(
                legacy,
                {
                    "upperBody": None,
                    "detectionConfidence": "detection_confidence",
                    "trackingConfidence": "tracking_confidence",
                },
                "Pose",
            )
        )
        self.max_poses = max_poses
        self.detection_confidence = options["detection_confidence"]
        self.tracking_confidence = options["tracking_confidence"]
        self._model_path = resolve_model(_POSE_MODELS[model], model_path)
        super().__init__(static_image)

    def _create_task(self, running_mode: vision.RunningMode) -> Any:
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._model_path),
                running_mode=running_mode,
                num_poses=self.max_poses,
                min_pose_detection_confidence=self.detection_confidence,
                min_pose_presence_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
        )

    def detect(
        self,
        frame: np.ndarray,
        *,
        draw: bool = True,
        color: Color = (224, 224, 224),
        point_color: Color = (0, 0, 255),
        **legacy: Any,
    ) -> list[Landmarks]:
        """Find bodies in ``frame`` and optionally draw them on it.

        Returns:
            One :class:`Landmarks` per body, or an empty list when nobody is visible.
        """
        draw = _translate_legacy(legacy, {"drawOnPose": "draw"}, "Pose.detect").get("draw", draw)
        result = self._run(frame)
        self._last = [_to_landmarks(lms, frame.shape) for lms in result.pose_landmarks]
        if draw:
            for pose in self._last:
                _draw_skeleton(frame, pose.points, self._connections, color, point_color, 2, 3)
        return self._last

    def locate(
        self,
        frame: np.ndarray,
        *,
        pose: int = 0,
        draw: bool = True,
        color: Color = (0, 255, 0),
        radius: int = 6,
        **legacy: Any,
    ) -> list[list[int]]:
        """Pixel positions of one body's landmarks from the latest :meth:`detect` call.

        Returns:
            ``[[id, x, y], ...]`` for the 33 landmarks, or ``[]`` if nobody is visible.
        """
        options = _translate_legacy(
            legacy,
            {"drawOnPose": "draw", "drawingColor": "color", "drawingSize": "radius"},
            "Pose.locate",
        )
        return self._locate(
            frame, pose, options.get("draw", draw), options.get("color", color), options.get("radius", radius)
        )


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

    def __init__(
        self,
        *,
        max_faces: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        static_image: bool = False,
        model_path: PathLike | None = None,
        **legacy: Any,
    ) -> None:
        options = {
            "max_faces": max_faces,
            "detection_confidence": detection_confidence,
            "tracking_confidence": tracking_confidence,
        }
        options.update(
            _translate_legacy(
                legacy,
                {
                    "facesMax": "max_faces",
                    "detectionConfidence": "detection_confidence",
                    "trackingConfidence": "tracking_confidence",
                },
                "FaceMesh",
            )
        )
        self.max_faces = options["max_faces"]
        self.detection_confidence = options["detection_confidence"]
        self.tracking_confidence = options["tracking_confidence"]
        self._model_path = resolve_model("face_landmarker.task", model_path)
        super().__init__(static_image)

    def _create_task(self, running_mode: vision.RunningMode) -> Any:
        return vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._model_path),
                running_mode=running_mode,
                num_faces=self.max_faces,
                min_face_detection_confidence=self.detection_confidence,
                min_face_presence_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
        )

    def detect(
        self,
        frame: np.ndarray,
        *,
        draw: bool = True,
        color: Color = (224, 224, 224),
        thickness: int = 1,
        radius: int = 2,
        **legacy: Any,
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
        options = _translate_legacy(
            legacy,
            {
                "drawOnFace": "draw",
                "drawingColor": "color",
                "drawingThickness": "thickness",
                "drawingCircleRadius": "radius",
            },
            "FaceMesh.detect",
        )
        draw = options.get("draw", draw)
        color = options.get("color", color)
        thickness = options.get("thickness", thickness)
        radius = options.get("radius", radius)
        result = self._run(frame)
        self._last = [_to_landmarks(lms, frame.shape) for lms in result.face_landmarks]
        if draw:
            for face in self._last:
                _draw_skeleton(frame, face.points, self._connections, color, None, thickness, 0)
                for point in face.points:
                    cv.circle(frame, point, radius, color, thickness, cv.LINE_AA)
        return self._last

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

    _last: list[FaceBox] | None = None

    def __init__(
        self,
        *,
        detection_confidence: float = 0.5,
        static_image: bool = False,
        model_path: PathLike | None = None,
        **legacy: Any,
    ) -> None:
        options = _translate_legacy(
            legacy,
            {"detectionConfidence": "detection_confidence", "faceDistanceLevel": None},
            "Face",
        )
        self.detection_confidence = options.get("detection_confidence", detection_confidence)
        self._model_path = resolve_model("blaze_face_short_range.tflite", model_path)
        super().__init__(static_image)

    def _create_task(self, running_mode: vision.RunningMode) -> Any:
        return vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=self._model_path),
                running_mode=running_mode,
                min_detection_confidence=self.detection_confidence,
            )
        )

    def detect(
        self, frame: np.ndarray, *, draw: bool = True, color: Color = (0, 255, 0), **legacy: Any
    ) -> list[FaceBox]:
        """Find faces in ``frame`` and optionally draw a box and key points on each.

        Returns:
            One :class:`FaceBox` per face, or an empty list when no face is visible.
        """
        draw = _translate_legacy(legacy, {"drawOnFace": "draw"}, "Face.detect").get("draw", draw)
        result = self._run(frame)
        height, width = frame.shape[:2]
        self._last = [
            FaceBox(
                x=d.bounding_box.origin_x,
                y=d.bounding_box.origin_y,
                width=d.bounding_box.width,
                height=d.bounding_box.height,
                score=d.categories[0].score,
                keypoints=tuple((int(k.x * width), int(k.y * height)) for k in d.keypoints),
            )
            for d in result.detections
        ]
        if draw:
            for face in self._last:
                cv.rectangle(frame, face.box, color, 2)
                _draw_points(frame, face.keypoints, (0, 0, 255), 3)
        return self._last

    def locate(
        self, frame: np.ndarray, *, draw: bool = True, color: Color = (0, 255, 0), **legacy: Any
    ) -> list[list[Any]]:
        """Face boxes from the latest :meth:`detect` call.

        Args:
            frame: The image to draw on.
            draw: Draw a framed box and the confidence percentage on each face.
            color: Drawing color in BGR.

        Returns:
            ``[[id, (x, y, width, height), score], ...]``, or ``[]`` when no face is visible.
        """
        options = _translate_legacy(legacy, {"drawOnFace": "draw", "drawingColor": "color"}, "Face.locate")
        draw = options.get("draw", draw)
        color = options.get("color", color)
        if self._last is None:
            self.detect(frame, draw=False)
        assert self._last is not None
        boxes = []
        for i, face in enumerate(self._last):
            boxes.append([i, face.box, face.score])
            if draw:
                _draw_corner_box(frame, face.box, color)
                cv.putText(
                    frame, f"{int(face.score * 100)}%", (face.x, face.y - 20), cv.FONT_HERSHEY_PLAIN, 2, color, 2
                )
        return boxes


def _draw_corner_box(
    frame: np.ndarray, box: tuple[int, int, int, int], color: Color, length: int = 30, thickness: int = 5
) -> None:
    x, y, w, h = box
    x2, y2 = x + w, y + h
    cv.rectangle(frame, box, color, 1)
    for cx, cy, dx, dy in ((x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)):
        cv.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness)
        cv.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness)
