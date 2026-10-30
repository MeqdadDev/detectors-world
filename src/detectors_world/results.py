from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Landmarks:
    """Landmarks of one detected hand, body, or face mesh.

    Attributes:
        points: ``(x, y)`` pixel coordinates, one per landmark.
        normalized: ``(x, y, z)`` coordinates relative to the image size (0.0 to 1.0 for x and y).
        label: ``"Left"`` or ``"Right"`` for hands, otherwise ``None``. This is the person's
            own hand, and MediaPipe swaps the labels on a mirrored frame.
        score: Confidence of ``label`` for hands, otherwise ``None``.
    """

    points: tuple[tuple[int, int], ...]
    normalized: tuple[tuple[float, float, float], ...]
    label: str | None = None
    score: float | None = None

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.points[index]


@dataclass(frozen=True)
class FaceBox:
    """One detected face.

    Attributes:
        x: Left edge in pixels.
        y: Top edge in pixels.
        width: Box width in pixels.
        height: Box height in pixels.
        score: Detection confidence, from 0.0 to 1.0.
        keypoints: ``(x, y)`` pixel positions of the eyes, nose tip, mouth and ear tragions.
    """

    x: int
    y: int
    width: int
    height: int
    score: float
    keypoints: tuple[tuple[int, int], ...] = ()

    @property
    def box(self) -> tuple[int, int, int, int]:
        """The box as ``(x, y, width, height)``."""
        return (self.x, self.y, self.width, self.height)
