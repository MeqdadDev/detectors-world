import numpy as np
import pytest

import detectors_world as dw
from detectors_world import Face, FaceBox, FaceMesh, Hand, Landmarks, Pose, create_detector


@pytest.mark.parametrize(
    ("kind", "cls"),
    [("hand", Hand), ("Hands", Hand), ("pose", Pose), ("face", Face), ("facemesh", FaceMesh), ("face-mesh", FaceMesh)],
)
def test_create_detector_returns_matching_class(kind, cls):
    with create_detector(kind) as detector:
        assert type(detector) is cls


def test_create_detector_unknown_name_lists_choices():
    with pytest.raises(ValueError, match="'hand', 'pose', 'face', 'face_mesh'"):
        create_detector("car")


def test_options_reach_the_model():
    with create_detector("hand", max_hands=1, detection_confidence=0.7) as hand:
        assert hand.max_hands == 1
        assert hand.detection_confidence == 0.7


def test_hand_detects_both_hands(image):
    frame = image("hands")
    with Hand(static_image=True) as hand:
        hands = hand.detect(frame, draw=False)
        assert len(hands) == 2
        assert all(isinstance(h, Landmarks) and len(h) == 21 for h in hands)
        assert {h.label for h in hands} == {"Left", "Right"}
        height, width = frame.shape[:2]
        assert all(0 <= x < width and 0 <= y < height for h in hands for x, y in h.points)

        located = hand.locate(frame, hand=1, draw=False)
        assert [row[0] for row in located] == list(range(21))
        assert [tuple(row[1:]) for row in located] == list(hands[1].points)
        assert hand.locate(frame, hand=5) == []


def test_pose_detects_one_body(image):
    with Pose(static_image=True) as pose:
        bodies = pose.detect(image("pose"))
        assert len(bodies) == 1
        assert len(bodies[0]) == 33


def test_face_detects_one_face(image):
    frame = image("person")
    with Face(static_image=True) as face:
        faces = face.detect(frame, draw=False)
        assert len(faces) == 1
        assert isinstance(faces[0], FaceBox)
        assert faces[0].score > 0.8
        assert len(faces[0].keypoints) == 6
        assert face.locate(frame, draw=False) == [[0, faces[0].box, faces[0].score]]


def test_face_mesh_detects_478_landmarks(image):
    with FaceMesh(static_image=True) as mesh:
        faces = mesh.detect(image("person"))
        assert len(faces) == 1
        assert len(faces[0]) == 478


def test_face_mesh_draws_rings_on_landmarks_outside_the_contours(image):
    nose_tip = 1
    frame = image("person")
    with FaceMesh(static_image=True) as mesh:
        drawn = frame.copy()
        x, y = mesh.detect(drawn, color=(255, 0, 255), radius=3)[0][nose_tip]
    ring = drawn[y - 4 : y + 5, x - 4 : x + 5].astype(int)
    # Rings are anti-aliased, so pixels get close to the color without matching it exactly.
    assert np.abs(ring - (255, 0, 255)).sum(axis=2).min() < 30


@pytest.mark.parametrize("cls", [Hand, Pose, Face, FaceMesh])
def test_nothing_found_on_blank_frame(cls, blank):
    with cls() as detector:
        assert detector.detect(blank) == []
        assert detector.locate(blank) == []
        assert not blank.any()


def test_draw_changes_frame_only_when_asked(image):
    frame = image("hands")
    with Hand(static_image=True) as hand:
        untouched = frame.copy()
        hand.detect(untouched, draw=False)
        assert np.array_equal(untouched, frame)
        drawn = frame.copy()
        hand.detect(drawn)
        assert not np.array_equal(drawn, frame)


def test_locate_without_detect_runs_detection(image):
    with Hand(static_image=True) as hand:
        assert len(hand.locate(image("hands"), draw=False)) == 21


def test_video_mode_accepts_back_to_back_frames(image):
    frame = image("hands")
    with Hand() as hand:
        for _ in range(5):
            assert len(hand.detect(frame, draw=False)) == 2


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((120, 160), dtype=np.uint8),
        np.zeros((120, 160, 4), dtype=np.uint8),
    ],
)
def test_grayscale_and_bgra_frames_are_accepted(frame):
    with Hand() as hand:
        assert hand.detect(frame) == []


def test_none_frame_explains_camera_problem():
    with Hand() as hand, pytest.raises(ValueError, match="camera returned no image"):
        hand.detect(None)


def test_float_frame_is_rejected():
    with Hand() as hand, pytest.raises(ValueError, match="uint8"):
        hand.detect(np.zeros((10, 10, 3), dtype=np.float32))


def test_unknown_pose_model_is_rejected():
    with pytest.raises(ValueError, match="'lite', 'full', 'heavy'"):
        Pose(model="huge")


def test_missing_model_path_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        Hand(model_path=tmp_path / "missing.task")


def test_version_is_exposed():
    assert dw.__version__.count(".") == 2
