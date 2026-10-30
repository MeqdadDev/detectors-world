"""Code written for detectors_world 0.x must keep running, with a DeprecationWarning."""

import pytest

from detectors_world import DetectorCreator, Face, FaceMesh, Hand, Pose


def test_detector_creator_still_works(image):
    with pytest.deprecated_call():
        creator = DetectorCreator()
    hand = creator.getDetector("HAND")
    assert isinstance(hand, Hand)
    assert creator.getDetector("hand") is hand

    frame = image("hands")
    with pytest.deprecated_call():
        hand.detect(frame, drawOnHand=True)
    with pytest.deprecated_call(match="counts from 0"):
        landmarks = hand.locate(frame, drawOnHand=False, handsNumber=1)
    assert len(landmarks) == 21
    hand.close()


def test_camel_case_constructor_options_are_translated():
    with pytest.deprecated_call(match="'handsMax' is deprecated, use 'max_hands'"):
        hand = Hand(handsMax=1, detectionConfidence=0.6)
    assert hand.max_hands == 1
    assert hand.detection_confidence == 0.6
    hand.close()


def test_removed_options_warn_and_are_ignored():
    with pytest.deprecated_call(match="'upperBody' has no effect"):
        Pose(upperBody=True).close()
    with pytest.deprecated_call(match="'faceDistanceLevel' has no effect"):
        Face(faceDistanceLevel=1).close()


def test_face_mesh_legacy_drawing_options(image):
    with FaceMesh(static_image=True) as mesh, pytest.deprecated_call():
        faces = mesh.detect(image("person"), drawOnFace=True, drawingColor=(0, 0, 255), drawingCircleRadius=2)
    assert len(faces) == 1


def test_unknown_keyword_is_a_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument 'maxHands'"):
        Hand(maxHands=1)
