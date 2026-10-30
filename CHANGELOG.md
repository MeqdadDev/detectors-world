# Changelog

All notable changes to this project are documented here. The project follows [Semantic Versioning](https://semver.org/).

## [1.0.0] - Unreleased

The package works again with current MediaPipe releases. Recent MediaPipe releases removed the
`mediapipe.solutions` API that 0.x was built on, so a fresh `pip install detectors-world` failed with
`AttributeError: module 'mediapipe' has no attribute 'solutions'`.

### Added
- `create_detector("hand" | "pose" | "face" | "face_mesh", **options)`: creates only the detector you ask for and accepts its options.
- `detect()` returns results: `Landmarks` objects (pixel `points`, `normalized` coordinates, and `label`/`score` for left/right hands) or `FaceBox` objects (`x`, `y`, `width`, `height`, `score`, `keypoints`).
- Detectors work as context managers and have a `close()` method.
- `static_image=True` option for unrelated photos, and `model_path=` for offline use.
- `Pose(model="lite" | "full" | "heavy")`.
- `FaceMesh.locate(show_ids=True)` to label landmark numbers on the frame.
- Clear errors for an unknown detector name, a `None` frame (camera read failed) and unsupported image types.
- Grayscale and BGRA frames are accepted.
- Type hints and a `py.typed` marker for editor autocompletion.
- Tests, GitHub Actions CI on Python 3.10–3.14, and runnable scripts in `examples/`.

### Changed
- Rebuilt on the MediaPipe Tasks API (`mediapipe>=1.2,<2`). Model files are downloaded on first use, checked against pinned hashes and cached. Damaged cached files are replaced automatically.
- Depends on `opencv-contrib-python` (the build MediaPipe installs) instead of `opencv-python`. When upgrading from 0.x, run `pip uninstall opencv-python` so that only one copy of `cv2` is installed.
- Requires Python 3.10 or newer.
- Installs as a single `detectors_world` package instead of loose `detectors.py`, `detectors_world.py` and `__init__.py` modules in `site-packages`.
- Options are keyword-only and use snake_case names. The 0.x camelCase names still work with a `DeprecationWarning` and will be removed in 2.0.
- `locate()` returns a list for every detector, and `[]` when nothing is found. `Face.locate()` returns `[[id, (x, y, w, h), score], ...]`, `FaceMesh.locate()` returns one face's `[[id, x, y], ...]` and no longer draws ids unless `show_ids=True`.
- `Hand.locate()` returns `[]` when the requested hand is not visible, instead of printing a message and returning the first hand.
- `detect()` returns its results instead of the frame.
- `DetectorCreator` creates detectors on first request instead of loading every model up front.

### Fixed
- Options such as `handsMax`, `facesMax` and the confidence values were ignored, and video frames were processed as unrelated still images (no tracking).
- `FaceMesh.detect()` returned `None` when no face was visible and drew only the first face.
- `FaceMesh.locate()` passed BGR frames to the model as RGB.
- `locate()` raised a bare `AttributeError` when called before `detect()`. It now raises a `RuntimeError` that says what to do.

### Removed
- The `upperBody` (`Pose`) and `faceDistanceLevel` (`Face`) options have no MediaPipe Tasks equivalent; passing them shows a warning and has no effect.

## [0.0.2] - 2023-02-12
- Last release built on the legacy `mediapipe.solutions` API.
