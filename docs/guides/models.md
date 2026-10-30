# Models and Offline Use

Each detector needs a model file. Detectors World downloads the one it needs the first time, saves it, and reuses it after that.

## Model files

| Detector | File | Size |
|---|---|---|
| Hand | `hand_landmarker.task` | 7.8 MB |
| Pose, `"lite"` | `pose_landmarker_lite.task` | 5.8 MB |
| Pose, `"full"` | `pose_landmarker_full.task` | 9.4 MB |
| Pose, `"heavy"` | `pose_landmarker_heavy.task` | 30.7 MB |
| Face | `blaze_face_short_range.tflite` | 0.2 MB |
| Face Mesh | `face_landmarker.task` | 3.8 MB |

## Where models are stored

| System | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\detectors_world` |
| Linux and macOS | `~/.cache/detectors_world`, or `$XDG_CACHE_HOME/detectors_world` if set |

To check the folder from Python:

```python
from detectors_world import cache_dir

print(cache_dir())
```

## Use another folder

Set the `DETECTORS_WORLD_CACHE` environment variable:

=== "Windows PowerShell"

    ```powershell
    $env:DETECTORS_WORLD_CACHE = "D:\models"
    ```

=== "Linux and macOS"

    ```bash
    export DETECTORS_WORLD_CACHE=~/models
    ```

## Use models offline

Run each detector once on a computer with internet access, so that its model is downloaded. Then copy the model files to the offline computer in either of these ways:

- Put them in that computer's model folder, and detectors find them automatically.
- Keep them anywhere, and pass `model_path`:

```python
hand = create_detector("hand", model_path="models/hand_landmarker.task")
```

`model_path` also lets you use a model of your own. Files passed this way aren't checked, so make sure yours matches the detector.

## Damaged downloads

Downloaded models are checked against a known hash. A damaged download is rejected, and a damaged file in the model folder is replaced automatically the next time it is needed.
