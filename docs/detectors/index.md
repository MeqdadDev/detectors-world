# Detectors

Detectors World has four detectors. Pick the one that matches what you want to find.

| Detector | Name | Finds | Results | Models download |
|---|---|---|---|---|
| [Hand](hand.md) | `"hand"` | Hands, 21 landmarks each, left or right | `Landmarks` | 7.8 MB |
| [Pose](pose.md) | `"pose"` | Bodies, 33 landmarks each | `Landmarks` | 5.8 to 30.7 MB |
| [Face](face.md) | `"face"` | Face boxes and 6 key points | `FaceBox` | 0.2 MB |
| [Face Mesh](face-mesh.md) | `"face_mesh"` | Faces, 478 landmarks each | `Landmarks` | 3.8 MB |

## Which one should I use?

- **Is someone there, and where?** Use [Face](face.md). It is the smallest model and returns simple boxes.
- **Where are the eyes, lips or nose exactly?** Use [Face Mesh](face-mesh.md).
- **What are the hands and fingers doing?** Use [Hand](hand.md).
- **How is the whole body moving?** Use [Pose](pose.md).

## Options every detector accepts

| Option | Default | Meaning |
|---|---|---|
| `detection_confidence` | `0.5` (`0.6` for `"face"`) | Minimum confidence, from 0.0 to 1.0, to report something new |
| `tracking_confidence` | `0.5` | Minimum confidence to keep following something between frames (not used by `"face"`) |
| `static_image` | `False` | Set to `True` for unrelated photos |
| `model_path` | `None` | Use a model file you already have |

Options after these are specific to each detector and are listed on its page.
