# Video File Processing

Runs a detector on every frame of a video and saves a new video with the drawings.

## Run it

```bash
python examples/process_video.py pose dance.mp4
python examples/process_video.py hand clip.mp4 --output tracked.mp4
```

## Code

```python title="examples/process_video.py"
--8<-- "examples/process_video.py"
```

## How it works

- `cap.get()` reads the speed and size of the source, so that the new video matches.
- `cv.VideoWriter` saves frames as an MP4 file.
- The detector stays in its default video mode, so it follows things between frames.
- `writer.release()` finishes the file. Without it, the video can't be played.
