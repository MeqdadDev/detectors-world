# Detectors World
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

Detectors World is a Python package created for building Computer Vision applications in easy and handy way.

Developed by [@MeqdadDev](https://www.github.com/MeqdadDev)


## Available Computer Vision Detectors
The available detectors in Detectors World package are: 
- Hand Detection
- Face Detection
- Pose Detection
- Face Mesh Detection

More detectors will be added in the next releases. Check out contribution guides below.


## Documentation

The complete documentation will be added soon....


## Dependencies

Detectors World dependencies are:

```bash
  opencv-python
  mediapipe
```

## Installation

Install detectors world package with pip

```bash
  pip install detectors_world
```


## Run Detectors

### Hand Detector

<p align="center">
<img src="assets/hand-detector.gif" width=50% height=40%>
</p>

Hand detector examples with OpenCV:

#### Example 1:

```python
from detectors_world import DetectorCreator
import cv2 as cv

cap = cv.VideoCapture(0)

creator = DetectorCreator()
hand = creator.getDetector("hand")

while True:
    status, img = cap.read()
    hand.detect(img, drawOnHand=True)
    cv.imshow("Hand Detection", img)
    cv.waitKey(1)
```

#### Example 2:

```python
from detectors_world import DetectorCreator
import cv2 as cv

cap = cv.VideoCapture(0)

creator = DetectorCreator()
hand = creator.getDetector("hand")

while True:
    status, img = cap.read()
    hand.detect(img, drawOnHand=True)
    landmarks = hand.locate(img, drawOnHand=True, handsNumber=1)
    cv.imshow("Hand Detection", img)
    cv.waitKey(1)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

For new projects/examples, please make sure you've tested your code in real environment. And to avoid duplications, please take a sneak peek on the uploaded projects before making your PR.

## 🔗 Find me on
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/meqdad-darwish)

[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/)
