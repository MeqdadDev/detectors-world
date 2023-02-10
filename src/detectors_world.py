import detectors


class DetectorCreator():
    """
    Detectors World
    ================

    Description
    -----------
    Detectors World is a Python package created for building Computer Vision applicationS in easy and handy way.

    Detectors World is a tool that was built on top of `mediapipe` package from Google.

    Detectors World developed by Eng. Meqdad Darwish.

    Source Code is published on GitHub:
    https://github.com/MeqdadDev/detectors-world

    Example
    -------
    >>> from detectors_world import DetectorCreator
    >>> import cv2 as cv
    >>> cap = cv.VideoCapture(0)
    >>> creator = DetectorCreator()
    >>> hand = creator.getDetector("hand")

    >>> while True:
    >>>   _, img = cap.read()
    >>>   hand.detect(img)
    >>>   cv.imshow("Detection", img)
    >>>   cv.waitKey(1)


    References
    ----------
    * https://github.com/MeqdadDev/detectors-world
    * https://pypi.org/project/detectors-world

    """

    def __init__(self):
        self._supportedDetectors = {"hand": detectors.Hand(),
                                    "pose": detectors.Pose(),
                                    "face": detectors.Face(),
                                    "facemesh": detectors.FaceMesh(),
                                    "face_mesh": detectors.FaceMesh()}

    def getDetector(self, detectorType: str):
        self.detectorType = detectorType.lower()
        return self._supportedDetectors[self.detectorType]
