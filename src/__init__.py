# from directory.filename.py import classname
import detectors_world

__author__ = 'MeqdadDev'
__email__ = 'meqdad.darweesh@gmail.com'
__version__ = '0.1.0'

__doc__ = """
Detectors World
================

Description
-----------
Detectors World is a Python package created for building Computer Vision applicationS in easy and handy way.
Detectors World is a tool that was built on top of `mediapipe` package from Google.
Detectors World developed by Eng. Meqdad Darwish at Purpose for Smart Education company.
Source Code is published on GitHub:
https://github.com/MeqdadDev

Example
-------
>>> from detectors_world import DetectorCreator
>>> import cv2 as cv
>>> cap = cv.VideoCapture(0)
>>> creator = DetectorCreator()
>>> hand = creator.getDetector("hand")

>>> while True:
>>>     _, img = cap.read()
>>>     hand.detect(img)
>>>     cv.imshow("Detection", img)
>>>     cv.waitKey(1)


References
----------
* https://github.com/MeqdadDev

"""
