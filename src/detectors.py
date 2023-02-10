import cv2 as cv
import mediapipe as mp
from abc import ABCMeta, abstractmethod


class DetectorTemplate(metaclass=ABCMeta):
    @abstractmethod
    def detect(self):
        pass

    @abstractmethod
    def locate(self):
        pass


class FaceMesh(DetectorTemplate):
    def __init__(self, facesMax=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self._facesMax = facesMax
        self._detectionConfidence = detectionConfidence
        self._trackingConfidence = trackingConfidence
        self._mpDraw = mp.solutions.drawing_utils
        self._mpFaceMesh = mp.solutions.face_mesh
        self._faceMesh = self._mpFaceMesh.FaceMesh(self)
        self._drawSpec = self._mpDraw.DrawingSpec()

    def detect(self, sourceFrame, drawOnFace=True, drawingColor=(224, 224, 224), drawingThickness=1, drawingCircleRadius=2):
        imageRGB = cv.cvtColor(sourceFrame, cv.COLOR_BGR2RGB)
        self.outcome = self._faceMesh.process(imageRGB)
        self._drawSpec = self._mpDraw.DrawingSpec(
            color=drawingColor, thickness=drawingThickness, circle_radius=drawingCircleRadius)

        if self.outcome.multi_face_landmarks:
            for faceLandmarks in self.outcome.multi_face_landmarks:
                if drawOnFace:
                    return self._drawDefault(sourceFrame, faceLandmarks)
                return sourceFrame

    def locate(self, sourceFrame):
        faceMeshLandmarks = []
        self.locating = self._faceMesh.process(sourceFrame)
        if self.locating.multi_face_landmarks:
            faceMeshLandmark = []
            for meshLm in self.locating.multi_face_landmarks:
                for id, meshLandmark in enumerate(meshLm.landmark):
                    row, column, channel = sourceFrame.shape
                    xPosition, yPosition = int(
                        meshLandmark.x * column), int(meshLandmark.y * row)
                    cv.putText(sourceFrame, str(int(id)), (xPosition, yPosition),
                               cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    faceMeshLandmark.append([xPosition, yPosition])
            faceMeshLandmarks.append(faceMeshLandmark)
        return sourceFrame, faceMeshLandmarks

    def _drawDefault(self, sourceFrame, faceLandmarks):
        self._mpDraw.draw_landmarks(
            sourceFrame, faceLandmarks, self._mpFaceMesh.FACEMESH_CONTOURS, self._drawSpec, self._drawSpec)
        return sourceFrame


class Face(DetectorTemplate):
    def __init__(self, detectionConfidence=0.6, faceDistanceLevel=0):
        self._detectionConfidence = detectionConfidence
        self._faceDistanceLevel = faceDistanceLevel
        self._mpFaceDetection = mp.solutions.face_detection
        self._mpDraw = mp.solutions.drawing_utils
        self._faceDetection = self._mpFaceDetection.FaceDetection(
            self._detectionConfidence, self._faceDistanceLevel)

    def detect(self, sourceFrame, drawOnFace=True):
        imageRGB = cv.cvtColor(sourceFrame, cv.COLOR_BGR2RGB)
        self.outcome = self._faceDetection.process(imageRGB)

        if drawOnFace:
            return self._drawDefault(sourceFrame)
        return self.outcome.detections

    def locate(self, sourceFrame, drawOnFace=True, drawingColor=(0, 255, 0)):
        sourceFrame, boundingBoxes = self._getBoundingBoxes(
            sourceFrame, drawOnFace, drawingColor)
        return (sourceFrame, boundingBoxes) if drawOnFace else boundingBoxes

    def _getBoundingBoxes(self, sourceFrame, drawOnFace, drawingColor):
        boundingBoxes = []
        if self.outcome.detections:
            for id, detection in enumerate(self.outcome.detections):
                classBoundingBoxes = detection.location_data.relative_bounding_box
                xPosition, yPosition, channel = sourceFrame.shape
                boundingBox = int(
                    classBoundingBoxes.xmin * yPosition), int(classBoundingBoxes.ymin * xPosition), int(classBoundingBoxes.width * yPosition), int(classBoundingBoxes.height * xPosition)
                boundingBoxes.append([id, boundingBox, detection.score])
                if drawOnFace:
                    sourceFrame = self._drawRectangle(
                        sourceFrame, boundingBox, drawingColor)
                    cv.putText(
                        sourceFrame, f'{int(detection.score[0]*100)}%', (boundingBox[0], boundingBox[1]-20), cv.FONT_HERSHEY_PLAIN, 2, drawingColor, 2)
        return sourceFrame, boundingBoxes

    def _drawRectangle(self, sourceFrame, boundingBox, drawingColor, l=30, t=5, rt=1):
        xInitialPoint, yInitialPoint, width, height = boundingBox
        xDiagonalPoint, yDiagonalPoint = xInitialPoint + width, yInitialPoint + height
        cv.rectangle(sourceFrame, boundingBox, drawingColor, rt)

        # Top left
        cv.line(sourceFrame, (xInitialPoint, yInitialPoint),
                (xInitialPoint+l, yInitialPoint), drawingColor, t)
        cv.line(sourceFrame, (xInitialPoint, yInitialPoint),
                (xInitialPoint, yInitialPoint+l), drawingColor, t)

        # Top right
        cv.line(sourceFrame, (xDiagonalPoint, yInitialPoint),
                (xDiagonalPoint-l, yInitialPoint), drawingColor, t)
        cv.line(sourceFrame, (xDiagonalPoint, yInitialPoint),
                (xDiagonalPoint, yInitialPoint+l), drawingColor, t)

        # Bottom left
        cv.line(sourceFrame, (xInitialPoint, yDiagonalPoint),
                (xInitialPoint+l, yDiagonalPoint), drawingColor, t)
        cv.line(sourceFrame, (xInitialPoint, yDiagonalPoint),
                (xInitialPoint, yDiagonalPoint-l), drawingColor, t)

        # Bottom right
        cv.line(sourceFrame, (xDiagonalPoint, yDiagonalPoint),
                (xDiagonalPoint-l, yDiagonalPoint), drawingColor, t)
        cv.line(sourceFrame, (xDiagonalPoint, yDiagonalPoint),
                (xDiagonalPoint, yDiagonalPoint-l), drawingColor, t)

        return sourceFrame

    def _drawDefault(self, sourceFrame):
        if self.outcome.detections:
            for detection in self.outcome.detections:
                self._mpDraw.draw_detection(sourceFrame, detection)
        return sourceFrame


class Pose(DetectorTemplate):
    def __init__(self, upperBody=False, detectionConfidence=0.5, trackingConfidence=0.5):
        self._upperBody = upperBody
        self._detectionCon = detectionConfidence
        self._trackCon = trackingConfidence
        self._smooth = True
        self._mpDraw = mp.solutions.drawing_utils
        self._mpPose = mp.solutions.pose
        self._pose = self._mpPose.Pose(self)

    def detect(self, sourceFrame, drawOnPose=True):
        imageRGB = cv.cvtColor(sourceFrame, cv.COLOR_BGR2RGB)
        self.outcome = self._pose.process(imageRGB)
        if drawOnPose and self.outcome.pose_landmarks:
            self._mpDraw.draw_landmarks(
                sourceFrame, self.outcome.pose_landmarks, self._mpPose.POSE_CONNECTIONS)
        return sourceFrame

    def locate(self, sourceFrame, drawOnPose=True, drawingColor=(0, 255, 0), drawingSize=6):
        landmarksList = []
        if self.outcome.pose_landmarks:
            for id, landmark in enumerate(self.outcome.pose_landmarks.landmark):
                row, column, channel = sourceFrame.shape
                xPosition, yPosition = int(
                    landmark.x * column), int(landmark.y * row)
                landmarksList.append([id, xPosition, yPosition])
                if drawOnPose:
                    cv.circle(sourceFrame, (xPosition, yPosition),
                              drawingSize, drawingColor, cv.FILLED)
        return landmarksList


class Hand(DetectorTemplate):
    def __init__(self, handsMax=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self._handsMax = handsMax
        self._detectionCon = detectionConfidence
        self._trackCon = trackingConfidence
        self._mpHands = mp.solutions.hands
        self._hands = self._mpHands.Hands(self)
        self._mpDraw = mp.solutions.drawing_utils

    def detect(self, sourceFrame, drawOnHand=True):
        imageRGB = cv.cvtColor(sourceFrame, cv.COLOR_BGR2RGB)
        self.outcome = self._hands.process(imageRGB)

        if self.outcome.multi_hand_landmarks:
            for handLandmraks in self.outcome.multi_hand_landmarks:
                if drawOnHand:
                    self._mpDraw.draw_landmarks(
                        sourceFrame, handLandmraks, self._mpHands.HAND_CONNECTIONS)
        return sourceFrame

    def locate(self, sourceFrame, handsNumber=1, drawOnHand=True, drawingColor=(0, 255, 0), drawingSize=6):
        landmarksList = []

        if self.outcome.multi_hand_landmarks:
            if handsNumber >= 0 and handsNumber <= 10:
                try:
                    if handsNumber in [1, 0]:
                        hand = self.outcome.multi_hand_landmarks[0]
                    else:
                        hand = self.outcome.multi_hand_landmarks[handsNumber-1]
                except IndexError as err:
                    hand = self.outcome.multi_hand_landmarks[0]
                    print(
                        "HandsNumberError: Not enough hands are showing up, check out `handsNumber` parameter")
            else:
                print(
                    "BigHandsNumber: Detecting 1 hand now, make `handsNumber` smaller")
                hand = self.outcome.multi_hand_landmarks[0]

            for id, landmark in enumerate(hand.landmark):
                row, column, channel = sourceFrame.shape
                xPosition, yPosition = int(
                    landmark.x*column), int(landmark.y*row)
                landmarksList.append([id, xPosition, yPosition])
                if drawOnHand:
                    cv.circle(sourceFrame, (xPosition, yPosition),
                              drawingSize, drawingColor, cv.FILLED)
        return landmarksList
