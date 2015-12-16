import cv2
import os

from .exceptions import InvalidHaarCascade

HAARS_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "haars")


class FaceExtractor:
    def __init__(self, classifier):
        if os.path.exists(classifier):
            self._face_extractor = cv2.CascadeClassifier(classifier)
        elif os.path.exists(os.path.join(HAARS_DIRECTORY, "haarcascade_{}.xml".format(classifier))):
            self._face_extractor = cv2.CascadeClassifier(
                os.path.join(HAARS_DIRECTORY, "haarcascade_{}.xml".format(classifier))
            )
        else:
            raise InvalidHaarCascade(classifier)

    def get_faces(self, frame):
        gray = frame.gray_copy()

        faces = self._face_extractor.detectMultiScale(
            gray.frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.HAAR_SCALE_IMAGE
        )
        return faces
