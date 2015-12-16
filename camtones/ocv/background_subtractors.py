import cv2

from .frames import Frame
from .exceptions import InvalidBackgrounSubtractor


class BackgroundSubtractor:
    def __init__(self, subtractor="MOG2"):
        if subtractor == "MOG2":
            self._subtractor = cv2.createBackgroundSubtractorMOG2()
        elif subtractor == "KNN":
            self._subtractor = cv2.createBackgroundSubtractorKNN()
        else:
            raise InvalidBackgrounSubtractor(subtractor)

    def apply(self, frame):
        result_frame = self._subtractor.apply(frame.frame)

        return Frame(result_frame)
