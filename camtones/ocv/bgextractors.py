import cv2

from .frames import Frame


class BackgroundExtractor:
    def __init__(self, extractor_type="MOG2"):
        if extractor_type == "MOG2":
            self._extractor = cv2.createBackgroundSubtractorMOG2()
        elif extractor_type == "KNN":
            self._extractor = cv2.createBackgroundSubtractorKNN()

    def apply(self, frame):
        result_frame = self._extractor.apply(frame.frame)

        return Frame(result_frame)
