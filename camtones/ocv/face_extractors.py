import cv2


class FaceExtractor:
    def __init__(self, classifier):
        self._face_extractor = cv2.CascadeClassifier(classifier)

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
