import cv2

from camtones.windows import CamtonesWindow
from camtones.frames import Frame


class FaceBaseProcess:
    def __init__(self, video, debug, classifier):
        self.video = video
        self.debug = debug
        self.classifier = classifier
        try:
            self.camera = cv2.VideoCapture(int(self.video))
        except ValueError:
            self.camera = cv2.VideoCapture(self.video)
        self.faceCascade = cv2.CascadeClassifier(self.classifier)

    def run(self):
        while True:
            if not self.process_frame():
                break

    def __del__(self):
        self.camera.release()

    def get_faces(self, frame):
        gray = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.HAAR_SCALE_IMAGE
        )

        return faces

class FaceDetectProcess(FaceBaseProcess):
    def __init__(self, video, debug, classifier):
        super().__init__(video, debug, classifier)
        self.window = CamtonesWindow("Face detect")

    def process_frame(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            return False

        frame = Frame(frame)

        faces = self.get_faces(frame)

        for (x, y, w, h) in faces:
            frame.draw_rect((x, y), (x + w, y + h), (0, 255, 0))

        self.window.show(frame)
        return self.window.handle(self.camera)


class FaceExtractProcess(FaceBaseProcess):
    def __init__(self, video, debug, output, classifier):
        super().__init__(video, debug, classifier)
        self.output = output

    def process_frame(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            return False

        frame = Frame(frame)

        faces = self.get_faces(frame)

        milisecond = self.camera.get(cv2.CAP_PROP_POS_MSEC)

        counter = 0
        for (x, y, w, h) in faces:
            counter += 1
            croped = frame.frame[y:y + h, x:x + w]
            cv2.imwrite("{}/{}-{}.png".format(self.output, milisecond, counter), croped)

        return True
