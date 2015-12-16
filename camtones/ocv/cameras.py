import cv2

from .frames import Frame


class Camera:
    def __init__(self, id_or_filename):
        try:
            self._camera = cv2.VideoCapture(int(id_or_filename))
        except ValueError:
            self._camera = cv2.VideoCapture(id_or_filename)

    @property
    def frames(self):
        return self._camera.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def fps(self):
        return self._camera.get(cv2.CAP_PROP_FPS)

    @property
    def frame_width(self):
        return self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)

    @property
    def frame_height(self):
        return self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @property
    def current_pos(self):
        return self._camera.get(cv2.CAP_PROP_POS_MSEC)

    @property
    def current_frame(self):
        return self._camera.get(cv2.CAP_PROP_POS_FRAMES)

    def seek(self, frame):
        self._camera.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def read(self):
        (grabbed, frame) = self._camera.read()
        if grabbed:
            frame = Frame(frame)
        return (grabbed, frame)

    def __del__(self):
        self._camera.release()
