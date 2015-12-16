import cv2


class VideoWriter:
    def __init__(self, video_writer):
        self._video_writer = video_writer

    def write(self, frame):
        self._video_writer.write(frame.frame)
