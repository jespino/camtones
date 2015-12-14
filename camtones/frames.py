import cv2
import time
import copy
import imutils


class Frame:
    def __init__(self, frame):
        self.frame = frame

    @property
    def width(self):
        return len(self.frame[0])

    @property
    def height(self):
        return len(self.frame)

    @property
    def area(self):
        return self.width * self.height

    def draw_top_label(self, text, color):
        cv2.putText(
            self.frame,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    def draw_rect(self, point1, point2, color):
        cv2.rectangle(self.frame, point1, point2, color, 1)

    def draw_timer(self, time_struct):
        cv2.putText(
            self.frame,
            time.strftime("%H:%M:%S", time_struct),
            (10, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1
        )

    def resize(self, width):
        return Frame(imutils.resize(self.frame, width=width))

    def clone(self):
        return copy.deepcopy(self)
