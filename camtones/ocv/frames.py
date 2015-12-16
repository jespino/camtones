import cv2
import time
import copy
import imutils
import numpy

from .contours import Contour


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

    def get_contours(self):
        contours = []
        (cnts, _) = cv2.findContours(self.frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        for c in cnts:
            contours.append(Contour(c))
        return contours

    def blur(self, quantity):
        self.frame = cv2.blur(self.frame, (quantity, quantity))

    def threshold(self, minimun):
        self.frame = cv2.threshold(self.frame, minimun, 255, cv2.THRESH_BINARY)[1]

    def gray(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def gray_copy(self):
        return Frame(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY))

    def crop(self, point1, point2):
        self.frame = self.frame[point1[1]:point2[1], point1[0]:point2[0]]

    def crop_copy(self, point1, point2):
        return Frame(self.frame[point1[1]:point2[1], point1[0]:point2[0]])

    def write(self, filename):
        cv2.imwrite(filename, self.frame)

    def gray_to_rgb(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2RGB)

    def bgr_to_rgb(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def gray_to_rgb_copy(self):
        return Frame(cv2.cvtColor(self.frame, cv2.COLOR_GRAY2RGB))

    def bgr_to_rgb_copy(self):
        return Frame(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

    def to_string(self):
        return numpy.ndarray.tostring(self.frame)
