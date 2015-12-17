import cv2


class Contour:
    def __init__(self, contour):
        self.contour = contour
        (self.x, self.y, self.w, self.h) = cv2.boundingRect(self.contour)
        self.area = cv2.contourArea(self.contour)

    @property
    def width(self):
        return self.w

    @property
    def height(self):
        return self.h

    @property
    def point1(self):
        return (self.x, self.y)

    @property
    def point2(self):
        return (self.x + self.w, self.y + self.h)
