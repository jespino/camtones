import cv2
import time

from camtones.contours import Contour
from camtones.frames import Frame
from camtones.windows import CamtonesWindow


class MotionBaseProcess:
    def exclude_frame(self, contour, frame):
        data = {
            "contour": contour,
            "frame": frame,
        }

        return self.exclude and eval(self.exclude, {}, data)

    def run(self):
        while True:
            if not self.process_frame():
                break

    def get_contours(self, mask):
        (cnts, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        return cnts


class MotionDetectProcess(MotionBaseProcess):
    def __init__(self, video, debug, exclude, resize, blur):
        self.video = video
        self.debug = debug
        self.exclude = exclude
        self.resize = resize
        self.blur = blur

        try:
            self.camera = cv2.VideoCapture(int(self.video))
        except ValueError:
            self.camera = cv2.VideoCapture(self.video)

        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.window = CamtonesWindow("Motion extract")

        if self.debug:
            self.fgmask_window = CamtonesWindow("FGMASK")

    def __del__(self):
        self.camera.release()

    def process_frame(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            return False

        frame = Frame(frame)

        if self.resize:
            frame = frame.resize(width=self.resize)

        fgmask = self.fgbg.apply(frame.frame)
        if self.blur:
            fgmask = cv2.blur(fgmask, (self.blur, self.blur))
        fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)[1]
        if self.debug:
            self.fgmask_window.show(Frame(fgmask))

        cnts = self.get_contours(fgmask)

        moving = False
        for c in cnts:
            contour = Contour(c)
            if self.exclude_frame(contour, frame):
                continue

            frame.draw_rect(contour.point1, contour.point2, (0, 255, 0))
            frame.draw_top_label("Moving", (0, 0, 255))
            moving = True

        if not moving:
            frame.draw_top_label("Not Moving", (0, 255, 0))

        self.window.show(frame)

        return self.window.handle(self.camera)


class MotionExtractProcess(MotionBaseProcess):
    def __init__(self, video, debug, exclude, output, progress, resize, blur, show_time):
        self.video = video
        self.exclude = exclude
        self.resize = resize
        self.blur = blur
        self.output = output
        self.show_time = show_time
        self.progress = progress
        self.debug = debug

        try:
            self.camera = cv2.VideoCapture(int(self.video))
        except ValueError:
            self.camera = cv2.VideoCapture(self.video)

        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.window = CamtonesWindow("Motion extract")

        fps = self.camera.get(cv2.CAP_PROP_FPS)
        size = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output = cv2.VideoWriter(output, fourcc, fps, size)

        self.total_frames = self.camera.get(cv2.CAP_PROP_FRAME_COUNT)
        self.last_percentage = 0

    def __del__(self):
        self.camera.release()

    def process_frame(self):
        (grabbed, frame) = self.camera.read()
        current_msec_pos = self.camera.get(cv2.CAP_PROP_POS_MSEC)
        current_frame = self.camera.get(cv2.CAP_PROP_POS_FRAMES)

        if not grabbed:
            return False

        frame = Frame(frame)

        if self.resize:
            miniframe = frame.resize(width=self.resize)
        else:
            miniframe = frame

        fgmask = self.fgbg.apply(miniframe.frame)
        if self.blur:
            fgmask = cv2.blur(fgmask, (self.blur, self.blur))
        fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)[1]

        cnts = self.get_contours(fgmask)

        for c in cnts:
            contour = Contour(c)
            if self.exclude_frame(contour, frame):
                continue

            if self.show_time:
                current_time = time.gmtime(int(current_msec_pos/1000))
                frame.draw_timer(current_time)

            self.output.write(frame.frame)
            break

        if self.progress:
            percentage = (current_frame * 100) / self.total_frames
            if int(self.last_percentage) != int(percentage):
                print("{}%".format(int(percentage)))
                self.last_percentage = percentage

        return True
