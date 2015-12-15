import cv2
import time

from camtones.contours import Contour
from camtones.frames import Frame
from camtones.windows import CamtonesWindow


class MotionBaseProcess:
    def __init__(self, video, debug):
        self.video = video
        self.debug = debug

        try:
            self.camera = cv2.VideoCapture(int(self.video))
        except ValueError:
            self.camera = cv2.VideoCapture(self.video)

        self.fgbg = cv2.createBackgroundSubtractorMOG2()

        if self.debug:
            self.fgmask_window = CamtonesWindow("FGMASK")

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

    def is_moving(self, frame, with_contours=True):
        fgmask = self.fgbg.apply(frame.frame)
        if self.blur:
            fgmask = cv2.blur(fgmask, (self.blur, self.blur))
        fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)[1]

        if self.debug:
            self.fgmask_window.show(Frame(fgmask))

        cnts = self.get_contours(fgmask)

        moving = False
        included_cnts = []
        for c in cnts:
            contour = Contour(c)
            if self.exclude_frame(contour, frame):
                continue

            if with_contours:
                included_cnts.append(contour)
                moving = True
            else:
                return (True, [])

        return (moving, included_cnts)


class MotionDetectProcess(MotionBaseProcess):
    def __init__(self, video, debug, exclude, resize, blur):
        super().__init__(video, debug)
        self.exclude = exclude
        self.resize = resize
        self.blur = blur

        self.window = CamtonesWindow("Motion extract")

    def __del__(self):
        self.camera.release()

    def process_frame(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            return False

        frame = Frame(frame)

        if self.resize:
            frame = frame.resize(width=self.resize)

        (moving, cnts) = self.is_moving(frame, with_contours=True)

        if moving:
            for contour in cnts:
                frame.draw_rect(contour.point1, contour.point2, (0, 255, 0))
                frame.draw_top_label("Moving", (0, 0, 255))
        else:
            frame.draw_top_label("Not Moving", (0, 255, 0))

        self.window.show(frame)

        return self.window.handle(self.camera)


class MotionExtractProcess(MotionBaseProcess):
    def __init__(self, video, debug, exclude, output, progress, resize, blur, show_time):
        super().__init__(video, debug)
        self.exclude = exclude
        self.resize = resize
        self.blur = blur
        self.output = output
        self.show_time = show_time
        self.progress = progress

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

        (moving, _) = self.is_moving(miniframe, with_contours=False)

        if moving:
            if self.show_time:
                current_time = time.gmtime(int(current_msec_pos/1000))
                frame.draw_timer(current_time)

            self.output.write(frame.frame)

        if self.progress:
            percentage = (current_frame * 100) / self.total_frames
            if int(self.last_percentage) != int(percentage):
                print("{}%".format(int(percentage)))
                self.last_percentage = percentage

        return True


class MotionExtractEDLProcess(MotionBaseProcess):
    def __init__(self, video, debug, exclude, output, progress, resize, blur):
        super().__init__(video, debug)
        self.exclude = exclude
        self.resize = resize
        self.blur = blur
        self.output = output
        self.progress = progress

        self.output = open(output, "w")
        self.start_silence = 0

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

        (moving, _) = self.is_moving(miniframe, with_contours=False)

        if not moving and self.start_silence is None:
            self.start_silence = current_msec_pos/1000

        if moving and self.start_silence is not None:
            end_silence = current_msec_pos/1000
            self.output.write("{} {} {}\n".format(self.start_silence, end_silence, 0))
            self.start_silence = None

        if self.progress:
            percentage = (current_frame * 100) / self.total_frames
            if int(self.last_percentage) != int(percentage):
                print("{}%".format(int(percentage)))
                self.last_percentage = percentage

        return True
