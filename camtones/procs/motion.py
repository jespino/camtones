import time

from camtones.ocv import api as ocv


class MotionBaseProcess(object):
    def __init__(self, video, debug, subtractor):
        self.debug = debug

        self.camera = ocv.get_camera(video)
        self.fgbg = ocv.get_background_subtractor(subtractor)

        if self.debug:
            self.fgmask_window = ocv.get_window("FGMASK")

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

    def is_moving(self, frame, with_contours=True):
        fgmask = self.fgbg.apply(frame)
        if self.blur:
            fgmask.blur(self.blur)
        fgmask.threshold(128)

        if self.debug:
            self.fgmask_window.show(fgmask)

        cnts = fgmask.get_contours()

        moving = False
        included_cnts = []
        for contour in cnts:
            if self.exclude_frame(contour, frame):
                continue

            if with_contours:
                included_cnts.append(contour)
                moving = True
            else:
                return (True, [])

        return (moving, included_cnts)


class MotionDetectProcess(MotionBaseProcess):
    def __init__(self, video, debug, exclude, resize, blur, subtractor):
        super(MotionDetectProcess, self).__init__(video, debug, subtractor)
        self.exclude = exclude
        self.resize = resize
        self.blur = blur

        self.window = ocv.get_window("Motion extract")

    def process_frame(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            return False

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
    def __init__(self, video, debug, exclude, output, progress, resize, blur, show_time, subtractor):
        super(MotionExtractProcess, self).__init__(video, debug, subtractor)
        self.exclude = exclude
        self.resize = resize
        self.blur = blur
        self.output = output
        self.show_time = show_time
        self.progress = progress

        self.output = ocv.get_video_writer(self.camera, output)

        self.last_percentage = 0

    def process_frame(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            return False

        if self.resize:
            miniframe = frame.resize(width=self.resize)
        else:
            miniframe = frame

        (moving, _) = self.is_moving(miniframe, with_contours=False)

        if moving:
            if self.show_time:
                current_time = time.gmtime(int(self.camera.current_pos/1000))
                frame.draw_timer(current_time)

            self.output.write(frame.frame)

        if self.progress:
            percentage = (self.camera.current_frame * 100) / self.camera.frames
            if int(self.last_percentage) != int(percentage):
                print("{}%".format(int(percentage)))
                self.last_percentage = percentage

        return True


class MotionExtractEDLProcess(MotionBaseProcess):
    def __init__(self, video, debug, exclude, output, progress, resize, blur, subtractor):
        super(MotionExtractProcess, self).__init__(video, debug, subtractor)
        self.exclude = exclude
        self.resize = resize
        self.blur = blur
        self.output = output
        self.progress = progress

        self.output = open(output, "w")
        self.start_silence = 0
        self.last_percentage = 0

    def __del__(self):
        self.camera.release()

    def process_frame(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            return False

        if self.resize:
            miniframe = frame.resize(width=self.resize)
        else:
            miniframe = frame

        (moving, _) = self.is_moving(miniframe, with_contours=False)

        if not moving and self.start_silence is None:
            self.start_silence = self.camera.current_pos/1000

        if moving and self.start_silence is not None:
            end_silence = self.camera.current_pos/1000
            self.output.write("{} {} {}\n".format(self.start_silence, end_silence, 0))
            self.start_silence = None

        if self.progress:
            percentage = (self.camera.current_frame * 100) / self.camera.frames
            if int(self.last_percentage) != int(percentage):
                print("{}%".format(int(percentage)))
                self.last_percentage = percentage

        return True
