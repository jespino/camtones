import time
import gi
import os

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, Gdk, GLib, GObject

from progressbar import ProgressBar, Bar, ETA

from camtones.ocv import api as ocv


class MotionBaseProcess(object):
    def __init__(self, video, debug, subtractor, blur, resize, exclude, threshold):
        self.debug = debug
        self.exclude = exclude
        self.resize = resize
        self.blur = blur
        self.threshold = threshold

        self.camera = ocv.get_camera(video)
        self.fgbg = ocv.get_background_subtractor(subtractor)

        if self.debug:
            self.fgmask_window = ocv.get_window("FGMASK")

    def exclude_frame(self, contour, frame):
        data = {
            "rect": contour,
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

        if self.threshold:
            fgmask.threshold(self.threshold)

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
    def __init__(self, video, debug, exclude, resize, blur, threshold, subtractor):
        super(MotionDetectProcess, self).__init__(video, debug, subtractor, blur, resize, exclude, threshold)

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
    def __init__(self, video, debug, exclude, output, progress, resize, blur, threshold, show_time, subtractor):
        super(MotionExtractProcess, self).__init__(video, debug, subtractor, blur, resize, exclude, threshold)
        self.output = output
        self.show_time = show_time
        self.progress = None
        if progress:
            widgets = [Bar('>'), ' ', ETA()]
            self.progress = ProgressBar(widgets=widgets, max_value=self.camera.frames)

        self.output = ocv.get_video_writer(self.camera, output)

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

            self.output.write(frame)

        if self.progress:
            self.progress.update(self.camera.current_frame)

        return True


class MotionExtractEDLProcess(MotionBaseProcess):
    def __init__(self, video, debug, exclude, output, progress, resize, blur, threshold, subtractor):
        super(MotionExtractEDLProcess, self).__init__(video, debug, subtractor, blur, resize, exclude, threshold)
        self.output = output

        self.progress = None
        if progress:
            widgets = [Bar('>'), ' ', ETA()]
            self.progress = ProgressBar(widgets=widgets, max_value=self.camera.frames)

        self.output = open(output, "w")
        self.start_silence = 0

    def __del__(self):
        if self.progress:
            self.progress.finish()

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
            self.progress.update(self.camera.current_frame)

        return True


class MotionCalibrate(object):
    def __init__(self, video):
        self.ui = UI(video)

    def run(self):
        self.ui.main()


class FrameProcessor:
    def __init__(self, camera, orig_image, mask_image):
        self.camera = camera
        self.mog2 = ocv.get_background_subtractor("MOG2")
        self.knn = ocv.get_background_subtractor("KNN")

        self.subtractor_name = "MOG2"
        self.subtractor = self.mog2

        self.blur = 0
        self.threshold = 128
        self.resize = None
        self.orig_image = orig_image
        self.mask_image = mask_image
        self.exclude = ""


    def frame_to_gtk_pixbuf(self, frame, is_gray):
        if is_gray:
            color_frame = frame.gray_to_rgb_copy()
        else:
            color_frame = frame.bgr_to_rgb_copy()

        data = color_frame.to_string()
        return GdkPixbuf.Pixbuf.new_from_data(data, GdkPixbuf.Colorspace.RGB, False, 8, frame.width, frame.height, frame.width*3, None, None)



    def extract_and_draw(self, frame, subtractor, detection, mask):
        if self.resize:
            frame = frame.resize(self.resize)

        result_mask = subtractor.apply(frame)
        if self.blur:
            result_mask.blur(self.blur)
        if self.threshold:
            result_mask.threshold(self.threshold)

        mask.set_from_pixbuf(self.frame_to_gtk_pixbuf(result_mask, True).copy())

        cnts = result_mask.get_contours()
        result_frame = frame.clone()

        included_cnts = []
        for contour in cnts:
            data = {
                "rect": contour,
                "frame": frame,
            }

            try:
                if self.exclude and eval(self.exclude, {}, data):
                    continue
            except Exception:
                pass

            result_frame.draw_rect(contour.point1, contour.point2, (0, 255, 0))

        detection.set_from_pixbuf(self.frame_to_gtk_pixbuf(result_frame, False).copy())

    def run(self):
        (grabbed, frame) = self.camera.read()
        if not grabbed:
            self.camera.seek(0)
            (_, frame) = self.camera.read()

        self.extract_and_draw(frame, self.subtractor, self.orig_image, self.mask_image)


class UI:
    def __init__(self, video):
        self.command = None
        self.builder = Gtk.Builder()
        self.builder.add_from_file(os.path.join(os.path.dirname(__file__), "gui.glade"))

        orig_image = self.builder.get_object("image_orig")
        mask_image = self.builder.get_object("image_mask")

        self.video = video
        camera = ocv.get_camera(video)
        self.frame_processor = FrameProcessor(camera, orig_image, mask_image)

        self.builder.connect_signals(self)

        window = self.builder.get_object("window")
        window.show_all()
        blur_spin = self.builder.get_object("blur_spin")
        blur_spin.set_value(0)
        threshold_spin = self.builder.get_object("threshold_spin")
        threshold_spin.set_value(128)
        resize_spin = self.builder.get_object("resize_spin")
        resize_spin.set_value(0)


    def on_quit(self, widget, event):
        Gtk.main_quit()

    def on_blur_change(self, widget, *args):
        try:
            new_blur = int(widget.get_value())
            self.frame_processor.blur = new_blur
        except ValueError:
            pass

    def on_resize_change(self, widget, *args):
        try:
            new_resize = int(widget.get_value())
            if new_resize > 0:
                self.frame_processor.resize = new_resize
            else:
                self.frame_processor.resize = None
        except ValueError:
            pass

    def on_threshold_change(self, widget, *args):
        try:
            new_threshold = int(widget.get_value())
            self.frame_processor.threshold = new_threshold
        except ValueError:
            pass

    def on_exclude_change(self, widget, *args):
        self.frame_processor.exclude = widget.get_text()

    def on_generate_motion_edl_clicked(self, widget, *args):
        self.command = MotionExtractEDLProcess(
            video=self.video,
            subtractor=self.frame_processor.subtractor_name,
            blur=self.frame_processor.blur,
            exclude=self.frame_processor.exclude,
            resize=self.frame_processor.resize,
            debug=False,
            progress=True,
            threshold=self.frame_processor.threshold,
            output="{}.edl".format(self.video)
        )
        self.builder.get_object("window").hide()
        self.builder.get_object("window").destroy()

        Gtk.main_quit()

    def on_generate_motion_video_clicked(self, widget, *args):
        self.command = MotionExtractProcess(
            video=self.video,
            subtractor=self.frame_processor.subtractor_name,
            blur=self.frame_processor.blur,
            exclude=self.frame_processor.exclude,
            resize=self.frame_processor.resize,
            debug=False,
            progress=True,
            show_time=True,
            threshold=self.frame_processor.threshold,
            output="{}-out.avi".format(self.video)
        )
        self.builder.get_object("window").hide()
        self.builder.get_object("window").destroy()

        Gtk.main_quit()

    def on_subtractor_toggle(self, widget, *args):
        if widget.get_active():
            if widget.get_label() == "MOG2":
                self.frame_processor.subtractor = self.frame_processor.mog2
                self.frame_processor.subtractor_name = "MOG2"
            elif widget.get_label() == "KNN":
                self.frame_processor.subtractor = self.frame_processor.knn
                self.frame_processor.subtractor_name = "KNN"

    def on_timeout(self, *args):
        self.frame_processor.run()
        GObject.timeout_add(1, self.on_timeout, None)

    def main(self):
        counter = 0
        self.on_timeout()

        Gtk.main()

        if self.command:
            self.command.run()

