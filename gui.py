#!/usr/bin/python
from __future__ import print_function

import gi
import time
import os

gi.require_version('Gtk', '3.0')

from gi.repository import Gtk, GdkPixbuf

from camtones.ocv import api as ocv


def frame_to_gtk_pixbuf(frame, is_gray):
    if is_gray:
        color_frame = frame.gray_to_rgb_copy()
    else:
        color_frame = frame.bgr_to_rgb_copy()

    data = color_frame.to_string()
    return GdkPixbuf.Pixbuf.new_from_data(data, GdkPixbuf.Colorspace.RGB, False, 8, frame.width, frame.height, frame.width*3, None, None)


class FrameProcessor:
    def __init__(self, orig_image, mask_image):
        self.mog2 = ocv.get_background_subtractor("MOG2")
        self.knn = ocv.get_background_subtractor("KNN")

        self.subtractor_name = "MOG2"
        self.subtractor = self.mog2

        self.camera = ocv.get_camera("test.mp4")
        self.blur = 0
        self.threshold = 128
        self.resize = None
        self.orig_image = orig_image
        self.mask_image = mask_image
        self.exclude = ""

    def extract_and_draw(self, frame, subtractor, detection, mask):
        if self.resize:
            frame = frame.resize(self.resize)

        result_mask = subtractor.apply(frame)
        if self.blur:
            result_mask.blur(self.blur)
        if self.threshold:
            result_mask.threshold(self.threshold)

        mask.set_from_pixbuf(frame_to_gtk_pixbuf(result_mask, True).copy())

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

        detection.set_from_pixbuf(frame_to_gtk_pixbuf(result_frame, False).copy())

    def run(self):
        (grabbed, frame) = self.camera.read()
        if not grabbed:
            self.camera.seek(0)
            (_, frame) = self.camera.read()

        self.extract_and_draw(frame, self.subtractor, self.orig_image, self.mask_image)


class UI:
    def __init__(self):
        self.builder = Gtk.Builder()
        self.builder.add_from_file(os.path.join(os.path.dirname(__file__), "gui.glade"))

        orig_image = self.builder.get_object("image_orig")
        mask_image = self.builder.get_object("image_mask")

        self.frame_processor = FrameProcessor(orig_image, mask_image)

        self.builder.connect_signals(self)

        window = self.builder.get_object("window")
        window.show_all()
        blur_spin = self.builder.get_object("blur_spin")
        blur_spin.set_value(0)
        threshold_spin = self.builder.get_object("threshold_spin")
        threshold_spin.set_value(128)
        resize_spin = self.builder.get_object("resize_spin")
        resize_spin.set_value(0)

        self.quit = False

    def on_quit(self, widget, event):
        self.quit = True

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
        pass

    def on_generate_motion_video_clicked(self, widget, *args):
        pass

    def on_subtractor_toggle(self, widget, *args):
        if widget.get_active():
            if widget.get_label() == "MOG2":
                self.frame_processor.subtractor = self.frame_processor.mog2
                self.frame_processor.subtractor_name = "MOG2"
            elif widget.get_label() == "KNN":
                self.frame_processor.subtractor = self.frame_processor.knn
                self.frame_processor.subtractor_name = "KNN"

    def main(self):
        counter = 0
        while self.quit is False:
            Gtk.main_iteration_do(False)

            counter +=1
            if counter % 100 == 0:
                self.frame_processor.run()
                counter = 0
        print("--subtractor {} --blur {} --exclude \"{}\" --resize {} --threshold {}".format(
            self.frame_processor.subtractor_name,
            self.frame_processor.blur,
            self.frame_processor.exclude,
            self.frame_processor.resize,
            self.frame_processor.threshold
        ))



if __name__ == '__main__':
    ui = UI()
    ui.main()
