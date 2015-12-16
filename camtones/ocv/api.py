import cv2
import os
import re

from .contours import Contour
from .frames import Frame
from .windows import Window
from .cameras import Camera
from .bgextractors import BackgroundExtractor
from .video_writers import VideoWriter
from .face_extractors import FaceExtractor, HAARS_DIRECTORY


def get_camera(id_or_filename):
    return Camera(id_or_filename)


def get_background_extractor(extractor_type):
    return BackgroundExtractor(extractor_type)


def get_face_extractor(extractor_classifier):
    return FaceExtractor(extractor_classifier)


def get_video_writer(camera, filename):
    fps = camera.get(cv2.CAP_PROP_FPS)
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return VideoWriter(cv2.VideoWriter(output, fourcc, fps, size))


def get_window(name):
    return Window(name)


def get_supported_subtractors():
    supported_subtractors = {}
    if hasattr(cv2, "createBackgroundSubtractorGMG"):
        supported_subtractors["GMG"] = cv2.createBackgroundSubtractorGMG
    if hasattr(cv2, "createBackgroundSubtractorKNN"):
        supported_subtractors["KNN"] = cv2.createBackgroundSubtractorKNN
    if hasattr(cv2, "createBackgroundSubtractorMOG"):
        supported_subtractors["MOG"] = cv2.createBackgroundSubtractorMOG
    if hasattr(cv2, "createBackgroundSubtractorMOG2"):
        supported_subtractors["MOG2"] = cv2.createBackgroundSubtractorMOG2

    return supported_subtractors


def get_stock_classifiers():
    for haar in os.listdir(HAARS_DIRECTORY):
        result = re.match("haarcascade_(\w+).xml", haar)
        if result:
            yield result.group(1)
