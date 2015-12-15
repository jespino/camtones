import cv2

from .contours import Contour
from .frames import Frame
from .windows import Window
from .cameras import Camera
from .bgextractors import BackgroundExtractor
from .video_writers import VideoWriter
from .face_extractors import FaceExtractor


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
