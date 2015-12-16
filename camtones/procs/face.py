import os
from progressbar import ProgressBar, Bar, ETA

from camtones.ocv import api as ocv


class FaceBaseProcess(object):
    def __init__(self, video, debug, classifier):
        self.debug = debug

        self.camera = ocv.get_camera(video)
        self.faceCascade = ocv.get_face_extractor(classifier)

    def run(self):
        while True:
            if not self.process_frame():
                break


class FaceDetectProcess(FaceBaseProcess):
    def __init__(self, video, debug, classifier):
        super(FaceDetectProcess, self).__init__(video, debug, classifier)
        self.window = ocv.get_window("Face detect")

    def process_frame(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            return False

        faces = self.faceCascade.get_faces(frame)

        for (x, y, w, h) in faces:
            frame.draw_rect((x, y), (x + w, y + h), (0, 255, 0))

        self.window.show(frame)
        return self.window.handle(self.camera)


class FaceExtractProcess(FaceBaseProcess):
    def __init__(self, video, debug, output, classifier, progress):
        super(FaceExtractProcess, self).__init__(video, debug, classifier)
        self.output = output

        self.progress = None
        if progress:
            widgets = [Bar('>'), ' ', ETA()]
            self.progress = ProgressBar(widgets=widgets, max_value=self.camera.frames)

    def process_frame(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            return False

        faces = self.faceCascade.get_faces(frame)

        counter = 0
        for (x, y, w, h) in faces:
            counter += 1
            croped = frame.crop_copy((x, y), (x + w, y + h))
            filename = os.path.join(self.output, "{}-{}.png".format(self.camera.current_pos, counter))
            croped.write(filename)

        if self.progress:
            self.progress.update(self.camera.current_frame)

        return True
