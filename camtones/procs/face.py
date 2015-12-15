from camtones.ocv import api as ocv

class FaceBaseProcess:
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
        super().__init__(video, debug, classifier)
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
    def __init__(self, video, debug, output, classifier):
        super().__init__(video, debug, classifier)
        self.output = output

    def process_frame(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            return False

        faces = self.faceCascade.get_faces(frame)

        counter = 0
        for (x, y, w, h) in faces:
            counter += 1
            croped = frame.crop_copy((x, y), (x + w, y + h))
            filename = "{}/{}-{}.png".format(self.output, self.camera.current_pos, counter)
            ocv.write_frame(filename)

        return True
