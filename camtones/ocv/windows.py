import cv2


class Window:
    def __init__(self, name):
        self.name = name
        self.extra_handlers = {}
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    def show(self, frame):
        cv2.imshow(self.name, frame.frame)

    def add_handler(self, key, handler):
        self.extra_handlers[key] = handler

    @property
    def is_fullscreen(self):
        return cv2.getWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN

    def to_normal(self):
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    def to_fullscreen(self):
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def handle(self, camera):
        key = cv2.waitKey(1) & 0xFF

        if key in self.extra_handlers:
            return self.extra_handlers[key](camera)

        if key == ord("q"):
            if camera.frames != -1:
                camera.seek(camera.frames)
            else:
                return False
        elif key == ord("f"):
            if self.is_fullscreen:
                self.to_normal()
            else:
                self.to_fullscreen()
        elif key == 81:  # left
            camera.seek(max(camera.current_frame - 100, 0))
        elif key == 82:  # up
            camera.seek(min(camera.current_frame + 1000, camera.frames))
        elif key == 83:  # right
            camera.seek(min(camera.current_frame + 100, camera.frames))
        elif key == 84:  # down
            camera.seek(max(camera.current_frame - 1000, 0))
        return True

    def __del__(self):
        cv2.destroyWindow(self.name)
