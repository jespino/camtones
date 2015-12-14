import cv2


class CamtonesWindow:
    def __init__(self, name):
        self.name = name
        self.extra_handlers = {}
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    def show(self, frame):
        cv2.imshow(self.name, frame.frame)

    def add_handler(self, key, handler):
        self.extra_handlers[key] = handler

    def handle(self, camera):
        key = cv2.waitKey(1) & 0xFF

        if key in self.extra_handlers:
            return self.extra_handlers[key](camera)

        if key == ord("q"):
            total_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames != -1:
                camera.set(cv2.CAP_PROP_POS_FRAMES, total_frames)
            else:
                return False
        elif key == ord("f"):
            fullScreen = cv2.getWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN)
            if fullScreen == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == 81:  # left
            current_position = camera.get(cv2.CAP_PROP_POS_FRAMES)
            camera.set(cv2.CAP_PROP_POS_FRAMES, max(current_position - 100, 0))
        elif key == 82:  # up
            current_position = camera.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
            camera.set(cv2.CAP_PROP_POS_FRAMES, min(current_position + 1000, total_frames))
        elif key == 83:  # right
            current_position = camera.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
            camera.set(cv2.CAP_PROP_POS_FRAMES, min(current_position + 100, total_frames))
        elif key == 84:  # down
            current_position = camera.get(cv2.CAP_PROP_POS_FRAMES)
            camera.set(cv2.CAP_PROP_POS_FRAMES, max(current_position - 1000, 0))
        return True

    def __del__(self):
        cv2.destroyWindow(self.name)
