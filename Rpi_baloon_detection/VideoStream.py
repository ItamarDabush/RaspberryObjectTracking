from threading import Thread
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10


class VideoStream:
    def __init__(self, on_new_frame, width=IM_WIDTH, height=IM_HEIGHT, framerate=FRAME_RATE):
        self.on_new_frame = on_new_frame
        self.there_is_a_new_frame = False

        self.camera = PiCamera()
        self.camera.resolution = (width, height)
        self.camera.framerate = framerate

        self.rawCapture = PiRGBArray(self.camera, size=(width, height))
        self.stream = self.camera.capture_continuous(
            self.rawCapture, format="bgr", use_video_port=True)

        self.frame = []

    def start(self):
        Thread(target=self.update, args=()).start()
        time.sleep(1)
        
        while True:
            if self.there_is_a_new_frame:
                self.there_is_a_new_frame = False
                self.on_new_frame(self.frame)


    def update(self):
        for frame in self.stream:
            self.frame = frame.array
            self.rawCapture.truncate(0)
            self.there_is_a_new_frame = True

