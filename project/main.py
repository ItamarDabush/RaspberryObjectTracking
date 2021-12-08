import cv2
from project.VideoStream import VideoStream


def on_new_frame(frame):
    cv2.imshow("Card Detector", frame)


def main():
    videostream = VideoStream(on_new_frame)
    videostream.start()


if __name__ == "__main__":
    main()
