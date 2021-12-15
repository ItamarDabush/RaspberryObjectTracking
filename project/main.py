import cv2
import sys
sys.path.append("Desktop/project")
from project.VideoStream import VideoStream

def mark_detected_object(frame, center, width, height):
    first_corner = (center[0]-width/2, center[1]-height/2)
    opposite_corner = (center[0]+width/2, center[1]+height/2)
    cv2.rectangle(frame, first_corner, opposite_corner, (0,255,0), thickness=2)


def on_new_frame(frame):
    cv2.imshow("Card Detector", frame)


def main():
    videostream = VideoStream(on_new_frame)
    videostream.start()


if __name__ == "__main__":
    main()
