from picamera import PiCamera
from time import sleep

def main():
  print("Start Running")
  camera = PiCamera()
  camera.start_preview()
  sleep(5)
  camera.stop_preview()
if __name__ == "__main__":
  main()