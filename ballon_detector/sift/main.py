import cv2


def main():

    # Loading the image
    img = cv2.imread('../resources/ballon.jpg')

    # Converting image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying SIFT detector
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)

    # Marking the keypoint on the image using circles
    img = cv2.drawKeypoints(gray,
                            kp,
                            img,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("image", img)
    cv2.waitKey(100000)


if __name__ == "__main__":
    main()
