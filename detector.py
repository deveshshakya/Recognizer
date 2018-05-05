import numpy
import cv2


def convertToGRAY(image):
    """Returns gray-sacle image of input image."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def widthHeightDividedBy(image, value):
    """Divides width and height of an image by a given value."""

    w, h = image.shape[:2]
    return int(w/value), int(h/value)


def detectFace(image):
    """Detects face in the image and returns only face region in gray-scale and face coordinates."""

    gray = convertToGRAY(image)  # detection is works well on gray-scale image.
    #  load the face classifier
    faceClassifier = cv2.CascadeClassifier('haarcascades\\haarcascade_frontalface_alt.xml')

    if faceClassifier.empty():  # handling the classifier empty error.
        print("Your cascade is empty.")
        exit()

    minSize = widthHeightDividedBy(gray, 8)  # getting min size for detection.
    faces = faceClassifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE,
                                            minSize=minSize)
    if len(faces) == 0:  # checking if faces found or not.
        return None, None

    x, y, w, h = faces[0]  # considering that there will be only one face in the image.
    return gray[x:x+w, y:y+h], faces[0]
