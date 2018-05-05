import argparse
import numpy
import cv2
import cv2.face
from detector import detectFace
from trainer import training


subjects = ['']  # add names of the people


def drawRect(image, rect):
    """Draws rectangle by the given coordinates."""

    x, y, w, h = rect   # rect is a tuple
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


def drawText(image, text, x, y):
    """Write text, above the given coordinates."""

    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(image, face_recognizer):
    """Predicts face on the image and returns the image."""

    try:
        img = image.copy()  # getting the copy of image so that original image doesn't get affected.
    except AttributeError:
        print("Test image is empty.")
        exit()

    face, rect = detectFace(img)

    label, confidence = face_recognizer.predict(face)
    labelText = subjects[label]
    drawRect(img, rect)
    drawText(img, labelText, rect[0], rect[1] - 5)

    return img


#  starts main program.
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This scripts trains the recognizer with available faces. This script'
                                                 'should be run before going on recognizing.')
    parser.add_argument('dataDir_Name', type=str, help='Please input path of the training images.')
    parser.add_argument('testImage_Dir', type=str, help='Please input path of the test images.')
    args = parser.parse_args()
    dataDir_Name = args.dataDir_Name
    testImage_Dir = args.testImage_Dir

    # Starts training.
    print("Preparing Data.")
    faces, labels = training(dataDir_Name)  # stores faces and the associated labels for Recognizer training
    print("Data Prepared.")
    print("Total faces: ", len(faces))  # returns total available faces
    print("Total labels: ", len(labels))
    # Training over.

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # creates object of LBPHFaceRecognizer class.
    face_recognizer.train(faces, numpy.array(labels))  # trains the recognizer
    # face_recognizer.read("LBPHFace.yml")
    # face_recognizer.write("LBPHFace.yml")

    test_image = cv2.imread(testImage_Dir)  # opening the test image
    predictImg = predict(test_image, face_recognizer)  # predicts the image. Love this part.

    print("Prediction Complete.")
    cv2.imshow('DetectedImg', cv2.resize(predictImg, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
