import os
import argparse
import numpy
import cv2
from recognizer import subjects


def createFaces(index):
    """Captures faces from camera and stores for training data. Only one face at a time works fine."""

    camera = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier('haarcascades\\haarcascade_frontalface_alt.xml')

    try:
        os.chdir('training-data')
        os.mkdir('s' + str(index))
    except FileNotFoundError:
        print(r'"data" folder is not found.')
    except FileExistsError:
        pass

    os.chdir('s' + str(index))

    img_index = 1

    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 1:
            cv2.imwrite(str(img_index) + '.jpg', frame)
            print('Image no: ', img_index)
            img_index += 1

        cv2.imshow('Detecting', numpy.fliplr(frame).copy())
        if cv2.waitKey(25) & 0xFF == ord('q'):
            camera.release()
            break


if __name__ == "__main__":
    """Run this script to capture faces."""

    parser = argparse.ArgumentParser(description='This scripts takes faces from webCam for training.')
    parser.add_argument('subject_name', type=str, help='Please input name of the person as string.')
    args = parser.parse_args()
    subjects.append(args.subject_name)
    index = len(subjects)
    createFaces(index)
    cv2.destroyAllWindows()
