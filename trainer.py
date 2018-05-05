import os
import cv2
from detector import detectFace


def training(dataDir_Name):
    """Gather the available faces data to return faces, and labels .
       Each person's faces folder name should be follow the following convention."""

    """ dataDir_Name --->
            ---> s1
            ---> s2
            and so on...
    """

    faces = []  # for storing faces
    labels = []  # for storing labels
    dirs = os.listdir(dataDir_Name)
    for dir in dirs:
        if not dir.startswith('s'):  # check for faces folder name is not matched the convention then skip.
            continue
        label = int(dir.replace('s', ''))  # getting label names
        subject_dir = dataDir_Name + '/' + dir   # getting subjects directory
        subject_images = os.listdir(subject_dir)  # getting subjects images
        for image in subject_images:  # iterating through all the available images.
            if image.startswith('.'):
                continue
            img_path = subject_dir + '/' + image
            print(img_path)
            person = cv2.imread(img_path)
            face, rect = detectFace(person)
            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels
