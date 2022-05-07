import cv2
import os
from pathlib import Path
import numpy as np
import matplotlib as plt

ROOT = Path(__file__).parent
NAMES_DIR = Path.joinpath(ROOT, "faces/lfw-deepfunneled/")


def get_names():
    names_dir = Path.joinpath(ROOT, "faces/lfw-deepfunneled/")
    names = [x.name for x in names_dir.iterdir()]
    return names


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


if __name__ == '__main__':
    img = cv2.imread("faces/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
    print(detect_face(img))
    cv2.imshow("aaa", detect_face(img)[0])
    print(detect_face(img)[0])
    cv2.waitKey(2000)
