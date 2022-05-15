import cv2
import os
from pathlib import Path
import numpy as np
import matplotlib as plt
from random import sample
import json


ROOT = Path(__file__).parent
NAMES_DIR = Path.joinpath(ROOT, "faces/lfw-deepfunneled/")
FACES_TEST = "faces_test.npy"
FACES_TRAIN = "faces_train.npy"
LABELS_TEST = "labels_test.npy"
LABELS_TRAIN = "labels_train.npy"


def get_names():
    names_dir = Path.joinpath(ROOT, "faces/lfw-deepfunneled/")
    names = [x.name for x in names_dir.iterdir()]
    return names


def detect_face(image):
    """Returns a tuple containing (image cropped to face, the matrix representation of it)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


def prepare_data_for_training(limit):
    """Returns a tuple containing (faces_train, labels_train, faces_test, labels_test)
    only performs face detection from images if a previous result is not already saved to disk"""
    if FACES_TEST not in os.listdir(ROOT):
        faces = []
        labels = []
        count = 0
        limit_count = 0
        for name in os.listdir(NAMES_DIR):
            count += 1
            for image in os.listdir(Path.joinpath(NAMES_DIR, name)):
                limit_count += 1
                if limit_count >= limit:
                    break
                filename = Path.joinpath(NAMES_DIR, name, image)
                im = cv2.imread(str(filename))
                face, rect = detect_face(im)
                if face is not None:
                    faces += [face]
                    labels += [count]
                    print(f"{limit_count}. Elaborato {image} per tizio {name}")
            if limit_count >= limit:
                break

        faces_train = []
        faces_test = []
        labels_train = []
        labels_test = []

        indices = [x for x in range(len(faces))]
        test_indices = sample(indices, len(faces) // 10)
        for i in indices:
            if i in test_indices:
                faces_test += [faces[i]]
                labels_test += [labels[i]]
            else:
                faces_train += [faces[i]]
                labels_train += [labels[i]]

        np.save(FACES_TEST, faces_test)
        np.save(FACES_TRAIN, faces_train)
        np.save(LABELS_TEST, labels_test)
        np.save(LABELS_TRAIN, labels_train)
    else:
        faces_test = np.load(FACES_TEST, allow_pickle=True)
        faces_train = np.load(FACES_TRAIN, allow_pickle=True)
        labels_test = np.load(LABELS_TEST, allow_pickle=True)
        labels_train = np.load(LABELS_TRAIN, allow_pickle=True)

    return faces_train, labels_train, faces_test, labels_test


def show_image(image):
    cv2.imshow("Training on image...", image)
    cv2.waitKey(2000)


def train_face_recognizer(faces, labels):
    # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    # face_recognizer = cv2.face.FisherFaceRecognizer_create()
    labels = np.array(labels)
    face_recognizer.train(faces, labels)
    return face_recognizer


def predict(face_recognizer, test_image):
    label, confidence = face_recognizer.predict(test_image)
    return label, confidence


def lazy_reshape(faces):
    new_faces = []
    for f in faces:
        x = f.copy()
        x.resize((168, 168), refcheck=False)
        new_faces += [x]
    return new_faces


def reshape(faces):
    new_faces = []
    for f in faces:
        x = f.copy()
        x = cv2.resize(x, (100, 100))
        new_faces += [x]
    return new_faces


if __name__ == '__main__':
    faces_train, labels_train, faces_test, labels_test = prepare_data_for_training(5000)

    faces_train = reshape(faces_train)
    faces_test = reshape(faces_test)
    print("Reshape finished")

    # fai train di tutti e 3 i modelli e salva su disk
    face_recognizer = train_face_recognizer(faces_train, labels_train)
    print("Training finished")

    correct = 0
    total = len(faces_test)
    for index, face in enumerate(faces_test):
        guess, confidence = predict(face_recognizer, face)
        if guess == labels_test[index]:
            correct += 1
            print(f"Correct guess for {get_names()[guess - 1]}")
    print(f"accuracy: {correct / total}, correct guesses: {correct}, total: {total}")
