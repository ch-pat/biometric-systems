import cv2
import os
from pathlib import Path
import numpy as np
import matplotlib as plt
from random import sample
import json


ROOT = Path(__file__).parent
NAMES_DIR = Path.joinpath(ROOT, "faces/lfw-deepfunneled/")
EXTRACTED_FACES_DIR = "extracted_faces_data/"
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
    if FACES_TEST not in os.listdir(Path.joinpath(ROOT, EXTRACTED_FACES_DIR)):
        faces = []
        labels = []
        count = 0
        limit_count = 0
        for name in os.listdir(NAMES_DIR):
            count += 1
            for image in os.listdir(Path.joinpath(NAMES_DIR, name)):
                if len(os.listdir(Path.joinpath(NAMES_DIR, name))) > 2:
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

        np.save(EXTRACTED_FACES_DIR + FACES_TEST, faces_test)
        np.save(EXTRACTED_FACES_DIR + FACES_TRAIN, faces_train)
        np.save(EXTRACTED_FACES_DIR + LABELS_TEST, labels_test)
        np.save(EXTRACTED_FACES_DIR + LABELS_TRAIN, labels_train)
    else:
        faces_test = np.load(EXTRACTED_FACES_DIR + FACES_TEST, allow_pickle=True)
        faces_train = np.load(EXTRACTED_FACES_DIR + FACES_TRAIN, allow_pickle=True)
        labels_test = np.load(EXTRACTED_FACES_DIR + LABELS_TEST, allow_pickle=True)
        labels_train = np.load(EXTRACTED_FACES_DIR + LABELS_TRAIN, allow_pickle=True)
        print("Loaded train and test data from disk.")

    return faces_train, labels_train, faces_test, labels_test


def show_image(image):
    cv2.imshow("Training on image...", image)
    cv2.waitKey(2000)


def train_models(faces, labels, force_lbph=False, force_eigen=False, force_fisher=False):
    labels = np.array(labels)

    if "models" not in os.listdir():
        os.mkdir("models")

    LBPH_face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    if "LBPH.yml" not in os.listdir("models") or force_lbph:
        LBPH_face_recognizer.train(faces, labels)
        LBPH_face_recognizer.save("models/LBPH.yml")
        print("Finished LBPH and saved to models/LBPH.yml")
    else:
        LBPH_face_recognizer.read("models/LBPH.yml")
        print("LBPH face recognizer loaded from pretrained model")

    Eigen_face_recognizer = cv2.face.EigenFaceRecognizer_create()
    if "Eigen.yml" not in os.listdir("models") or force_eigen:
        Eigen_face_recognizer.train(faces, labels)
        Eigen_face_recognizer.save("models/Eigen.yml")
        print("Finished Eigen and saved to models/Eigen.yml")
    else:
        Eigen_face_recognizer.read("models/Eigen.yml")
        print("Eigen face recognizer loaded from pretrained model")

    Fisher_face_recognizer = cv2.face.FisherFaceRecognizer_create()
    if "Fisher.yml" not in os.listdir("models") or force_fisher:
        Fisher_face_recognizer.train(faces, labels)
        Fisher_face_recognizer.save("models/Fisher.yml")
        print("Finished Fisher and saved to models/Fisher.yml")
    else:
        Fisher_face_recognizer.read("models/Eigen.yml")
        print("Fisher face recognizer loaded from pretrained model")

    return LBPH_face_recognizer, Eigen_face_recognizer, Fisher_face_recognizer


def predict(face_recognizer, test_image):
    label, confidence = face_recognizer.predict(test_image)
    return label, confidence


def reshape(faces):
    new_faces = []
    for f in faces:
        x = f.copy()
        x = cv2.resize(x, (100, 100))
        new_faces += [x]
    return new_faces


def test_accuracy(face_recognizer, faces_test, labels_test):
    correct = 0
    total = len(faces_test)
    names = get_names()
    for index, face in enumerate(faces_test):
        guess, distance = predict(face_recognizer, face)
        if guess == labels_test[index]:
            correct += 1
            print(
                f"Correct guess for {names[guess - 1]}. Accuracy: {correct / (index + 1)}, correct guesses: {correct}, total: {index + 1}. Distance: {distance}")
        else:
            print(
                f"Wrong   guess for {names[labels_test[index] - 1]}. Accuracy: {correct / (index + 1)}, correct guesses: {correct}, total: {index + 1}. Guess was: {names[guess - 1]}. Distance: {distance}")
    print(f"Accuracy: {correct / total}, correct guesses: {correct}, total: {total}")


if __name__ == '__main__':
    faces_train, labels_train, faces_test, labels_test = prepare_data_for_training(20000)

    faces_train = reshape(faces_train)
    faces_test = reshape(faces_test)
    print("Reshape finished")

    # fai train di tutti e 3 i modelli e salva su disk

    LBPH_recognizer, Eigen_recognizer, Fisher_recognizer = train_models(faces_train, labels_train)
    print("Training finished")

    #TODO: aggiungi qui cose per fare confronti tra i vari modelli
    test_accuracy(LBPH_recognizer, faces_test, labels_test)
