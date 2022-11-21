"""
Python module that uses Sklearn's Support Vector Machine (SVM) implementation to recognize faces
from a directory of images that act as training data
"""

import os
import pickle

import cv2
import face_recognition
import numpy as np
from sklearn import svm

IMG_DATA_DIR = "Data"


def find_encodings(img_file_location: str) -> np.ndarray:
    """
    Function that will find face encodings from images given the img path
    """
    img = face_recognition.load_image_file(img_file_location)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_facial_locations = face_recognition.face_locations(img)
    img_encoding = face_recognition.face_encodings(
        img, known_face_locations=img_facial_locations
    )[0]

    return img_encoding

def find_names_encodings(img_data: list) -> tuple[list, list]:
    """
    Function that when given a list of directories in the current direction, will
    extract the names of the persons (given that the names of the directories are
    the same as those of the persons) and also find faces in any available image files
    and extract the encodings and locations of the faces

    returns a tuple in the form: tuple[encodings, names]
    """

    persons = img_data

    # We'll check if previous face encodings exist
    encodings_exist = False
    if os.path.isfile("face_encodings_data.dat"):
        encodings_exist = True

    all_face_encodings = {}
    encodings = []
    names = []

    for person in persons:
        person_imgs = os.listdir(f"{IMG_DATA_DIR}/{person}")

        for person_img in person_imgs:
            name = os.path.splitext(person)[0]

            if not encodings_exist:
                # If previous face encodings do not exist, we'll proceed to create new encodings
                all_face_encodings[name] = find_encodings(f"{IMG_DATA_DIR}/{person}/{person_img}")

                # We'll then use the pickle library to dump the face encodings to a data file
                with open("face_encodings_data.dat", "wb") as data_file:
                    pickle.dump(all_face_encodings, data_file)

            else:
                # If previous face encodings do exist, we simply use the pickle library to load them into memory
                with open("face_encodings_data.dat", "rb") as data_file:
                    all_face_encodings = pickle.load(data_file)

    encodings = np.array(list(all_face_encodings.values()))
    names = np.array(list(all_face_encodings.keys()))

    return (encodings, names)


img_data_obj = os.listdir(IMG_DATA_DIR)
encodings, names = find_names_encodings(img_data_obj)

# Using SVM to fit the training data to train our model
clf = svm.SVC(gamma="scale", probability=True)
clf.fit(encodings, names)


# Initializing webcam
camera = cv2.VideoCapture(0)

process_this_frame = True

while True:
    success, img = camera.read()

    if process_this_frame:
        # We will process the image after reducing it's size by 50% to improve speed
        img_small = cv2.resize(img, (0, 0), None, 0.50, 0.50)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        camera_faces_loc = face_recognition.face_locations(img_small, model="hog")
        camera_encodings = face_recognition.face_encodings(img_small, camera_faces_loc)

        face_names = []
        for encoding in camera_encodings:
            name = clf.predict([encoding])
            print(name)
            face_names.extend(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(camera_faces_loc, face_names):
        # Since we reduced the image's size by 50%, we are doubling the sizes when viewing
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Drawing a rectangle around the detected faces in the current camera frame
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(
            img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        # Displaying the person(s) name(s) under the person(s) face(s)
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("WebCam", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        # We'll close the application when the user presses "q"
        break

camera.release()
cv2.destroyAllWindows()
