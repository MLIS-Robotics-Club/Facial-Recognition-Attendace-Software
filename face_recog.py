import os

import cv2
import face_recognition
import numpy as np

IMG_DATA_DIR = "Data"
class_names = []
image_objects = []
image_files = os.listdir(IMG_DATA_DIR)

for img_file in image_files:
    img = cv2.imread(f"{IMG_DATA_DIR}/{img_file}")
    image_objects.append(img)
    class_names.append(os.path.splitext(img_file)[0])


def find_encodings(img_list):
    encodings = []
    for img in img_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encodings.append(encoding)

    return encodings


known_faces_encodings = find_encodings(image_objects)

# Initializing webcam
camera = cv2.VideoCapture(0)

process_this_frame = True

while True:
    success, img = camera.read()

    if process_this_frame:
        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        camera_faces_loc = face_recognition.face_locations(img_small)
        camera_encodings = face_recognition.face_encodings(img_small, camera_faces_loc)

        face_names = []
        for encoding in camera_encodings:
            matches = face_recognition.compare_faces(known_faces_encodings, encoding)
            name = "Unknown"

            face_distance = face_recognition.face_distance(
                known_faces_encodings, encoding
            )
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = class_names[best_match_index].title()

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(camera_faces_loc, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(
            img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("WebCam", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
