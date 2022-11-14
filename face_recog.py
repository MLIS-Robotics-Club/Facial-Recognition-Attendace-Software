import cv2
import numpy as np
import face_recognition
import os

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

while True:
    success, img = camera.read()
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    camera_faces_loc = face_recognition.face_locations(img_small)
    camera_encodings = face_recognition.face_encodings(img_small, camera_faces_loc)

    for (encoding, face_loc) in zip(camera_encodings, camera_faces_loc):
        matches = face_recognition.compare_faces(known_faces_encodings, encoding)
        face_distance = face_recognition.face_distance(known_faces_encodings, encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = class_names[best_match_index].title()
            print(name)

    cv2.imshow("WebCam", img)
    cv2.waitKey(1)


# imgTaylor = face_recognition.load_image_file("Data/Taylor Swift.jpeg")
# imgTaylor = cv2.cvtColor(imgTaylor, cv2.COLOR_BGR2RGB)
# taylor_face_loc = face_recognition.face_locations(imgTaylor)[0]
# taylor_encode = face_recognition.face_encodings(imgTaylor)[0]

# imgTaylorTest = face_recognition.load_image_file("Data/Taylor Swift Test 2.webp")
# imgTaylorTest = cv2.cvtColor(imgTaylorTest, cv2.COLOR_BGR2RGB)
# taylor_test_face_loc = face_recognition.face_locations(imgTaylorTest)[0]
# taylor_test_encode = face_recognition.face_encodings(imgTaylorTest)[0]


# cv2.rectangle(
#     imgTaylor,
#     (taylor_face_loc[3], taylor_face_loc[0]),
#     (taylor_face_loc[1], taylor_face_loc[2]),
#     (255, 0, 255),
#     2,
# )

# cv2.rectangle(
#     imgTaylorTest,
#     (taylor_test_face_loc[3], taylor_test_face_loc[0]),
#     (taylor_test_face_loc[1], taylor_test_face_loc[2]),
#     (255, 0, 255),
#     2,
# )



# results = face_recognition.compare_faces(
#     known_face_encodings=[taylor_encode], face_encoding_to_check=taylor_test_encode
# )
# print(results)


# # Resizing the image for better viewing
# width = 650
# height = 650

# imgTaylor = cv2.resize(imgTaylor, (width, height), interpolation=cv2.INTER_LINEAR)
# imgTaylorTest = cv2.resize(
#     imgTaylorTest, (width, height), interpolation=cv2.INTER_LINEAR
# )


# cv2.imshow("Taylor Swift", imgTaylor)
# cv2.imshow("Taylor Swift Test", imgTaylorTest)
# cv2.waitKey(0)
