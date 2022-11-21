# Facial-Recognition-Attendance-Software

A simple and intuitive attendance program that uses facial recognition to log attendances. The AI can be trained using custom images. The program uses the Support Vector Machine (SVM) algorithm to train an AI model based on 128 different facial encoding points.

The programme will use a folder in the current directory called "Data". This directory should be structured in this way:

```html
Structure:
        <Data>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
```

Make sure that each persons' directories are named as the person's actual name. The individual images inside each persons' directories can have any name.

## Frameworks used

As I do not plan to re-invent the wheel, the program uses some standard but powerful machine learning libraries for accurate facial recognition

* [face-recognition](https://github.com/ageitgey/face_recognition)
* [numpy](https://github.com/numpy/numpy)
* [opencv-python](https://github.com/opencv/opencv-python)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [dlib](https://github.com/davisking/dlib)

## TODO

* A custom attendance-keeping software made with Python is currently in the works.

## Known Issues

* Although using SVM has made the AI much more accurate, one drawback of using general SVM is that detecting completely unknown faces is really buggy/inaccurate. I am currently working on using using One Class SVM which allows for novelty/outlier detection to address this issue. Other machine learning algorithms like K-Nearest Neighbor do allow for unknown faces to be detected but the accuracy is much lower. For the time being, it is best advised that the AI is only used to detect known faces.