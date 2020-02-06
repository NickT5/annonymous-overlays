import numpy as np
import cv2 as cv


def blur_face():
    pass


def open_face():
    """ Open and return an image. """
    img_name = "img/elon-musk.jpg"
    img = cv.imread(img_name)                    # imread returns None if it fails.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray


def show_face(img):
    """ Show the image. """
    if img is not None:
        window_name = "Face"
        cv.imshow(window_name, img)
        cv.waitKey(0)
        cv.destroyWindow(window_name)
    else:
        print("Unable to show image.")


def load_face_classifier():
    """ Load and return the haar cascade face classifier. """
    face_cascade = cv.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')
    return face_cascade


def detect_face(face_cascade, gray):
    """ Find faces in the image. If faces are found, the detectMultiScale function returns the positions of
     detection faces as Rect(x, y, w, h). """
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


def show_faces(faces, img):
    """ Show the detected faces with a rectangle on the original image. """
    for (x, y, w, h) in faces:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    show_face(img)


def main():
    img, gray = open_face()
    show_face(img)
    show_face(gray)

    face_classifier = load_face_classifier()
    face_detections = detect_face(face_classifier, gray)
    show_faces(face_detections, img)


if __name__ == "__main__":
    main()
