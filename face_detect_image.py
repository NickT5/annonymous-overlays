import cv2 as cv


def blur_face_avg(img):
    """ Return a (average) blurred version of the input image. """
    return cv.blur(img, (29, 29))


def blur_face_median(img):
    """ Return a (median) blurred version of the input image. """
    return cv.medianBlur(img, 25)


def blur_face_gaussian(img):
    """ Return a (gaussian) blurred version of the input image. """
    return cv.GaussianBlur(img, (25, 25), cv.BORDER_DEFAULT)


def blur_face_bilateral(img):
    """ Return a (bilateral) blurred version of the input image.
     Bilateral filtering preserves the edges (= does not filter them) but is slower compared to the rest. """
    return cv.bilateralFilter(img, 9, 150, 150)


def open_face():
    """ Open and return an image. """
    img_name = "img/elon-musk.jpg"
    img = cv.imread(img_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray


def show_face(img, window_name="Face"):
    """ Show the image. """
    if img is not None:
        cv.imshow(window_name, img)
        cv.waitKey(0)
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
    roi, blur_a, blur_m, blur_g, blur_b = None, None, None, None, None
    for (x, y, w, h) in faces:
        # Draw a rectangle on the image.
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Create a ROI.
        roi = roi_image(x, y, w, h, img)
        # Blur the ROI.
        blur_a = blur_face_avg(roi)
        # blur_m = blur_face_median(roi)
        # blur_g = blur_face_gaussian(roi)
        # blur_b = blur_face_bilateral(roi)
        # Place the blurred ROI in the image.
        img[y:y+h, x:x+w] = blur_a

    show_face(img)
    # show_face(roi)
    # show_face(blur_a, "avg")
    # show_face(blur_m, "median")
    # show_face(blur_g, "gaussian")
    # show_face(blur_b, "bilateral")

    cv.destroyAllWindows()


def roi_image(x, y, w, h, img):
    """ Return a ROI image based on the input rectangle (x, y, w, h) and the image. """
    return img[y:y+h, x:x+w]


def face_detect_image():
    img, gray = open_face()
    # show_face(img)

    face_classifier = load_face_classifier()
    face_detections = detect_face(face_classifier, gray)
    show_faces(face_detections, img)
