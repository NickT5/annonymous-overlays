import numpy as np
import cv2 as cv


def blur_face_avg(img):
    """ Return a (average) blurred version of the input image. """
    return cv.blur(img, (91, 91))


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


def open_image(img_name):
    """ Open and return an image. """
    img = cv.imread(img_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray


def show_image(img, window_name="Frame"):
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
    """ Show the detected faces with circle and blurred on the original image. """
    roi, blur_ = None, None
    for (x, y, w, h) in faces:
        # Calculate center point and radius for circular masks.
        cx = w // 2
        cy = h // 2
        r = h // 2  # int(((w ** 2 + h ** 2) ** 0.5)/2)  # c² = a² + b², for a bigger circle, the roi itself needs to
        # be bigger which results in inconsistencies.

        # Draw a rectangle/circle on the image.
        # img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img = cv.circle(img, (x+cx, y+cy), r, (101, 201, 255), 2)

        # Create a ROI.
        roi = roi_image(x, y, w, h, img)

        # Blur the ROI.
        blur = blur_face_avg(roi)

        # Create masks (normal and inverted).
        mask = np.full((roi.shape[0], roi.shape[1], 1), 0, dtype=np.uint8)  # Init black mask.
        cv.circle(mask, (cx, cy), r, (255, 255, 255), -1)  # Fill in a white circle in the middle to create the mask.
        mask_inverted = 255 - mask

        # Apply the masks.
        fg = cv.bitwise_and(blur, blur, mask=mask)         # Foreground: blurred circle face.
        bg = cv.bitwise_and(roi, roi, mask=mask_inverted)  # Background: part outside the detected face.

        # Add the foreground & background together and insert it into the frame.
        img[y:y+h, x:x+w] = cv.add(fg, bg)

    show_image(img)
    cv.destroyAllWindows()


def roi_image(x, y, w, h, img):
    """ Return a ROI image based on the input rectangle (x, y, w, h) and the image. """
    return img[y:y+h, x:x+w]


def detect_blur_image():
    img, gray = open_image("img/elon-musk.jpg")

    face_classifier = load_face_classifier()
    face_detections = detect_face(face_classifier, gray)
    show_faces(face_detections, img)
