import cv2 as cv


def blur_eye_avg(img):
    """ Return a (average) blurred version of the input image. """
    return cv.blur(img, (29, 29))


def show_image(img, window_name="Frame"):
    """ Show the image. """
    if img is not None:
        cv.imshow(window_name, img)


def load_eye_classifier():
    """ Load and return the haar cascade eye classifier. """
    eye_cascade = cv.CascadeClassifier('classifier/haarcascade_eye.xml')
    return eye_cascade


def detect_eye(eye_cascade, gray):
    """ Find eyes in the image. If eyes are found, the detectMultiScale function returns the positions of
     detection eyes as Rect(x, y, w, h). """
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    return eyes


def show_eyes(eyes, img):
    """ Show the detected eyes with a rectangle on the original image. """
    roi, blur = None, None
    for (x, y, w, h) in eyes:
        # Draw a rectangle on the image.
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Create a ROI.
        roi = roi_image(x, y, w, h, img)
        # Blur the ROI.
        blur = blur_eye_avg(roi)
        # Place the blurred ROI in the image.
        img[y:y+h, x:x+w] = blur

    show_image(img)


def roi_image(x, y, w, h, img):
    """ Return a ROI image based on the input rectangle (x, y, w, h) and the image. """
    return img[y:y + h, x:x + w]


def googly_eyes():
    # Toggle googly eyes variable.
    to_googly = True

    # Open a video capture.
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)

    # Load the face classifier.
    eye_classifier = load_eye_classifier()

    while True:
        key = 0xFF & cv.waitKey(1)

        # Exit the loop if 'q' is pressed.
        if key == ord('q'):
            break

        # Toggle blurring is 'b' is pressed.
        if key == ord('g'):
            to_googly = not to_googly

        # Get a frame from the video capture.
        ret, frame = capture.read()

        # Continue if can't get a frame from the video capture.
        if not ret:
            continue

        # Convert RGB frame to grayscale.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        show_image(gray)
        if to_googly:
            # Detect eyes.
            eye_detections = detect_eye(eye_classifier, gray)

            # Show detected eyes.
            show_eyes(eye_detections, frame)
        else:
            show_image(frame)

    # When everything done, release the capture
    capture.release()
    cv.destroyAllWindows()