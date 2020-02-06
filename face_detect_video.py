import cv2 as cv
from time import time


def blur_face_avg(img):
    """ Return a (average) blurred version of the input image. """
    return cv.blur(img, (29, 29))


def show_face(img, window_name="Face"):
    """ Show the image. """
    if img is not None:
        cv.imshow(window_name, img)


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
    roi, blur = None, None
    for (x, y, w, h) in faces:
        # Draw a rectangle on the image.
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Create a ROI.
        roi = roi_image(x, y, w, h, img)
        # Blur the ROI.
        blur = blur_face_avg(roi)
        # Place the blurred ROI in the image.
        img[y:y+h, x:x+w] = blur

    show_face(img)


def roi_image(x, y, w, h, img):
    """ Return a ROI image based on the input rectangle (x, y, w, h) and the image. """
    return img[y:y+h, x:x+w]


def face_detect_video():
    # State variable: to blur or not to blur.
    to_blur = True

    # Open a video capture.
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)

    # Load the face classifier.
    face_classifier = load_face_classifier()

    # Get start time.
    time_start = time()

    # Init number of frames variable.
    n_frames = 0
    while True:
        key = 0xFF & cv.waitKey(1)

        # Exit the loop if 'q' is pressed.
        if key == ord('q'):
            break

        # Toggle blurring is 'b' is pressed.
        if key == ord('b'):
            to_blur = not to_blur

        # Get a frame from the video capture.
        ret, frame = capture.read()

        # Continue if can't get a frame from the video capture.
        if not ret:
            continue

        # Convert RGB frame to grayscale.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if to_blur:
            # Detect faces.
            face_detections = detect_face(face_classifier, gray)

            # Show detected faces.
            show_faces(face_detections, frame)
        else:
            show_face(frame)

        n_frames += 1

    # Get end time.
    time_end = time()

    # Elapsed time in seconds.
    time_elapsed = time_end - time_start
    # Calculate and print frames per second.
    fps = n_frames / time_elapsed
    print(f"Fps = {fps:.2f}; Time elapsed = {time_elapsed:.2f} seconds.")

    # When everything done, release the capture
    capture.release()
    cv.destroyAllWindows()
