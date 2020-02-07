import numpy as np
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


def show_eyes(eyes, img, overlays):
    """ Show the detected eyes with a rectangle on the original image. """
    roi, resized_eye = None, None
    for i, (x, y, w, h) in enumerate(eyes):
        # Draw a rectangle on the image.
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Resize overlay
        resized_eye = resize_overlay(overlays[0], w, h)

        # Place overlay(eyes) on the image.
        img = place_overlay(img, resized_eye, x, y, w, h)

        # Place the googly eye overlay in the image.
        #img[y:y+h, x:x+w] = resized_eye

    show_image(img)


def roi_image(x, y, w, h, img):
    """ Return a ROI image based on the input rectangle (x, y, w, h) and the image. """
    return img[y:y + h, x:x + w]


def load_overlays():
    eye_1 = cv.imread("img/googly_eye_1.png", -1)  # Load image with an alpha channel.
    eye_2 = cv.imread("img/googly_eye_2.png", -1)
    return [eye_1, eye_2]


def resize_overlay(img, w, h):
    return cv.resize(img, (w, h))


def place_overlay(img, overlay, x, y, w, h):
    # Remove alpha channel from overlay.
    b, g, r, a = cv.split(overlay)
    overlay_rgb = cv.merge((b, g, r))

    img[y:y+h, x:x+w] = overlay_rgb

    # Apply some simple filtering to remove edge noise
    mask = cv.medianBlur(a, 5)

    roi = img[y:y+h, x:x+w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv.bitwise_and(roi.copy(), roi.copy(), mask=cv.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv.bitwise_and(overlay_rgb, overlay_rgb, mask=mask)

    # Update the original image with our new ROI
    img[y:y+h, x:x+w] = cv.add(img1_bg, img2_fg)

    return img


def googly_eyes():
    # Toggle googly eyes variable.
    to_googly = True

    # Open a video capture.
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)

    # Load the face classifier.
    eye_classifier = load_eye_classifier()

    # Load overlays
    overlays = load_overlays()

    while True:
        key = 0xFF & cv.waitKey(1)

        # Exit the loop if 'q' is pressed.
        if key == ord('q'):
            break

        # Toggle googly eyes if 'g' is pressed.
        if key == ord('g'):
            to_googly = not to_googly

        # Get a frame from the video capture.
        ret, frame = capture.read()

        # Continue if can't get a frame from the video capture.
        if not ret:
            continue

        # Convert RGB frame to grayscale.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Make a frame with an alpha channel.
        # frame_rgba = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
        #print("frame: ", frame.shape)
        # b_channel, _, _ = cv.split(frame)
        # frame_rgba[:, :, 3] = np.ones(b_channel.shape, dtype=b_channel.dtype)
        #print("frame_rgba: ", frame_rgba.shape)

        if to_googly:
            # Detect eyes.
            eye_detections = detect_eye(eye_classifier, gray)

            # Show detected eyes.
            show_eyes(eye_detections, frame, overlays)
        else:
            show_image(frame)

    # When everything done, release the capture
    capture.release()
    cv.destroyAllWindows()
