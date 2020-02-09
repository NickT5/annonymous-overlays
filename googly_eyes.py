import numpy as np
import cv2 as cv


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
    """ Show the detected eyes with an overlay on the original image. """
    roi, resized_eye = None, None
    for i, (x, y, w, h) in enumerate(eyes):
        # # Draw a rectangle on the image.
        # img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #
        # # Draw a circle around the detected eye.
        # cx = x + (w//2)
        # cy = y + (h//2)
        # r = int(((w ** 2 + h ** 2) ** 0.5)/2)    # c² = a² + b²
        # img = cv.circle(img, (cx, cy), r, (0, 255, 0), 2)

        # Resize overlay
        if i >= len(overlays): i = 0
        resized_eye = resize_overlay(overlays[i], w, h)

        # Place overlay into the image frame.
        img = place_overlay(x, y, w, h, img, resized_eye)

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


def place_overlay(x, y, w, h, img, resized_eye):
    """# Place the googly eye overlay in the image.
    # Place a transparent overlay (googly eye) on the image.
    # Difficulty is placing rgba image with a transparent channel on an image with 3 channels (rgb).
    # 1. Create an inverted mask of the overlay using its alpha channel.
    # 1a. Take alpha channel from overlay.
    # 1b. Subtract 255 to inverted the mask.
    # 1c. Normalize alpha channel to values between 0 and 1 by dividing by 255.
    # Why? So we can later apply the inverted mask using multiplication operator.
    # 2. Apply the inverted mask on the region of interest, i.e. the detected eye in rgb.
    # This results in an image where the values around the eyes are from the rgb and the eye is completely black.
    # 3. Create a mask of the overlay using its alpha channel.
    # 3a. Take alpha channel, 3b. Normalize values between 0 and 1.
    # 4. Apply the mask on rgba overlay image.
    # This results in an image where the values around the eye are black and the eye is from the rgb eye image.
    # 5. Add up the two result images (background and foreground).
    # 6. Insert the roi into the image."""

    roi = img[y:y + h, x:x + w]         # The region to be replaced by an overlay.
    bg = roi.copy()                     # Background.
    fg = resized_eye[:, :, 0:3].copy()  # Foreground.

    # Create masks.
    mask_inverted = np.atleast_3d(255 - resized_eye[:, :, 3]) / 255.0
    mask = np.atleast_3d(resized_eye[:, :, 3]) / 255.0

    # Apply masks.
    np.multiply(roi, mask_inverted, out=bg, casting="unsafe")  # Step 1 and 2.
    np.multiply(resized_eye[:, :, 0:3], mask, out=fg, casting="unsafe")

    # Add the foreground & background together and insert it into the frame.
    img[y:y + h, x:x + w] = np.add(bg, fg)
    return img


def googly_eyes():
    """ Main function """
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

        if to_googly:
            # Detect eyes.
            eye_detections = detect_eye(eye_classifier, gray)

            # Show detected eyes.
            show_eyes(eye_detections, frame, overlays)
        else:
            show_image(frame)

    # When everything done, release the capture.
    capture.release()
    cv.destroyAllWindows()
