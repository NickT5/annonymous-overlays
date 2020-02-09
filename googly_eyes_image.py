import numpy as np
import cv2 as cv


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


def load_eye_classifier():
    """ Load and return the haar cascade eye classifier. """
    eye_cascade = cv.CascadeClassifier('classifier/haarcascade_eye.xml')
    return eye_cascade


def detect_eye(eye_cascade, gray):
    """ Find eyes in the image. If eyes are found, the detectMultiScale function returns the positions of
     detected eyes as Rect(x, y, w, h). """
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    return eyes


def show_eyes(eyes, img, overlays):
    """ Put the googly eye overlay on the detected eye and show it. """
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
    np.multiply(roi, mask_inverted, out=bg, casting="unsafe")
    np.multiply(resized_eye[:, :, 0:3], mask, out=fg, casting="unsafe")

    # Add the foreground & background together and insert it into the frame.
    img[y:y + h, x:x + w] = np.add(bg, fg)
    return img


def googly_eyes_image():
    """ Main function """
    img, gray = open_image("img/elon-musk.jpg")

    overlays = load_overlays()
    eye_classifier = load_eye_classifier()
    eye_detections = detect_eye(eye_classifier, gray)
    show_eyes(eye_detections, img, overlays)
