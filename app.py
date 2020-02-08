from detect_blur_image import detect_blur_image
from detect_blur_video import detect_blur_video
from googly_eyes import googly_eyes
from googly_eyes_image import googly_eyes_image


def menu():
    choice = input('''
---- Detect and blur faces -----
| 0: Detect and blur an image. |
| 1: Detect and blur a video.  |
| 2: Googly eyes w/ video.     |
| 3: Googly eyes w/ photo.     |
| 4: exit                      |
|------------------------------|\n''')
    return choice


def main():
    while True:
        choice = menu()
        # choice = '3'
        # Execute chosen menu option.
        if choice == '0':
            detect_blur_image()
            break
        elif choice == '1':
            detect_blur_video()
            break
        elif choice == '2':
            googly_eyes()
            break
        elif choice == '3':
            googly_eyes_image()
            break
        elif choice == '4':
            print("Exiting program")
            break
        else:
            print("Given input is not a valid choice. Try again.")


if __name__ == "__main__":
    main()
