from detect_blur_image import detect_blur_image
from detect_blur_video import detect_blur_video
from googly_eyes import googly_eyes

def menu():
    choice = input('''
---- Detect and blur faces -----
| 0: Detect and blur an image. |
| 1: Detect and blur a video.  |
| 2: Googly eyes.              |
| 3: exit                      |
|------------------------------|\n''')
    return choice


def main():
    while True:
        choice = menu()
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
            print("Exiting program")
            break
        else:
            print("Given input is not a valid choice. Try again.")


if __name__ == "__main__":
    main()
