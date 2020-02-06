import numpy as np
from face_detect_image import face_detect_image


def menu():
    choice = input('''
---- Detect and blur faces -----
| 0: Detect and blur an image. |
| 1: Detect and blur a video.  |
| 2: exit                      |
|------------------------------|\n''')
    return choice


def main():
    while True:
        choice = menu()
        # Execute chosen menu option.
        if choice == '0':
            face_detect_image()
            break
        elif choice == '1':
            pass
            break
        elif choice == '2':
            print("Exiting program")
            break
        else:
            print("Given input is not a valid choice. Try again.")


if __name__ == "__main__":
    main()
