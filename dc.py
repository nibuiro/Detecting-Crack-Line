


def to_mono(image_name, t=127, mode=1):
    import cv2
    import numpy as np
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, th2 = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    if 0 == mode:
        cv2.imwrite(f"mono_{t}_{image_name}", th2)
    elif 1 == mode:
        cv2.imshow("to_mono", th2)
    else:
        pass


if __name__ == "__main__":
    main()