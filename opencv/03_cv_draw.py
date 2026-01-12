import numpy as np
import cv2


while True:
    img = np.zeros((512, 512, 3), np.uint8)
    img = cv2.line(img, (0, 0), (0, 512), (255, 0, 0), 5)  # 이미지,start,end,색상,두께
    img = cv2.line(img, (512, 0), (512, 512), (255, 0, 0), 5)
    # rectangle,circle,polygon

    pts = np.array([[1, 1], [200, 200], [100, 200]], np.int32)
    img = cv2.polylines(
        img,
        [pts],
        False,
        (0, 255, 255),
        10,
    )
    cv2.imshow("image", img)
    keys = cv2.waitKey(100)
    if keys == 27:
        break
    img = np.zeros((512, 512, 3), np.uint8)
    pts = pts.reshape((-1, 1, 2))
    # pts = np.array([[512, 0], [100, 100], [100, 200]], np.int32)
    img = cv2.polylines(
        img,
        [pts],
        False,
        (0, 255, 255),
        10,
    )
    cv2.imshow("image", img)
    keys = cv2.waitKey(100)

    if keys == 27:
        break

cv2.destroyAllWindows()
