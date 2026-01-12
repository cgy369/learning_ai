import cv2
import numpy as np

img = cv2.imread("opencv/logo.png")
he, wi = img.shape[:2]

shrink = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
zoom1 = cv2.resize(img, (wi * 2, he * 2), interpolation=cv2.INTER_CUBIC)
zoom2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


item_list = [img, shrink, zoom1, zoom2]
i = 0
while True:
    if i > 3:
        i = 0
    cv2.imshow("origin", item_list[i])
    i += 1
    keys = cv2.waitKey(100)
    if keys == 27:
        break
cv2.destroyAllWindows()
while True:
    M = np.float32([[1, 0, i], [0, 1, 20]])
    dst = cv2.warpAffine(img, M, (he, wi))
    cv2.imshow("Trans", dst)
    keys = cv2.waitKey(100)
    i += 1
    if i > 200:
        i = 0
    if keys == 27:
        break
cv2.destroyAllWindows()
