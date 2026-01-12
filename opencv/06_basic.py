import numpy as np
import cv2

img = cv2.imread("opencv/lena.png")
px = img[100, 200]
print(img.shape)
cp = img[0 : 0 + 20, 0 : 0 + 20]
i, j = 0, 0
addi, addj = 1, 2
addrand = 0
while True:
    cv2.imshow("image", img)
    img = cv2.imread("opencv/lena.png")
    i += addi
    j += addj
    if i > 230:
        addi = -1 + addrand
        i = 230
    if j > 230:
        addj = -2 + addrand
        j = 230
    if j < 0:
        addj = 1 + addrand
        j = 0
        addrand += 1
    if i < 0:
        addi = 2 + addrand
        addrand -= 1
        i = 0
    img[i : i + 20, j : j + 20] = cp
    print(i, j)
    keys = cv2.waitKey(1)
    if keys == 27:
        break

cv2.destroyAllWindows()
