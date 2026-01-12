import cv2
import numpy as np

img = cv2.imread("opencv/logo.png", cv2.IMREAD_GRAYSCALE)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(
    img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=25, maxRadius=0
)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 원그리기
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)  # 점찍기
cv2.imshow("img", cimg)
cv2.waitKey(0)
