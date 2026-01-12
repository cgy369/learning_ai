import cv2
import numpy as np

available = []
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 백엔드로 카메라 열기
    if cap.isOpened():
        available.append(i)
        print(f"Camera {i} is available")
        cap.release()
print(f"사용 가능 카메라 : {available}")

cap = cv2.VideoCapture(available[0])
img2 = cv2.imread("opencv/lena.png")
h, w, c = img2.shape
print(img2.shape)
cap.set(3, w)  # 너비
cap.set(4, h)  # 높이

while True:
    ret, frame = cap.read()
    if ret:
        # cv2.imread(frame)
        cv2.imshow("origin_test", frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # lower_blue = np.array([0, 0, 50])
        # upper_blue = np.array([80, 80, 255])
        # mask = cv2.inRange(frame, lower_blue, upper_blue)
        res = cv2.bitwise_and(
            frame, frame, mask=cv2.bitwise_not(mask)
        )  # 이렇게 해당 색만 안보이게도 할 수 있다 - 마스크를 반전해서
        cv2.imshow("test", res)
        keys = cv2.waitKey(1)
        if keys == 27:
            break
cv2.destroyAllWindows()
