import cv2

img = cv2.imread("opencv/lena.png")

w, h = img.shape[:2]
print(img.shape)
GAD = cv2.pyrDown(img)  # 블러 처리 후 사이즈 줄인다. 고주파 삭제 ( 외곽을 뭉갠다. )

GAU = cv2.pyrUp(GAD)  # 그냥 resize하면 과정에서 고주파(외곽)가 생길 수 있다.
cv2.imshow("test", GAU)
cv2.waitKey(0)
temp = cv2.resize(GAD, (250, 250))
cv2.imshow("test", temp)
cv2.waitKey(0)
res = cv2.subtract(img, temp)
cv2.imshow("test", res)
cv2.waitKey(0)
