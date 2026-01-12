import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import cv2
import numpy as np
from matplotlib import pyplot as plt

print(cv2.IMREAD_COLOR)  # 컬러 이미지 (height,width,rgb)
print(cv2.IMREAD_GRAYSCALE)  # 흑백 이미지 (height,width)
print(cv2.IMREAD_UNCHANGED)  # 알파 채널 필요할 때 (height,width,rgba)

imgcolor = cv2.imread("lena.png", cv2.IMREAD_COLOR)
print(imgcolor.shape)
print(imgcolor[0, 1])

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
print(img.shape)
print(img[0, 1])
cv2.imshow("image", img)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
elif key == ord("s"):
    cv2.imwrite("save.png", img)
    cv2.destroyAllWindows()

b, g, r = cv2.split(imgcolor)
a = np.full(b.shape, 128, dtype=b.dtype)
img_rgba = cv2.merge([r, g, b, a])
cv2.imshow("colored", img_rgba)
cv2.imwrite("test.png", img_rgba)
key = cv2.waitKey(0)

plt.imshow(img_rgba)
plt.xticks([])
plt.yticks([])
plt.show()
