import cv2
import numpy as np

# --- 1. 이미지 준비 ---
# cv2.imread() 함수로 합성할 이미지들을 불러옵니다.
# img1: 로고 이미지 ('test.png')
# img2: 배경이 될 이미지 ('lena.png')
img1 = cv2.imread("opencv/logo.png")
img2 = cv2.imread("opencv/test.jpg")

# --- 2. 로고를 삽입할 영역(ROI) 설정 ---
# 로고(img1)의 크기를 먼저 알아냅니다. shape 속성은 (세로, 가로, 채널) 정보를 가집니다.
rows, cols, channels = img1.shape

# 배경(img2)에서 로고를 넣을 위치를 지정합니다.
# 여기서는 배경의 좌측 상단(0,0)을 시작점으로, 로고의 크기만큼 영역을 잘라냅니다.
# 이 영역을 ROI (Region of Interest, 관심 영역)라고 부릅니다.
roi = img2[0:rows, 0:cols]
cv2.imshow("res", roi)  # 원본 이미지의 일부 공간을 logo 크기만큼 지정
cv2.waitKey(0)
# --- 3. 마스크(Mask) 및 반전 마스크 생성 ---
# '마스크'는 이미지에서 어느 부분을 보여주고 어느 부분을 가릴지 정하는 데 사용하는 흑백 이미지입니다.
# 흰색(255) 부분은 '보여줄 영역', 검은색(0) 부분은 '가릴 영역'을 의미합니다.

# 3-1. 로고 이미지를 흑백(Grayscale)으로 변환합니다.
# 색상 정보를 없애고 밝기 값만으로 처리하기 위함입니다.
img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 로고를 흑백 이미지로 변경

# 3-2. 흑백 이미지를 이진화(Binary)하여 마스크를 생성합니다.
# cv2.threshold는 임계값(여기서는 10)을 기준으로 픽셀을 흑/백으로 나눕니다.
# 픽셀 값이 10보다 크면 255(흰색), 작거나 같으면 0(검은색)으로 바꿉니다.
# 결과적으로 'mask'는 로고 모양만 흰색이고, 배경은 검은색인 이미지가 됩니다.

ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
# 흑백 이미지를 마스킹을 위해 0,255 값으로 변경함, 회색 영역 제거 이 경우

cv2.imshow("res", mask)
cv2.waitKey(0)
# 3-3. 마스크를 반전시킵니다. (흰색 <-> 검은색)
# 나중에 배경에서 로고 모양의 구멍을 뚫는 데 사용됩니다.
# 'mask_inv'는 로고 모양이 검은색이고, 배경은 흰색이 됩니다.
mask_inv = cv2.bitwise_not(mask)
cv2.imshow("res", mask_inv)
cv2.waitKey(0)
# --- 4. 이미지와 마스크를 이용한 비트와이즈(Bitwise) 연산 ---
# cv2.bitwise_and 연산은 마스크의 '흰색' 영역에 해당하는 원본 픽셀만 통과시킵니다.
# (검은색 영역은 연산 결과가 무조건 검은색이 됨)

# 4-1. 로고 이미지(img1)에서 배경을 제거하고 로고 부분만 추출합니다. (Foreground)
# 'mask'를 사용했으므로, 로고 모양(흰색)만 살아남고 나머지는 검게 됩니다.
img1_fg = cv2.bitwise_and(img1, img1, mask=mask)
cv2.imshow("res", img1_fg)
cv2.waitKey(0)
# 4-2. ROI(배경의 일부)에서 로고가 들어갈 자리를 검게 만듭니다. (Background)
# 반전된 'mask_inv'를 사용했으므로, 로고 모양(검은색) 부분이 구멍처럼 뚫립니다.
img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# 하지만 copyTo를 써서 해당 부분을 강제로 덮어씌우면? 해당 동작이 필요가 없다. 만약 이게 없이 add를 하면 픽셀의 bgr이 섞인다.
# --- 5. 두 이미지 합성 ---
# cv2.add 함수로 두 이미지를 합칩니다.
# img1_fg (로고만 있음) + img2_bg (로고 자리가 비어있는 배경)
# 결과적으로 배경 위에 로고가 자연스럽게 올라간 것처럼 보입니다.
dst = cv2.add(img1_fg, img2_bg)

# --- 6. 원본 이미지에 합성 결과 적용 ---
# 처음에 잘라냈던 배경 이미지의 ROI 영역에, 합성된 이미지(dst)를 다시 붙여넣습니다.
img2[0:rows, 0:cols] = dst

# --- 7. 결과 출력 ---
# 'res'라는 이름의 창에 최종 결과물을 표시합니다.
cv2.imshow("res", img2)
# 아무 키나 누를 때까지 대기합니다.
cv2.waitKey(0)
# 모든 창을 닫습니다.
cv2.destroyAllWindows()
cv2.destroyAllWindows()


logo = cv2.imread("opencv/logo.png", cv2.IMREAD_UNCHANGED)
# 이렇게 해야 알파채널이 살아난다.
rows, cols, channels = logo.shape
img2 = cv2.imread("opencv/test.jpg")
roi = img2[:rows, :cols]  # 메모리 카피, 해당 부분(roi)이 변경되면 img2의 값도 변경된다.
b, g, r, a = cv2.split(logo)
mask = a
logo_bgr = logo[:, :, :3]
cv2.copyTo(
    logo_bgr, mask, roi
)  # mask를 지원한다. roi값(메모리)을 변경하니까 img2의 해당 부분이 변경되서 적용된다.

# --- 6. 결과 출력 ---
cv2.imshow("result", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
