import cv2
import numpy as np
from matplotlib import pyplot as plt

# --- 1. 이미지 불러오기 ---
# cv2.imread(filename, 0): 이미지를 흑백(Grayscale)으로 불러옵니다.
# 템플릿 매칭은 색상 정보 없이, 픽셀의 밝기 값을 기반으로 비교하는 것이 일반적이므로 흑백 이미지를 사용합니다.

# img: 템플릿을 찾아낼 대상이 되는 원본 이미지.
img = cv2.imread("opencv/lena.png", 0)
# img2: 원본 이미지를 복사하여, 여기에 결과를 그릴 예정입니다. ( 원본을 훼손하지 않기 위함 )
img2 = img.copy()
# template: 원본 이미지에서 찾고 싶은 작은 이미지 조각.
template = cv2.imread("opencv/cap_lena.jpg", 0)

# --- 2. 템플릿의 가로, 세로 크기 저장 ---
# template.shape는 (세로, 가로) 순서로 값을 반환합니다. (흑백 이미지이므로 채널 정보는 없음)
# 사각형을 그릴 때는 (가로, 세로) 순서가 더 편리하므로, `[::-1]`을 사용하여 순서를 뒤집습니다.
# w: 템플릿의 가로(width), h: 템플릿의 세로(height)
w, h = template.shape[::-1]

# --- 3. 템플릿 매칭 방법(Method) 정의 ---
# OpenCV는 여러 가지 템플릿 매칭 계산 방식을 제공합니다.
methods = [
    "cv2.TM_CCOEFF",
    "cv2.TM_CCOEFF_NORMED",  # 상관 계수 매칭 (정규화) - 일반적으로 가장 성능이 좋음
    "cv2.TM_CCORR",
    "cv2.TM_CCORR_NORMED",  # 상관관계 매칭 (정규화)
    "cv2.TM_SQDIFF",  # 제곱 차이 매칭
    "cv2.TM_SQDIFF_NORMED",  # 제곱 차이 매칭 (정규화)
]

# methods 리스트에서 사용할 방법을 선택합니다.
# eval(string): 문자열을 실제 파이썬 코드로 실행합니다. 'cv2.TM_CCOEFF'라는 문자열을 실제 cv2.TM_CCOEFF 상수로 변환해줍니다.
method = eval(methods[0])  # 여기서는 첫 번째 'cv2.TM_CCOEFF'를 사용

# --- 4. 템플릿 매칭 실행 ---
# cv2.matchTemplate(image, template, method): 템플릿 매칭을 수행하는 핵심 함수입니다.
# - image: 원본 이미지
# - template: 찾을 템플릿 이미지
# - method: 사용할 매칭 방법
# 이 함수는 'res'라는 결과 행렬(결과 맵)을 반환합니다.
# res의 각 픽셀 (x,y)는 원본 이미지의 해당 위치에 템플릿의 좌측 상단을 놓았을 때의 '매칭 점수'를 의미합니다.
res = cv2.matchTemplate(img, template, method)

# --- 5. 매칭 결과에서 최적 위치 찾기 ---
# cv2.minMaxLoc(res): 결과 맵(res)에서 최소값, 최대값, 그리고 각각의 위치(좌표)를 찾습니다.
# - min_val, max_val: 결과 맵의 최소/최대 매칭 점수
# - min_loc, max_loc: 최소/최대 매칭 점수가 위치한 좌측 상단 좌표
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# *** 매우 중요한 부분 ***
# 원본 코드에서는 `top_left = max_loc`로 고정되어 있었으나, 이는 논리적 오류를 유발할 수 있습니다.
# 매칭 방법에 따라 최적 위치를 찾는 방식이 다릅니다.
# - TM_SQDIFF, TM_SQDIFF_NORMED: '차이'를 계산하므로, 점수가 '가장 낮은(min_loc)' 곳이 가장 유사한 위치입니다.
# - 나머지 CCOEFF, CCORR 계열: '상관관계'를 계산하므로, 점수가 '가장 높은(max_loc)' 곳이 가장 유사한 위치입니다.
# 아래 코드는 이 두 경우를 모두 처리하도록 수정한 것입니다.
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc

# --- 6. 결과 시각화 ---
# 찾은 위치(좌측 상단)에 템플릿의 너비와 높이를 더해 우측 하단 좌표를 계산합니다.
bottom_right = (top_left[0] + w, top_left[1] + h)
print(bottom_right)
# cv2.rectangle(image, start_point, end_point, color, thickness): 이미지에 사각형을 그립니다.
# - image: 사각형을 그릴 이미지 (img2)
# - start_point: 좌측 상단 좌표 (top_left)
# - end_point: 우측 하단 좌표 (bottom_right)
# - color: 선 색상 (255는 흑백 이미지에서 흰색)
# - thickness: 선 두께 (5)
cv2.rectangle(img2, top_left, bottom_right, 255, 5)

# 'rec'라는 이름의 창에 결과 이미지를 화면에 표시합니다.
cv2.imshow("rec", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
