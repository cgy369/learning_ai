import cv2
import numpy as np

# --- 1. 이미지 준비 및 전처리 ---

# cv2.imread(filename): 이미지를 불러옵니다.
img = cv2.imread("opencv/chess/frame01.jpg")
# 원본 이미지를 복사해 둡니다. 나중에 결과와 비교하기 위함입니다.
img_original = img.copy()

# 허프 변환은 엣지(edge) 이미지에 적용해야 하므로, 먼저 흑백으로 변환합니다.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.Canny(image, threshold1, threshold2, ...): 이미지의 엣지를 검출하는 함수입니다.
# 허프 선 변환을 적용하기 전, 반드시 엣지 검출 과정이 선행되어야 합니다.
# - image: 입력 흑백 이미지
# - threshold1, threshold2: 최소/최대 스레숄드. 두 값 사이의 엣지를 검출합니다.
# - apertureSize: Sobel 마스크 크기 (기본값 3)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# --- 2. 허프 선 변환 (Standard Hough Transform) ---

# cv2.HoughLines(image, rho, theta, threshold): 표준 허프 변환을 사용하여 선을 검출합니다.
# 이 함수는 검출된 선을 (rho, theta) 극좌표 형태로 반환합니다.
# - image: 엣지가 검출된 이미지 (Canny의 결과물)
# - rho (ρ): 거리 측정 해상도 (단위: 픽셀). 1로 설정하면 1픽셀의 정밀도를 가집니다.
# - theta (θ): 각도 측정 해상도 (단위: 라디안). np.pi / 180은 1도씩 검사하겠다는 의미입니다.
# - threshold: 선으로 판단하기 위한 최소한의 교차(vote) 횟수. 이 값보다 많은 교차점을 가진 직선만 선으로 인식됩니다.
#              값이 높을수록 더 정확하지만, 검출되는 선의 개수는 줄어듭니다.
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# --- 3. 검출된 선 그리기 ---
# lines 배열을 순회하며 검출된 모든 선을 원본 이미지 위에 그립니다.
# lines 배열의 각 요소는 [[rho, theta]] 형태의 2차원 배열입니다.
for i in range(len(lines)):
    for rho, theta in lines[i]:
        # (rho, theta) 극좌표를 일반적인 직선을 그리기 위한 두 점(x1, y1), (x2, y2)으로 변환하는 과정입니다.
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho  # x0 = r * cos(theta)
        y0 = b * rho  # y0 = r * sin(theta)
        # (x0, y0)는 원점에서 직선까지 수직으로 내린 선의 끝점입니다.

        # 이 점에서 직선의 방향으로 매우 길게(+-1000) 선을 확장하여 두 점을 찾습니다.
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # cv2.line(image, pt1, pt2, color, thickness): 이미지에 선을 그리는 함수입니다.
        # - image: 선을 그릴 이미지
        # - pt1, pt2: 선의 시작점과 끝점
        # - color: 선 색상 (BGR 순서, 여기서는 빨간색)
        # - thickness: 선 두께
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# --- 4. 결과 출력 ---

# np.hstack((img1, img2)): 두 이미지를 수평으로 연결하여 하나의 이미지로 만듭니다.
# 원본 이미지와 선이 그려진 이미지를 나란히 놓고 비교하기 위함입니다.
res = np.hstack((img_original, img))

# 'img'라는 이름의 창에 최종 결과 이미지를 표시합니다.
cv2.imshow("img", res)
cv2.waitKey(0)
cv2.destroyAllWindows()