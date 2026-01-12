# -*- coding:utf-8 -*-

"""
# Fourier Transform (푸리에 변환)
    - 개념: 시간 도메인(X축이 시간)의 신호를 주파수 도메인(X축이 주파수)으로 변환하는 수학적 기법입니다.
    - 이미지에서의 의미: 이미지는 2D 신호로 볼 수 있습니다. 푸리에 변환을 적용하면 이미지를 다양한 주파수 성분으로 분해합니다.
        - 저주파(Low Frequency): 이미지에서 색상이나 밝기가 부드럽고 천천히 변하는 부분 (예: 하늘, 벽, 피부 등). 스펙트럼의 '중심'에 나타납니다.
        - 고주파(High Frequency): 이미지에서 색상이나 밝기가 급격하게 변하는 부분 (예: 경계선, 모서리, 글자, 질감 등). 스펙트럼의 '가장자리'에 나타납니다.
    - 활용:
        1. 이미지 필터링: 특정 주파수 대역을 제거하거나 증폭시킬 수 있습니다.
           - 저주파 통과 필터(Low Pass Filter, LPF): 고주파를 제거하여 이미지를 부드럽게 만들거나(Blur), 노이즈를 줄입니다.
           - 고주파 통과 필터(High Pass Filter, HPF): 저주파를 제거하여 이미지의 경계선이나 윤곽선을 강조합니다. (이 예제에서 사용하는 방식)
        2. 이미지 압축, 패턴 분석 등 다양한 분야에 사용됩니다.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


# --- 1. 이미지 불러오기 및 전처리 ---

# cv2.imread(filename, [flag]): 이미지를 불러오는 함수입니다.
# - filename: 불러올 이미지의 경로.
# - flag: 이미지를 불러올 때의 형식 (기본값은 컬러).
img = cv2.imread("opencv/lena.png")

# OpenCV는 이미지를 BGR(파랑, 초록, 빨강) 순서로 불러옵니다.
# Matplotlib으로 이미지를 제대로 표시하려면 RGB(빨강, 초록, 파랑) 순서로 바꿔줘야 합니다.
# cv2.split(img): 이미지를 각 색상 채널(B, G, R)로 분리합니다.
b, g, r = cv2.split(img)
# cv2.merge([r, g, b]): 분리된 채널을 RGB 순서로 다시 합칩니다.
img = cv2.merge([r, g, b])

# 푸리에 변환은 보통 단일 채널(흑백) 이미지에 적용하므로, 컬러 이미지를 흑백으로 변환합니다.
# cv2.cvtColor(src, code): 이미지의 색상 체계를 변환하는 함수입니다.
# - src: 원본 이미지.
# - code: 변환할 색상 코드 (여기서는 BGR을 GRAY(흑백)로).
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --- 2. 푸리에 변환 및 스펙트럼 시각화 ---

"""
# 푸리에 변환을 적용하면 결과는 복소수(complex number) 배열로 나옵니다.
# 초기 결과에서 저주파(DC 성분)는 좌측 상단(0,0)에 위치합니다.
# 이를 보기 쉽게 만들기 위해 저주파를 이미지의 중심으로 이동시키고(fftshift),
# 값의 범위가 매우 크므로 로그 스케일(Log Scaling)을 적용하여 시각화합니다.
"""

# np.fft.fft2(input_array): 2차원 고속 푸리에 변환(FFT)을 수행하는 함수입니다.
# - input_array: 변환할 2D 배열(이미지).
# 결과 'f'는 주파수 도메인의 복소수 배열입니다.
f = np.fft.fft2(img)

# np.fft.fftshift(array): 주파수 스펙트럼을 재배치하는 함수입니다.
# 0 주파수 성분(저주파)을 배열의 중앙으로 옮깁니다. 이렇게 하면 중앙에 저주파, 주변부에 고주파가 모여 분석이 용이해집니다.
# - array: 푸리에 변환 결과('f').
fshift = np.fft.fftshift(f)

# 푸리에 변환 결과(복소수)를 시각화하기 위해 크기(Magnitude) 스펙트럼을 계산합니다.
# np.abs(fshift): 각 복소수 픽셀의 크기(magnitude)를 계산합니다.
# np.log(...): 값의 편차가 매우 크므로 로그 스케일로 변환하여 명암 대비를 높입니다.
# 20 * ...: 스케일링 팩터.
magnitude_spectrum = 20 * np.log(np.abs(fshift))


# --- 3. 고주파 통과 필터(High Pass Filter) 적용 ---
# 저주파를 제거하여 이미지의 경계선을 찾는 과정입니다.

rows, cols = img.shape
# 이미지의 중심 좌표를 구합니다. 정수 좌표를 얻기 위해 // 연산자 사용.
crow, ccol = rows // 2, cols // 2

# 필터링: 주파수 스펙트럼의 중심(저주파 영역)을 조작합니다.
# 여기서는 중심으로부터 10x10 픽셀 크기의 사각형 영역(가장 중요한 저주파 성분)을 1로 설정합니다.
# 로그 스케일에서 1은 매우 작은 값이므로, 사실상 저주파 성분을 제거하는 효과(고주파 통과 필터)를 냅니다.
d = 15
fshift[crow - d : crow + d, ccol - d : ccol + d] = 1


# --- 4. 역 푸리에 변환 ---
# 필터링된 주파수 스펙트럼을 다시 이미지로 되돌립니다.

# np.fft.ifftshift(array): fftshift의 역연산. 중앙에 있던 0 주파수 성분을 다시 좌측 상단으로 옮깁니다.
f_ishift = np.fft.ifftshift(fshift)
# np.fft.ifft2(array): 2차원 역 푸리에 변환을 수행하여 이미지를 복원합니다.
img_back = np.fft.ifft2(f_ishift)
# 결과는 여전히 복소수이므로, np.abs()로 크기를 취해 실제 이미지(실수)로 만듭니다.
img_back = np.abs(img_back)


# --- 5. 후처리 및 결과 시각화 ---

# cv2.threshold() 같은 OpenCV 함수에 사용하거나, 이미지로 제대로 표시하기 위해
# 실수(float) 타입의 배열을 8비트 부호 없는 정수(uint8) 타입으로 변환합니다.
img_new = np.uint8(img_back)

# cv2.threshold(src, thresh, maxval, type): 이미지를 이진화하는 함수. 경계선을 더 명확하게 보기 위함.
# - src: 입력 이미지 (여기서는 `img_new`).
# - thresh: 임계값 (30).
# - maxval: 임계값보다 클 때 적용할 값 (255, 흰색).
# - type: THRESH_BINARY_INV는 임계값보다 크면 0, 작으면 maxval을 적용 (검은 배경에 흰 경계선).
ret, thresh = cv2.threshold(img_new, 30, 255, cv2.THRESH_BINARY_INV)

# Matplotlib을 사용하여 4개의 이미지를 한 번에 비교/표시합니다.
# plt.subplot(행, 열, 인덱스): 플롯을 여러 개로 나누어 그립니다.
# plt.imshow(image, cmap='gray'): 이미지를 표시합니다. 흑백이므로 'gray' 컬러맵 사용.
# plt.title(...), plt.xticks([]), plt.yticks([]): 제목을 설정하고 x, y축 눈금을 제거합니다.

plt.subplot(221), plt.imshow(img, cmap="gray")
plt.title("Input Image"), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(img_back, cmap="gray")
plt.title("Image after HPF"), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(thresh, cmap="gray")
plt.title("Threshold With FT"), plt.xticks([]), plt.yticks([])

# plt.show(): 준비된 모든 플롯을 화면에 표시합니다.
plt.show()
