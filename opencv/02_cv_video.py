import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import cv2 as cv
import numpy as np

available = []
for i in range(10):
    cap = cv.VideoCapture(i, cv.CAP_DSHOW)  # 백엔드로 카메라 열기
    if cap.isOpened():
        available.append(i)
        print(f"Camera {i} is available")
        cap.release()
print(f"사용 가능 카메라 : {available}")

cap = cv.VideoCapture(available[0])
print(f"Width : {cap.get(3)},Height : {cap.get(4)}")
# | 숫자   | 상수 이름                    | 의미        |
# | ---- | ------------------------ | --------- |
# | `0`  | `CAP_PROP_POS_MSEC`      | 현재 위치(ms) |
# | `1`  | `CAP_PROP_POS_FRAMES`    | 현재 프레임 번호 |
# | `2`  | `CAP_PROP_POS_AVI_RATIO` | 영상 위치 비율  |
# | `3`  | `CAP_PROP_FRAME_WIDTH`   | 프레임 너비    |
# | `4`  | `CAP_PROP_FRAME_HEIGHT`  | 프레임 높이    |
# | `5`  | `CAP_PROP_FPS`           | FPS       |
# | `6`  | `CAP_PROP_FOURCC`        | 코덱        |
# | `7`  | `CAP_PROP_FRAME_COUNT`   | 전체 프레임 수  |
# | `8`  | `CAP_PROP_FORMAT`        | 픽셀 포맷     |
# | `10` | `CAP_PROP_BRIGHTNESS`    | 밝기        |
# | `11` | `CAP_PROP_CONTRAST`      | 대비        |
# | `12` | `CAP_PROP_SATURATION`    | 채도        |
# | `13` | `CAP_PROP_HUE`           | 색상        |
# | `14` | `CAP_PROP_GAIN`          | 게인        |
# | `15` | `CAP_PROP_EXPOSURE`      | 노출        |

fourcc = cv.VideoWriter_fourcc(*"DIVX")  # 인코더 선택
out = cv.VideoWriter(
    "output.avi", fourcc, cap.get(cv.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4)))
)
print(cap.get(cv.CAP_PROP_FPS))
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # print(frame.shape)
        b, g, r = cv.split(frame)
        _, r_binary = cv.threshold(r, 128, 255, cv.THRESH_BINARY_INV)
        _, g_binary = cv.threshold(g, 128, 255, cv.THRESH_BINARY_INV)
        _, b_binary = cv.threshold(b, 128, 255, cv.THRESH_BINARY_INV)
        a = np.full(b.shape, 0, dtype=b.dtype)
        gray = cv.merge([b_binary, g_binary, r_binary])
        cv.imshow("grey", gray)
        keys = cv.waitKey(1)
        # 0인경우 키를 누를떄까지 무한 대기, waitKey가 없을 경우 그리기만하고 갱신은 하지 않는다.
        if keys == 27:
            break
        out.write(frame)
        # if cv.waitKey(1) & 0xFF == ord("q"):
        # break
cap.release()
out.release()


vid = cv.VideoCapture("output.avi")
fps = vid.get(cv.CAP_PROP_FPS)
delay = int(1000 / fps)
while vid.isOpened():
    ret, frame = vid.read()
    cv.imshow("sigma", frame)
    cv.waitKey(delay)
cap.release()
cv.destroyAllWindows()
