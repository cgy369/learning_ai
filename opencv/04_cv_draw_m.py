import cv2

events = [i for i in dir(cv2) if "EVENT" in i]
print(events)

import numpy as np

draw_flag, gather_xy = False, False
start_x, start_y = 0, 0
current_x, current_y = 0, 0


def draw_circle(event, x, y, flags, param):
    global start_x, start_y, draw_flag, gather_xy, current_x, current_y, img
    if event == cv2.EVENT_LBUTTONDBLCLK:
        start_x = current_x
        start_y = current_y
        cv2.rectangle(
            img, (start_x, start_y), (current_x, current_y), (255, 0, 0), -1
        )  # -1이면 내부 채움
        gather_xy = False
    elif event == cv2.EVENT_LBUTTONUP:
        start_x, start_y = x, y
        if draw_flag == True:
            gather_xy = True
            draw_flag = False

    if event == cv2.EVENT_MOUSEMOVE:
        current_x, current_y = x, y
        if gather_xy:
            # img = np.zeros((512, 512, 3), np.uint8)
            cv2.rectangle(
                img, (start_x, start_y), (current_x, current_y), (255, 0, 0), -1
            )
            print(x, y)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_circle)

while True:
    cv2.imshow("image", img)
    keys = cv2.waitKey(1)
    # print(keys)
    if keys != -1:
        print(keys)
    if keys == 114:
        draw_flag = True
    if keys == 27:
        break
cv2.destroyAllWindows()
