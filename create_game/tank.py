import cv2
import numpy as np
import os
import time


def draw_object(bg, x, y, images):
    dx = x * CELL_SIZE
    dy = y * CELL_SIZE
    bg[dy : dy + CELL_SIZE, dx : dx + CELL_SIZE] = images
    # pass


def move_object(images, direction):
    pass


RunWhile = True


def input_keyhandler(input_key, x, y):
    match input_key:
        case 2555904:
            return min(x + 1, mapsizex - 1), y, right_tank, "RIGHT"
            # tank_x =
            # tank_shape = right_tank
            # my_direction = "RIGHT"
        case 2424832:
            return max(x - 1, 0), y, left_tank, "LEFT"
            # tank_x =
            # tank_shape =
            # my_direction =
        case 2490368:
            return x, max(y - 1, 0), top_tank, "TOP"
            # tank_y =
            # tank_shape = top_tank
            # my_direction = ""
        case 2621440:
            return x, min(y + 1, mapsizey - 1), bottom_tank, "BOTTOM"
            # tank_y =
            # tank_shape = bottom_tank
            # my_direction = ""
        case 32:
            bx, by = DIR_OFFSET[my_direction]
            bx = x + bx
            by = y + by
            if 0 <= bx < mapsizex and 0 <= by < mapsizey:
                if walls[by, bx] != 0:
                    walls[by, bx] = 0
                    destroy_list.append(
                        {
                            "x": bx,
                            "y": by,
                            "remain_count": -5,
                        }
                    )
                    return x, y, tank_shape, my_direction
                if bullet_list:  # 하나의 총알만 쏘기
                    # b_list = bullet_list[-1]
                    # print(b_list["delay"])
                    # if not (time.time() - b_list["delay"] > 100):
                    return x, y, tank_shape, my_direction
                bullet_list.append(
                    {
                        "x": bx,
                        "y": by,
                        "dir": my_direction,
                        "remain_count": -5,
                        "pixelmove": default_pixel,
                        "sub": 0,
                        "delay": time.time(),
                    }
                )
                grid[by, bx] = -5
                return x, y, tank_shape, my_direction
            return x, y, tank_shape, my_direction
        case 27:
            RunWhile = False
            return x, y, tank_shape, my_direction
            # cv2.destroyAllWindows()
        case _:
            return x, y, tank_shape, my_direction


# 맵 사이즈 (칸수)
mapsizex, mapsizey = 20, 20
# 한칸의 사이즈(픽셀)
CELL_SIZE = 32
# 최초 방향
my_direction = "TOP"
# 탱크 이미지
temp_tank = cv2.imread("create_game/main_tank.png", 3)
destroy = cv2.imread("create_game/boom.png", 3)
destroy_image = cv2.resize(
    destroy, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_CUBIC
)
# 벽 이미지
wall = cv2.imread("create_game/wall.jpg", 3)
# 이미지 전처리 과정, 돌리고 줄이고,
bullet_right = cv2.imread("create_game/bullet.png", 3)
temp_tank = cv2.resize(temp_tank, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_CUBIC)
wall = cv2.resize(wall, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_CUBIC)
right_tank = cv2.rotate(temp_tank, cv2.ROTATE_90_CLOCKWISE)
bottom_tank = cv2.rotate(right_tank, cv2.ROTATE_90_CLOCKWISE)
left_tank = cv2.rotate(bottom_tank, cv2.ROTATE_90_CLOCKWISE)
top_tank = cv2.rotate(left_tank, cv2.ROTATE_90_CLOCKWISE)

right_bullet = cv2.resize(
    bullet_right,
    # (int(CELL_SIZE / 3), int(CELL_SIZE / 3)),
    (int(CELL_SIZE), int(CELL_SIZE)),
    interpolation=cv2.INTER_CUBIC,
)
bottom_bullet = cv2.rotate(right_bullet, cv2.ROTATE_90_CLOCKWISE)
left_bullet = cv2.rotate(bottom_bullet, cv2.ROTATE_90_CLOCKWISE)
top_bullet = cv2.rotate(left_bullet, cv2.ROTATE_90_CLOCKWISE)
# 탱크의 현 좌표
tank_x, tank_y = 0, 0
# 벽이 있는 위치 리스트로 저장
walls = np.zeros((mapsizey, mapsizex), np.int16)
enemy_grid = np.zeros((mapsizey, mapsizex), np.int16)
# 벽만들기
walls[mapsizey // 2, 0 : int(mapsizex * 0.8)] = 10
walls[0 : (mapsizey // 2) - 2, int(mapsizex * 0.8)] = 10
# 탱크 이미지 저장소
tank_shape = temp_tank
# 총알 리스트(딕셔너리)
bullet_list = []
# 파괴 리스트
destroy_list = []
# 적 리스트
enemy_list = [
    {"x": 10, "y": 10, "dir": "TOP", "alive": True},
    {"x": 11, "y": 11, "dir": "TOP", "alive": True},
    {"x": 12, "y": 12, "dir": "TOP", "alive": True},
]
for e in enemy_list:
    enemy_grid[e["y"], e["x"]] = 2

# 총알이 한 셀에서 몇번 움직일지 지정
default_pixel = 6
# PIXEL_STEP = CELL_SIZE // default_pixel
PIXEL_STEP = 1
# 디렉토리에 따라 x,y 좌표
DIR_OFFSET = {
    "RIGHT": (PIXEL_STEP, 0),
    "LEFT": (-PIXEL_STEP, 0),
    "TOP": (0, -PIXEL_STEP),
    "BOTTOM": (0, PIXEL_STEP),
}
# 방향에 따라 총알 이미지
DIR_BULLET = {
    "RIGHT": right_bullet,
    "LEFT": left_bullet,
    "TOP": top_bullet,
    "BOTTOM": bottom_bullet,
}
while RunWhile:
    # 배경화면 지정, 흰색으로 채운다
    bgimg = np.full((mapsizey * CELL_SIZE, mapsizex * CELL_SIZE, 3), 255, np.uint8)
    # 행렬 그리드 그려서 형태 파악하기 (여기에 값 넣어서 이미지 맵핑? 할거)
    grid = np.zeros((mapsizey, mapsizex), np.int16)
    # 최소 딜레이 50ms
    key = cv2.waitKeyEx(10)
    bf_tankx, bf_tanky = tank_x, tank_y
    tank_x, tank_y, tank_shape, my_direction = input_keyhandler(key, tank_x, tank_y)
    # match key:
    #     case 2555904:
    #         tank_x = min(tank_x + 1, mapsizex - 1)
    #         tank_shape = right_tank
    #         my_direction = "RIGHT"
    #     case 2424832:
    #         tank_x = max(tank_x - 1, 0)
    #         tank_shape = left_tank
    #         my_direction = "LEFT"
    #     case 2490368:
    #         tank_y = max(tank_y - 1, 0)
    #         tank_shape = top_tank
    #         my_direction = "TOP"
    #     case 2621440:
    #         tank_y = min(tank_y + 1, mapsizey - 1)
    #         tank_shape = bottom_tank
    #         my_direction = "BOTTOM"
    #     case 32:
    #         bx, by = DIR_OFFSET[my_direction]
    #         bx = tank_x + bx
    #         by = tank_y + by
    #         if 0 <= bx < mapsizex and 0 <= by < mapsizey:
    #             if walls[by, bx] != 0:
    #                 walls[by, bx] = 0
    #                 destroy_list.append(
    #                     {
    #                         "x": bx,
    #                         "y": by,
    #                         "remain_count": -5,
    #                     }
    #                 )
    #                 continue
    #             if bullet_list:  # 하나의 총알만 쏘기
    #                 # b_list = bullet_list[-1]
    #                 # print(b_list["delay"])
    #                 # if not (time.time() - b_list["delay"] > 100):
    #                 continue
    #             bullet_list.append(
    #                 {
    #                     "x": bx,
    #                     "y": by,
    #                     "dir": my_direction,
    #                     "remain_count": -5,
    #                     "pixelmove": default_pixel,
    #                     "sub": 0,
    #                     "delay": time.time(),
    #                 }
    #             )
    #             grid[by, bx] = -5

    #     case 27:
    #         break
    new_enemy_list = []
    for enemy in enemy_list:
        x, y = enemy["x"], enemy["y"]
        if enemy_grid[y, x] == 0:
            enemy["alive"] = False
            continue
        if not enemy["alive"]:
            continue

        enemy["alive"] = True
        new_enemy_list.append(enemy)
        enemy_grid[y, x] = 2
        grid[y, x] = 2
        # px, py = x * CELL_SIZE, y * CELL_SIZE
        draw_object(bgimg, x, y, temp_tank)
        # bgimg[py : py + CELL_SIZE, px : px + CELL_SIZE] = temp_tank
    enemy_list = new_enemy_list
    new_bullets = []
    for b in bullet_list:
        if b["remain_count"] == 0:
            continue
            # print("Bullet Remove")
            # continue
        print(enemy_grid)
        if enemy_grid[b["y"], b["x"]] == 2:
            print("Hit", enemy_list)
            enemy_grid[b["y"], b["x"]] = 0
            b["remain_count"] = 0
            destroy_list.append(
                {
                    "x": b["x"],
                    "y": b["y"],
                    "remain_count": -5,
                }
            )
            continue

        new_bullets.append(b)
        addx, addy = DIR_OFFSET[b["dir"]]  # 움직이는 방향, 1,0으로 나타낸다.
        px = b["x"] * (CELL_SIZE) + addx * (CELL_SIZE // default_pixel * (b["sub"]))
        # 현재 x 위치를 나타낸다.
        py = b["y"] * (CELL_SIZE) + addy * (CELL_SIZE // default_pixel * (b["sub"]))
        # 현재 y 위치를 나타낸다
        b["sub"] += 1
        if b["sub"] > default_pixel:
            b["sub"] = 0
            b["remain_count"] = b["remain_count"] + 1
            b["x"] = b["x"] + addx
            b["y"] = b["y"] + addy
            if not (0 <= b["x"] < mapsizex and 0 <= b["y"] < mapsizey):
                b["remain_count"] = 0
                continue
            # print(b["y"], b["x"])
            if walls[b["y"], b["x"]] != 0:
                walls[b["y"], b["x"]] = 0
                b["remain_count"] = 0
                destroy_list.append(
                    {
                        "x": b["x"],
                        "y": b["y"],
                        "remain_count": -5,
                    }
                )
                continue

        if not (0 <= px < mapsizex * CELL_SIZE and 0 <= py < mapsizey * CELL_SIZE):
            b["remain_count"] = 0
            print("remove Bl")
            continue
        # print(px, mapsizex * CELL_SIZE)
        # print(b["x"], b["y"], px, py)

        if not (
            0 <= px + CELL_SIZE < mapsizex * CELL_SIZE
            and 0 <= py + CELL_SIZE < mapsizey * CELL_SIZE
        ):
            b["remain_count"] = 0
            print("remove Bl")
            continue
        grid[b["y"], b["x"]] = -5
        draw_object(bgimg, b["x"], b["y"], DIR_BULLET[b["dir"]])
        # bgimg[py : py + CELL_SIZE, px : px + CELL_SIZE] = DIR_BULLET[b["dir"]]
        # print(b["x"], b["y"], b["dir"], b["remain_count"], DIR_OFFSET[b["dir"]])
    bullet_list = new_bullets
    for y in range(mapsizey):  # 벽 그리기
        for x in range(mapsizex):
            if walls[y, x] != 0:
                # print(walls[y, x])
                grid[y, x] = 10
                # wall_x = x * CELL_SIZE
                # wall_y = y * CELL_SIZE
                draw_object(bgimg, x, y, wall)
                # bgimg[wall_y : wall_y + CELL_SIZE, wall_x : wall_x + CELL_SIZE] = wall
    new_d_list = []
    for des in destroy_list:
        print(des["x"], des["y"], des["remain_count"], "벽부서짐. 탱크터짐 이벤트 여기")
        # dx = des["x"] * CELL_SIZE
        # dy = des["y"] * CELL_SIZE
        draw_object(bgimg, des["x"], des["y"], destroy_image)
        # bgimg[dy : dy + CELL_SIZE, dx : dx + CELL_SIZE] = destroy_image
        grid[des["y"], des["x"]] = 100
        des["remain_count"] += 1
        if des["remain_count"] > 0:
            # destroy_list.remove(des)
            continue
        new_d_list.append(des)
    destroy_list = new_d_list
    if walls[tank_y, tank_x] != 0:
        tank_x, tank_y = bf_tankx, bf_tanky
    if enemy_grid[tank_y, tank_x] != 0:
        tank_x, tank_y = bf_tankx, bf_tanky
    draw_object(bgimg, tank_x, tank_y, tank_shape)
    # bgimg[py : py + CELL_SIZE, px : px + CELL_SIZE] = tank_shape
    grid[tank_y, tank_x] = 1

    cv2.imshow("game", bgimg)
    print(grid)
    # print(bullet_list)
