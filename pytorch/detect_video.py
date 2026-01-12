# ================================================
# YOLOv8 Segmentation + Centroid Tracking
# ================================================

from ultralytics import YOLO
import cv2
import os
import numpy as np
import time
from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image

# --------------------------------------------------
# 설정
# --------------------------------------------------
USE_SEGMENTATION = True
model = YOLO("yolov8x-seg.pt")

TARGET_RESOLUTION = (1280, 720)
DIST_THRESHOLD = 80  # centroid 매칭 거리 (px)
HIDE_TIMEOUT = 1.0  # 사라짐 판정 시간 (초)
MESSAGE_DURATION = 3.0  # 문구 표시 시간 (초)

# --------------------------------------------------
# 폰트
# --------------------------------------------------
try:
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
except:
    font = ImageFont.load_default()


def draw_text(frame, text, pos, color):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# --------------------------------------------------
# 입력 비디오
# --------------------------------------------------
VIDEO_INPUT_DIRECTORIES = ["video_out_1", "video_in_1", "video_in_out_1"]
videos = []

for d in VIDEO_INPUT_DIRECTORIES:
    for f in os.listdir(d):
        if f.endswith(".mp4"):
            videos.append(os.path.join(d, f))

# --------------------------------------------------
# 비디오 처리
# --------------------------------------------------
for video_path in videos:
    print(f"▶ {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        f"output_{os.path.basename(video_path)}",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        TARGET_RESOLUTION,
    )

    # ⭐ Centroid Tracking 상태
    vehicles = {}
    next_vehicle_id = 0
    hidden_parking_count = 0

    disappear_message = None
    disappear_time = None
    disappear_position = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME_SKIP = 2
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    with tqdm(total=total_frames // FRAME_SKIP, desc=base_name) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, TARGET_RESOLUTION)
            h, w, _ = frame.shape

            parking_polygon = np.array(
                [[0, 0], [w // 2, 0], [w // 2, h], [0, h]], np.int32
            )

            results = model(frame, verbose=False)

            # ------------------------------------------
            # 1️⃣ 이번 프레임 detection centroid 수집
            # ------------------------------------------
            detections = []  # (cx, cy, in_parking)

            for r in results:
                masks = (
                    r.masks.data.cpu().numpy() if USE_SEGMENTATION and r.masks else None
                )

                for i, box in enumerate(r.boxes):
                    cls = int(box.cls[0])
                    if cls not in [2, 5, 7]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    in_parking = False
                    if masks is not None:
                        mask = cv2.resize(masks[i], (w, h))
                        mask = (mask > 0.5).astype(np.uint8)
                        parking_mask = np.zeros((h, w), np.uint8)
                        cv2.fillPoly(parking_mask, [parking_polygon], 1)
                        if np.sum(cv2.bitwise_and(mask, parking_mask)) > 0:
                            in_parking = True
                    else:
                        if cv2.pointPolygonTest(parking_polygon, (cx, cy), False) >= 0:
                            in_parking = True

                    detections.append((cx, cy, in_parking))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ------------------------------------------
            # 2️⃣ 기존 차량과 매칭
            # ------------------------------------------
            matched_ids = set()
            now = time.time()

            for cx, cy, in_parking in detections:
                matched_id = None
                min_dist = float("inf")

                for vid, v in vehicles.items():
                    dist = np.linalg.norm(np.array(v["centroid"]) - np.array((cx, cy)))
                    if dist < min_dist and dist < DIST_THRESHOLD:
                        min_dist = dist
                        matched_id = vid

                # 기존 차량
                if matched_id is not None:
                    v = vehicles[matched_id]

                    # ⭐ hidden 차량 재등장
                    if v["hidden"]:
                        hidden_parking_count -= 1
                        v["hidden"] = False

                    v["centroid"] = (cx, cy)
                    v["in_parking"] = in_parking
                    v["last_seen"] = now
                    matched_ids.add(matched_id)

                # 신규 차량
                else:
                    vehicles[next_vehicle_id] = {
                        "centroid": (cx, cy),
                        "in_parking": in_parking,
                        "hidden": False,
                        "last_seen": now,
                    }
                    matched_ids.add(next_vehicle_id)
                    next_vehicle_id += 1

            # ------------------------------------------
            # 3️⃣ 사라짐 처리
            # ------------------------------------------
            for vid, v in vehicles.items():
                if vid in matched_ids:
                    continue

                if not v["hidden"] and now - v["last_seen"] > HIDE_TIMEOUT:
                    v["hidden"] = True
                    if v["in_parking"]:
                        hidden_parking_count += 1
                        disappear_message = "parking"
                        disappear_time = now
                        disappear_position = v["centroid"]

            # ------------------------------------------
            # 4️⃣ 카운트 계산
            # ------------------------------------------
            parking_count_current = sum(
                1 for v in vehicles.values() if not v["hidden"] and v["in_parking"]
            )

            # ------------------------------------------
            # 5️⃣ 시각화
            # ------------------------------------------
            cv2.polylines(frame, [parking_polygon], True, (255, 0, 0), 3)

            for vid, v in vehicles.items():
                cx, cy = v["centroid"]
                color = (0, 255, 0) if not v["hidden"] else (0, 0, 255)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(
                    frame,
                    f"ID {vid}",
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            frame = draw_text(
                frame, f"Parking Count = {parking_count_current}", (20, 20), (0, 255, 0)
            )
            frame = draw_text(
                frame,
                f"Hidden Parking Count = {hidden_parking_count}",
                (20, 55),
                (0, 255, 255),
            )

            if disappear_message and now - disappear_time <= MESSAGE_DURATION:
                frame = draw_text(
                    frame,
                    "여기에 주차된 것으로 예상됩니다.",
                    (disappear_position[0] - 180, disappear_position[1] - 30),
                    (0, 255, 255),
                )

            writer.write(frame)
            pbar.update(1)

    cap.release()
    writer.release()

print("🎉 완료")
