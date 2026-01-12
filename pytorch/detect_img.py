# ==============================
# YOLOv8 Segmentation 기반
# 차량 Bounding Box + Shape(마스크) 시각화 코드
# ==============================

# 필요한 라이브러리들을 임포트합니다.
from ultralytics import YOLO  # YOLO 모델을 사용하기 위한 ultralytics 패키지
import cv2  # 이미지 입출력 및 그리기용 OpenCV 라이브러리
import os  # 파일 경로 및 이름 처리를 위한 표준 라이브러리
import numpy as np  # 마스크 연산(배열 처리)을 위한 NumPy 라이브러리


# --------------------------------------------------
# 1. 모델 선택 및 로드
# --------------------------------------------------
# True로 설정하면 세그멘테이션 모델(yolov8n-seg.pt)을, False로 설정하면 탐지 모델(yolov8n.pt)을 사용합니다.
USE_SEGMENTATION = True

model_name = "yolov8x-seg.pt" if USE_SEGMENTATION else "yolov8x.pt"
print(f"INFO: Loading model '{model_name}'...")
model = YOLO(model_name)


# --------------------------------------------------
# 2. 이미지 입력 디렉토리 및 파일 목록 설정
# --------------------------------------------------
# 이미지를 읽어올 디렉토리 경로 목록을 설정합니다.
IMG_INPUT_DIRECTORIES = ["img_dir_1", "img_dir_2"]

# 처리할 이미지 파일 경로 목록을 동적으로 생성합니다.
img_paths = []
for directory in IMG_INPUT_DIRECTORIES:
    if os.path.exists(directory) and os.path.isdir(directory):
        for filename in os.listdir(directory):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_paths.append(os.path.join(directory, filename))
        print(f"✅ {directory} 디렉토리에서 이미지 파일을 찾았습니다.")
    else:
        print(f"⚠️ 경고: 입력 이미지 디렉토리 '{directory}'를 찾을 수 없습니다.")

if not img_paths:
    print("🚫 처리할 이미지를 찾지 못했습니다. 스크립트를 종료합니다.")
    exit()

print(f"총 {len(img_paths)}개의 이미지를 처리합니다.")


# --------------------------------------------------
# 이미지 하나씩 반복 처리
# --------------------------------------------------
for img_path in img_paths:

    # ------------------------------
    # 처리 결과 이미지를 저장할 경로 설정
    # ------------------------------
    base = os.path.basename(img_path)
    name, _ = os.path.splitext(base)

    input_dir = os.path.dirname(img_path)
    output_subdir = os.path.join(input_dir, "output")
    os.makedirs(output_subdir, exist_ok=True)

    # --------------------------------------------------
    # 3. YOLO 추론 (Segmentation + Detection)
    # --------------------------------------------------
    results = model(img_path)

    # --------------------------------------------------
    # 원본 이미지 로드 (OpenCV)
    # --------------------------------------------------
    img = cv2.imread(img_path)

    if img is None:
        print(
            f"🚫 오류: 이미지 파일을 찾을 수 없거나 로드할 수 없습니다: {img_path}. 이 이미지는 건너뜁니다."
        )
        continue

    # --------------------------------------------------
    # 3. 주차 공간 정의 (이미지 좌측 절반)
    # --------------------------------------------------
    height, width, _ = img.shape
    left_half_polygon = [[0, 0], [width // 1, 0], [width // 1, height], [0, height]]
    PARKING_SPACES = [{"id": 1, "polygon": left_half_polygon}]

    # --------------------------------------------------
    # 4. 차량 수 초기화
    # --------------------------------------------------
    parking_space_vehicle_count = {space["id"]: 0 for space in PARKING_SPACES}

    # --------------------------------------------------
    # 5. 주차 공간 그리기 (초기 상태)
    # --------------------------------------------------
    for space in PARKING_SPACES:
        polygon = np.array(space["polygon"], np.int32)
        cv2.polylines(img, [polygon], True, (255, 0, 0), 2)

    # --------------------------------------------------
    # 6. 차량 클래스 정의 (COCO Dataset 기준)
    # --------------------------------------------------
    VEHICLE_CLASSES = [2, 5, 7]

    # --------------------------------------------------
    # 7. YOLO 결과 순회 및 차량 수 집계
    # --------------------------------------------------
    for r in results:
        # masks 속성이 있는지 확인하여 분기 처리
        has_masks = USE_SEGMENTATION and r.masks is not None

        if has_masks:
            masks = r.masks.data.cpu().numpy()

        for i, box in enumerate(r.boxes):
            cls = int(box.cls[0])

            if cls in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                is_in_space = False
                for space in PARKING_SPACES:
                    space_id = space["id"]
                    space_polygon = np.array(space["polygon"], np.int32)

                    # --- 분기 처리 ---
                    if has_masks:
                        # [방법 1: Segmentation 마스크 기준]
                        vehicle_mask = masks[i]
                        vehicle_mask = cv2.resize(
                            vehicle_mask, (img.shape[1], img.shape[0])
                        )
                        vehicle_mask = (vehicle_mask > 0.5).astype(np.uint8)

                        parking_space_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(parking_space_mask, [space_polygon], 1)

                        intersection = cv2.bitwise_and(vehicle_mask, parking_space_mask)
                        if np.sum(intersection) > 0:
                            is_in_space = True
                    else:
                        # [방법 2: Bounding Box 중심점 기준]
                        vehicle_center_x = (x1 + x2) // 2
                        vehicle_center_y = (y1 + y2) // 2
                        if (
                            cv2.pointPolygonTest(
                                space_polygon,
                                (vehicle_center_x, vehicle_center_y),
                                False,
                            )
                            >= 0
                        ):
                            is_in_space = True

                    if is_in_space:
                        parking_space_vehicle_count[space_id] += 1
                        break  # 한 차량은 하나의 공간에만 속한다고 가정

                # 감지된 모든 차량에 대해 Bounding Box 그리기
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --------------------------------------------------
    # 9. 주차 공간 시각화 업데이트 (차량 수 반영)
    # --------------------------------------------------
    for space in PARKING_SPACES:
        polygon = np.array(space["polygon"], np.int32)
        space_id = space["id"]
        count = parking_space_vehicle_count[space_id]

        color = (255, 0, 0) if count == 0 else (0, 0, 255)
        text = f"Parking Area {space_id}: {count} Vehicles"

        cv2.polylines(img, [polygon], True, color, 3)
        text_x = polygon[0][0] + 10
        text_y = polygon[0][1] + 40
        cv2.putText(
            img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
        )

        # --------------------------------------------------
        # 10. 결과 이미지 저장
        # --------------------------------------------------
    output_filename = f"result_{name}_{model_name}_{count}.jpg"
    output_path = os.path.join(output_subdir, output_filename)
    cv2.imwrite(output_path, img)
    print(f"✅ 결과 이미지 저장 완료: {output_path}")

print("🎉 모든 이미지 처리가 완료되었습니다.")
