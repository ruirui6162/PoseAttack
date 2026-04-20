import json
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def calculate_iou(box1, box2):
    """
    计算两个边界框的 IoU。格式要求：[x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def generate_keypoints_for_json(img_dir, input_json_path, output_json_path):
    print(f"Loading YOLOv8-Pose model...")
    # 自动下载 yolov8m-pose.pt (Medium大小，精度和速度平衡很好)
    pose_model = YOLO('yolov8m-pose.pt')

    print(f"Loading annotations from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # 建立 image_id 到 image_info 的映射
    imgid2info = {img['id']: img for img in coco_data['images']}

    # 建立 image_id 到 annotations 的映射
    imgid2anns = {}
    for ann in coco_data['annotations']:
        # 只处理类别为 person
        img_id = ann['image_id']
        if img_id not in imgid2anns:
            imgid2anns[img_id] = []
        imgid2anns[img_id].append(ann)

    print("Extracting keypoints...")
    matched_persons = 0
    total_persons = 0

    # 遍历所有有标注的图片
    for img_id, anns in tqdm(imgid2anns.items()):
        img_info = imgid2info[img_id]
        img_path = os.path.join(img_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        # 读取图像
        img = cv2.imread(img_path)
        if img is None: continue

        # 运行姿态估计模型 (因为 FLIR 是红外图，特征较弱，所以降低置信度阈值尽力提取)
        results = pose_model(img, verbose=False, conf=0.1)[0]

        pred_bboxes = results.boxes.xyxy.cpu().numpy()  # [N, 4] -> x1, y1, x2, y2

        # 只有当模型检测到人且有关键点时才处理
        has_keypoints = hasattr(results, 'keypoints') and results.keypoints is not None and len(results.keypoints) > 0
        if has_keypoints:
            pred_kpts = results.keypoints.data.cpu().numpy()  # [N, 17, 3] -> x, y, conf
        else:
            pred_kpts = []

        # 遍历原 JSON 中的每一个 Ground Truth 框，为其匹配关键点
        for ann in anns:
            total_persons += 1
            # 原 COCO bbox 格式: [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            gt_box = [x, y, x + w, y + h]

            best_iou = 0.0
            best_kpts = None

            # 寻找 IoU 最大的预测框
            for i, p_box in enumerate(pred_bboxes):
                iou = calculate_iou(gt_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    if has_keypoints and i < len(pred_kpts):
                        best_kpts = pred_kpts[i]

            # 如果匹配成功且 IoU 大于 0.3
            if best_iou > 0.3 and best_kpts is not None:
                matched_persons += 1
                # 转换为 COCO 标准格式: [x, y, visibility] * 17，共 51 个数值
                # COCO visibility: 0=没标注/不可见, 1=遮挡但被标注, 2=清晰可见
                coco_kpts = []
                for kpt in best_kpts:
                    kx, ky, kconf = kpt
                    # 置信度转换：大于0.4认为可见(2)，0.1~0.4认为遮挡(1)，小于0.1认为不可见(0)
                    v = 2 if kconf > 0.4 else (1 if kconf > 0.1 else 0)
                    if v == 0:
                        kx, ky = 0.0, 0.0
                    coco_kpts.extend([float(kx), float(ky), int(v)])

                ann['keypoints'] = coco_kpts
                ann['num_keypoints'] = sum(1 for i in range(2, 51, 3) if coco_kpts[i] > 0)
            else:
                # 如果没匹配上姿态，填充 0
                ann['keypoints'] = [0] * 51
                ann['num_keypoints'] = 0

    print(f"Extraction done! Matched {matched_persons}/{total_persons} persons with keypoints.")

    # 保存新的 JSON 文件
    print(f"Saving new annotations to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)
    print("All Finished!")


if __name__ == '__main__':

    # --- 1. 处理验证集 (Val) ---
    generate_keypoints_for_json(
        img_dir='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_val',
        input_json_path='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_val/coco.json',
        output_json_path='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_val/person_keypoints_val_with_pose.json'
    )

    # --- 2. 处理训练集 (Train) ---
    generate_keypoints_for_json(
        img_dir='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_train',
        input_json_path='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_train/coco.json',
        output_json_path='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_train/person_keypoints_train_with_pose.json'
    )