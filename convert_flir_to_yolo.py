import json
import os
from tqdm import tqdm


def convert_coco_to_yolo(json_file, img_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)

    with open(json_file, 'r') as f:
        data = json.load(f)

    img_dict = {img['id']: (img['file_name'], img['width'], img['height']) for img in data['images']}

    # 按照图像分组标注
    ann_dict = {}
    for ann in data['annotations']:
        # 只提取人
        if ann.get('category_id') != 1:
            continue

        img_id = ann['image_id']
        if img_id not in ann_dict:
            ann_dict[img_id] = []
        ann_dict[img_id].append(ann)

    print(f"Converting {json_file}...")
    for img_id, (file_name, width, height) in tqdm(img_dict.items()):
        txt_name = os.path.splitext(os.path.basename(file_name))[0] + '.txt'
        txt_path = os.path.join(output_label_dir, txt_name)

        with open(txt_path, 'w') as f:
            if img_id in ann_dict:
                for ann in ann_dict[img_id]:
                    # COCO bbox 格式: [x_min, y_min, w, h]
                    x_min, y_min, w, h = ann['bbox']

                    # 转换为 YOLO 格式: [class_id, x_center, y_center, w, h] (均需归一化 0~1)
                    x_center = (x_min + w / 2.0) / width
                    y_center = (y_min + h / 2.0) / height
                    w_norm = w / width
                    h_norm = h / height

                    # 类别固定为 0 (代表 Person)
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


if __name__ == '__main__':
    # 1. 转换训练集
    convert_coco_to_yolo(
        json_file='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_train/person_keypoints_train_with_pose.json',
        img_dir='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_train',
        output_label_dir='/mnt/d/Dataset/FLIR_ADAS_v2_Person_YOLO/labels/train'
    )

    # 2. 转换验证集
    convert_coco_to_yolo(
        json_file='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_val/person_keypoints_val_with_pose.json',
        img_dir='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_val',
        output_label_dir='/mnt/d/Dataset/FLIR_ADAS_v2_Person_YOLO/labels/val'
    )