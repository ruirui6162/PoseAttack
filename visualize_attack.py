import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import FLIRKeypointDataset, collate_fn
from patch_model import PoseAwareAdversarialPatch


def draw_boxes(image_np, boxes, color=(0, 255, 0), thickness=2):
    """在 numpy 图像上画检测框"""
    img = image_np.copy()
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(img, f"Person: {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return img


def visualize_patches(num_images=800):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "attack_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Models...")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom',
                          path='/mnt/d/PycharmProjects/PoseAttack/yolov5-master/runs/train/exp/weights/best.pt',  # 本地权重路径
                          force_reload=True)
    yolo_model.conf = 0.25
    yolo_model.classes = [0]

    patch_model = PoseAwareAdversarialPatch(device=device)
    weight_path = "weights/adv_patch_5_epoch_20.pt"
    if os.path.exists(weight_path):
        patch_model.load_state_dict(torch.load(weight_path))
    patch_model.eval()

    test_dataset = FLIRKeypointDataset(
        img_dir='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_val',
        ann_file='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_val/person_keypoints_val_with_pose.json'
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print(f"Generating top {num_images} visualizations...")
    count = 0
    with torch.no_grad():
        for images, keypoints, bboxes in test_loader:
            if count >= num_images:
                break

            images = images.to(device)
            keypoints = [k.to(device) for k in keypoints]
            bboxes = [b.to(device) for b in bboxes] # 把边界框发到GPU

            if keypoints[0].shape[0] == 0:
                continue  # 跳过没有人的图片

            # --- 1. 干净图像 ---
            img_clean_res = torch.nn.functional.interpolate(images, size=(640, 640), mode='bilinear')
            img_clean_np = (img_clean_res.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            img_clean_np_bgr = cv2.cvtColor(img_clean_np, cv2.COLOR_RGB2BGR)  # OpenCV 需要 BGR

            res_clean = yolo_model(img_clean_np)
            boxes_clean = res_clean.xyxy[0].cpu().numpy() if len(res_clean.xyxy[0]) else []

            # --- 2. 对抗图像 ---
            patched_images, alphas = patch_model(images, keypoints, bboxes)
            img_adv_res = torch.nn.functional.interpolate(patched_images, size=(640, 640), mode='bilinear')
            img_adv_np = (img_adv_res.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            img_adv_np_bgr = cv2.cvtColor(img_adv_np, cv2.COLOR_RGB2BGR)

            res_adv = yolo_model(img_adv_np)
            boxes_adv = res_adv.xyxy[0].cpu().numpy() if len(res_adv.xyxy[0]) else []

            # --- 3. 绘图与拼接 ---
            # 干净图画绿框，对抗图画红框
            vis_clean = draw_boxes(img_clean_np_bgr, boxes_clean, color=(0, 255, 0))
            vis_adv = draw_boxes(img_adv_np_bgr, boxes_adv, color=(0, 0, 255))

            # 在图片上写字标注
            cv2.putText(vis_clean, f"Clean: {len(boxes_clean)} detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_adv, f"Attacked: {len(boxes_adv)} detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 左右拼接合并为一张图
            combined_img = np.hstack((vis_clean, vis_adv))

            # 保存
            save_path = os.path.join(output_dir, f"compare_{count:03d}.jpg")
            cv2.imwrite(save_path, combined_img)
            print(f"Saved: {save_path}")

            count += 1

    print("Visualization Done! Check the 'attack_visualizations' folder.")


if __name__ == '__main__':
    visualize_patches()