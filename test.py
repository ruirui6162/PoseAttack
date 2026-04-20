import torch
import torchvision.ops as ops
from dataset import FLIRKeypointDataset, collate_fn
from patch_model import PoseAwareAdversarialPatch
from torch.utils.data import DataLoader
from tqdm import tqdm


def test_patch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading YOLO model via Ultralytics for evaluation...")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                path='/mnt/d/PycharmProjects/PoseAttack/yolov5-master/runs/train/exp/weights/best.pt',
                                force_reload=True)
    yolo_model = yolo_model.to(device).eval()

    yolo_model.conf = 0.25
    yolo_model.iou = 0.45
    yolo_model.classes = [0]

    weight_path = "./weights/adv_patch_6_epoch_20.pt"
    patch_model = PoseAwareAdversarialPatch(device=device)
    try:
        patch_model.load_state_dict(torch.load(weight_path))
        print(f"Loaded patch weights from {weight_path}")
    except FileNotFoundError:
        print(f"Warning: {weight_path} not found. Running with random initialized patches!")
    patch_model.eval()

    MAX_PATCHES = 5  # 必须与你 train.py 中设置的数字保持一致！

    with torch.no_grad():
        # 获取训练好的 alphas
        current_alphas = torch.sigmoid(patch_model.alpha_logits)
        # 找到第 MAX_PATCHES 大的值作为及格线阈值
        kth_threshold = torch.topk(current_alphas, MAX_PATCHES)[0][-1]

        # 将达不到阈值的 Logit 强制设为一个极小的负数
        # 达到阈值的保持不变（Sigmoid 后 > 0.5，被保留）
        mask = current_alphas < kth_threshold
        patch_model.alpha_logits[mask] = -999.0

        print(f"\n[Hard Constraint Applied] Strict Max Patches Allowed: {MAX_PATCHES}")

    test_dataset = FLIRKeypointDataset(
        img_dir='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_val',
        ann_file='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_val/person_keypoints_val_with_pose.json'
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 我们现在只关心【被攻击的特定目标】是否被检测到
    targeted_clean_detected = 0
    targeted_adv_detected = 0
    total_targeted_persons = 0

    print("Evaluating Targeted Adversarial Attack Performance...")
    with torch.no_grad():
        for images, keypoints, bboxes in tqdm(test_loader):
            images = images.to(device)
            keypoints = [k.to(device) for k in keypoints]
            bboxes = [b.to(device) for b in bboxes]

            if bboxes[0].shape[0] == 0:
                continue

            total_targeted_persons += bboxes[0].shape[0]

            # 将 GT 框从 COCO 格式 [x, y, w, h] 转换为 YOLO 匹配所需的 [x1, y1, x2, y2]
            gt_boxes = bboxes[0].clone()
            gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]  # x2 = x + w
            gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]  # y2 = y + h

            img_clean_np = (images.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            results_clean = yolo_model(img_clean_np)

            if len(results_clean.xyxy[0]) > 0:
                pred_clean_boxes = results_clean.xyxy[0][:, :4]  # 提取预测框
                # 计算 GT 和 预测框的 IoU 矩阵
                ious_clean = ops.box_iou(gt_boxes, pred_clean_boxes)  # [num_gt, num_pred]
                # 对于每一个 GT(也就是我们锁定的目标)，看是否有一个预测框和它的 IoU > 0.45
                matched_clean = (ious_clean.max(dim=1).values > 0.45).sum().item()
                targeted_clean_detected += matched_clean

            patched_images, alphas = patch_model(images, keypoints, bboxes)
            img_adv_np = (patched_images.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            results_adv = yolo_model(img_adv_np)

            if len(results_adv.xyxy[0]) > 0:
                pred_adv_boxes = results_adv.xyxy[0][:, :4]
                ious_adv = ops.box_iou(gt_boxes, pred_adv_boxes)
                matched_adv = (ious_adv.max(dim=1).values > 0.45).sum().item()
                targeted_adv_detected += matched_adv

    clean_recall = targeted_clean_detected / total_targeted_persons if total_targeted_persons > 0 else 0
    adv_recall = targeted_adv_detected / total_targeted_persons if total_targeted_persons > 0 else 0

    # 真实的攻击成功率：(原本能识别出的目标 - 攻击后识别不出的目标) / 原本能识别出的目标
    true_attack_success_rate = (
                                           targeted_clean_detected - targeted_adv_detected) / targeted_clean_detected if targeted_clean_detected > 0 else 0

    print("\n===== 核心定向攻击测试结果 (Targeted Evaluation) =====")
    print(f"总计可被物理攻击的目标人数 (Valid Targeted GT): {total_targeted_persons}")
    print(f"原图成功识别的目标数 (Clean Matched): {targeted_clean_detected} (基础召回率: {clean_recall:.2%})")
    print(f"贴图后成功识别的目标数 (Adv Matched): {targeted_adv_detected} (攻击后召回率: {adv_recall:.2%})")
    print(f"真实定向攻击成功率 (True Attack Success Rate): {true_attack_success_rate:.2%} ")

    alphas = torch.sigmoid(patch_model.alpha_logits).detach().cpu().numpy()
    sizes = torch.sigmoid(patch_model.size_logits).detach().cpu().numpy()
    print("\n--- Learned Parameters per Keypoint ---")
    kpt_names = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist",
                 "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"]
    for i in range(17):
        status = "KEEP" if alphas[i] > 0.5 else "DROP"
        print(f"{kpt_names[i]}: Alpha={alphas[i]:.2f} ({status}), Scaled Size Factor={sizes[i]:.2f}")


if __name__ == '__main__':
    test_patch()