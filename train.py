import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FLIRKeypointDataset, collate_fn
from patch_model import PoseAwareAdversarialPatch


def train_patch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device:{device}')

    # 1. 加载白盒目标检测模型 (YOLOv5)
    print("Loading YOLOv5 model...")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom',
                          path='/mnt/d/PycharmProjects/PoseAttack/yolov5-master/runs/train/exp/weights/best.pt',  # 本地权重路径
                          force_reload=True)
    yolo_model = yolo_model.to(device)
    yolo_model.model = yolo_model.model.to(device)
    yolo_model.eval()  # eval 模式，固定 BN 和 Dropout

    # YOLO COCO 中 Person 的索引是 0
    PERSON_CLASS_INDEX = 0

    # 2. 初始化对抗贴图生成器
    patch_model = PoseAwareAdversarialPatch(device=device).to(device)
    patch_model.train()

    # 3. 优化器 (只优化贴图参数，不优化 YOLO)
    optimizer = optim.Adam(patch_model.parameters(), lr=0.005)

    # 4. 数据集加载
    train_dataset = FLIRKeypointDataset(
        img_dir='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_train',
        ann_file='/mnt/d/Dataset/FLIR_ADAS_v2_Person/images_thermal_train/person_keypoints_train_with_pose.json'
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    epochs = 20
    lambda_tv = 0.01  # 总变分(TV) Loss，保证九宫格内部平滑性
    #lambda_l1 = 0.05  # L1 Loss 控制数量(让某些骨骼点不生成贴图)
    lambda_l1 = 0.2  # L1 Loss 控制数量(让某些骨骼点不生成贴图)

    print("Starting Attack Training...")
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader)

        for images, keypoints, bboxes in pbar:
            images = images.to(device)
            keypoints = [k.to(device) for k in keypoints]
            bboxes = [b.to(device) for b in bboxes] # 把边界框发到GPU

            optimizer.zero_grad()

            # 生成贴图后的图像
            patched_images, alphas = patch_model(images, keypoints, bboxes)

            # 缩放图像到 YOLOv5 默认尺寸 (例如 640x640)
            patched_images_resized = torch.nn.functional.interpolate(patched_images, size=(640, 640), mode='bilinear')

            # 前向传播白盒 YOLOv5
            preds_raw = yolo_model.model(patched_images_resized)

            # 兼容不同 YOLOv5 版本：检查返回的是 Tuple 还是直接的 Tensor
            if isinstance(preds_raw, (tuple, list)):
                preds = preds_raw[0]
            else:
                preds = preds_raw

            # 安全校验：如果意外丢失了 Batch 维度（变成了 2D），则手动补全为 3D: [Batch, Anchors, 85]
            if preds.dim() == 2:
                preds = preds.unsqueeze(0)

            # 计算目标对象的对抗 Loss：使 Person 类的置信度最小化
            # 获取所有框对 Person 的分数：obj_conf * class_conf
            obj_conf = preds[..., 4]
            person_conf = preds[..., 5 + PERSON_CLASS_INDEX]
            score = obj_conf * person_conf

            # 对抗攻击目标：让图像中最大的人类检测分数尽可能低
            max_scores, _ = torch.max(score, dim=1)
            adv_loss = torch.mean(max_scores)

            # 数量约束 (L1 Loss on alphas) - 让部分 alpha 趋于 0，减少贴图数量
            l1_loss = torch.mean(alphas)

            # 九宫格颜色的 TV Loss (防止高频噪点，保持色块感)
            grid = patch_model.grid_texture
            tv_loss = torch.sum(torch.abs(grid[:, :, :, :-1] - grid[:, :, :, 1:])) + \
                      torch.sum(torch.abs(grid[:, :, :-1, :] - grid[:, :, 1:, :]))


            MAX_PATCHES = 6

            # 1. 提取物理允许攻击的 7 个候选点的 alphas
            allowed_indices = [0, 5, 6, 11, 12, 13, 14]
            valid_alphas = alphas[allowed_indices]

            # 2. 对这些合法的 alphas 进行降序排序
            sorted_valid_alphas, _ = torch.sort(valid_alphas, descending=True)

            # 3. 强制逼迫前 K 个点必须发光 (趋近于 1)
            loss_keep = torch.mean(1.0 - sorted_valid_alphas[:MAX_PATCHES])

            # 4. 强制逼迫 K 名之后的点必须熄灭 (趋近于 0)
            if MAX_PATCHES < len(valid_alphas):
                loss_drop = torch.mean(sorted_valid_alphas[MAX_PATCHES:])
            else:
                loss_drop = torch.tensor(0.0, device=device)

            # 5. 计算总 Loss
            # (通过系数 1.0 强行保证模型必须贴满 5 个，再去优化 YOLO 的 adv_loss)
            loss = adv_loss + 1.0 * loss_keep + 1.0 * loss_drop
            # ===================================================

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | AdvScore: {adv_loss.item():.4f}")

        # 保存每一轮的攻击补丁权重
        torch.save(patch_model.state_dict(), f"./weights/adv_patch_6_epoch_{epoch + 1}.pt")
        print(f"Epoch {epoch + 1} done. Avg Loss: {epoch_loss / len(train_loader):.4f}")


if __name__ == '__main__':
    train_patch()