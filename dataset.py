import os
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class FLIRKeypointDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.catIds = self.coco.getCatIds(catNms=['person'])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.transform = transform

        self.allowed_indices = [0, 5, 6, 11, 12, 13, 14]

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.imgIds[idx])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = torch.zeros((3, 640, 640))
            return img, torch.empty((0, 17, 3)), torch.empty((0, 4))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annIds = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)

        valid_bboxes = []
        valid_keypoints = []

        for ann in anns:
            if 'keypoints' in ann and 'bbox' in ann:
                # 提取原始 51 维数组并重塑
                kpts = torch.tensor(ann['keypoints'], dtype=torch.float32).view(17, 3)

                # 物理约束：强制把不允许攻击的部位 Visibility 置为 0
                for i in range(17):
                    if i not in self.allowed_indices:
                        kpts[i, 2] = 0.0  # visibility = 0

                if torch.sum(kpts[:, 2]) == 0:
                    continue

                valid_bboxes.append(ann['bbox'])  # [x, y, w, h]
                valid_keypoints.append(kpts)

        # 图像归一化到 [0, 1]
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # [3, H, W]

        # 将列表转换为统一的 Tensor 格式
        if len(valid_keypoints) > 0:
            keypoints_tensor = torch.stack(valid_keypoints)  # [num_persons, 17, 3]
            bboxes_tensor = torch.tensor(valid_bboxes, dtype=torch.float32)  # [num_persons, 4]
        else:
            # 如果整张图里没有一个符合条件的人，返回空的 tensor
            keypoints_tensor = torch.empty((0, 17, 3))
            bboxes_tensor = torch.empty((0, 4))

        return img, keypoints_tensor, bboxes_tensor


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    keypoints = [item[1] for item in batch]
    bboxes = [item[2] for item in batch]
    return images, keypoints, bboxes