import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseAwareAdversarialPatch(nn.Module):
    def __init__(self, num_keypoints=17, device='cuda'):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.device = device

        # 1. 矩形九宫格纹理 (3x3 grid)
        #self.grid_texture = nn.Parameter(torch.rand(num_keypoints, 3, 3, 3, device=device))
        self.grid_texture = nn.Parameter(torch.randn(num_keypoints, 1, 3, 3, device=device))

        self.size_logits = nn.Parameter(torch.zeros(num_keypoints, device=device))
        self.alpha_logits = nn.Parameter(torch.ones(num_keypoints, device=device))

        size_ranges_tensor = torch.tensor([
            [0.10, 0.20], [0.10, 0.20], [0.10, 0.20], [0.10, 0.20], [0.10, 0.20],  # 面部特征稍微小一点
            [0.15, 0.30], [0.15, 0.30], [0.10, 0.25], [0.10, 0.25], [0.10, 0.20], [0.10, 0.20],  # 手臂稍微大点
            [0.20, 0.40], [0.20, 0.40], [0.15, 0.30], [0.15, 0.30], [0.10, 0.25], [0.10, 0.25]  # 躯干和腿部可以更大
        ], dtype=torch.float32, device=device)
        self.register_buffer('size_ranges', size_ranges_tensor)

    def forward(self, images, targets_keypoints, targets_bboxes):
        """
        新增 targets_bboxes: list of tensors, 每个 tensor [num_persons, 4] -> (x, y, w, h)
        """
        bw_texture = torch.sigmoid(self.grid_texture * 10.0)
        bw_texture_3c = bw_texture.expand(-1, 3, -1, -1) # 形状变为 [17, 3, 3, 3]

        B, C, H, W = images.shape
        #base_patches = F.interpolate(torch.sigmoid(self.grid_texture), size=(30, 30), mode='nearest')
        base_patches = F.interpolate(bw_texture_3c, size=(30, 30), mode='nearest')

        scales = torch.sigmoid(self.size_logits)
        actual_scales = self.size_ranges[:, 0] + scales * (self.size_ranges[:, 1] - self.size_ranges[:, 0])
        alphas = torch.sigmoid(self.alpha_logits)

        patched_batch = []

        for b in range(B):
            current_img = images[b]
            kpts = targets_keypoints[b]
            bboxes = targets_bboxes[b]  # [num_persons, 4]

            for p in range(kpts.shape[0]):
                # 获取当前这个人的边框宽度 (w)
                person_w = bboxes[p, 2]

                # 防御性编程：如果由于某些原因人体宽为0，跳过
                if person_w <= 1.0: continue

                for k in range(self.num_keypoints):
                    x, y, v = kpts[p, k]

                    if v == 0: continue

                    s_factor = actual_scales[k]
                    alpha = alphas[k]

                    # 1. 计算贴图在当前人身上的绝对像素大小
                    patch_pixel_size = s_factor * person_w
                    # 设置一个下限，防止远处的人贴图小于3个像素，导致STN渲染崩溃或无意义
                    patch_pixel_size = torch.clamp(patch_pixel_size, min=3.0)

                    # 2. 将像素大小转为归一化比例 (占大图的比例)
                    norm_target_w = patch_pixel_size / W
                    norm_target_h = patch_pixel_size / H

                    # 3. 仿射矩阵所需的逆缩放
                    inv_s_x = 1.0 / norm_target_w
                    inv_s_y = 1.0 / norm_target_h

                    # 4. 坐标归一化到 [-1, 1]
                    norm_x = (x / W) * 2 - 1
                    norm_y = (y / H) * 2 - 1

                    tx = -norm_x * inv_s_x
                    ty = -norm_y * inv_s_y

                    zero = torch.zeros_like(inv_s_x)
                    row1 = torch.stack([inv_s_x, zero, tx])
                    row2 = torch.stack([zero, inv_s_y, ty])
                    theta = torch.stack([row1, row2]).unsqueeze(0)

                    grid = F.affine_grid(theta, size=(1, C, H, W), align_corners=False)
                    patch_warped = F.grid_sample(base_patches[k].unsqueeze(0), grid, align_corners=False,
                                                 padding_mode='zeros')

                    mask_base = torch.ones((1, 1, 30, 30), device=self.device)
                    mask_warped = F.grid_sample(mask_base, grid, align_corners=False, padding_mode='zeros')

                    current_img = current_img * (1 - mask_warped[0] * alpha) + patch_warped[0] * mask_warped[0] * alpha

            patched_batch.append(current_img)

        return torch.stack(patched_batch), alphas