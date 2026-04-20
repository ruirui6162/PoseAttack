import os
import torch
import torch.nn as nn
import numpy as np
import cv2


class PoseAwareAdversarialPatch(nn.Module):
    def __init__(self, num_keypoints=17, device='cpu'):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.device = device

        # 恢复旧版的参数名字，形状为 [17, 1, 3, 3]
        self.grid_texture = nn.Parameter(torch.randn(num_keypoints, 1, 3, 3, device=device))
        self.size_logits = nn.Parameter(torch.zeros(num_keypoints, device=device))
        self.alpha_logits = nn.Parameter(torch.ones(num_keypoints, device=device))

        # 尺寸范围约束
        size_ranges_tensor = torch.tensor([
            [0.10, 0.20], [0.10, 0.20], [0.10, 0.20], [0.10, 0.20], [0.10, 0.20],
            [0.15, 0.30], [0.15, 0.30], [0.10, 0.25], [0.10, 0.25], [0.10, 0.20], [0.10, 0.20],
            [0.20, 0.40], [0.20, 0.40], [0.15, 0.30], [0.15, 0.30], [0.10, 0.25], [0.10, 0.25]
        ], dtype=torch.float32, device=device)
        self.register_buffer('size_ranges', size_ranges_tensor)


def extract_blueprint(weight_path="./weights/adv_patch_epoch_20.pt", output_dir="./physical_blueprint"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型与权重
    print(f"Loading trained weights from: {weight_path}")
    patch_model = PoseAwareAdversarialPatch(device='cpu')
    patch_model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    patch_model.eval()

    # 2. 计算物理参数
    alphas = torch.sigmoid(patch_model.alpha_logits).detach().numpy()
    scales = torch.sigmoid(patch_model.size_logits).detach()
    actual_scales = (patch_model.size_ranges[:, 0] + scales * (
                patch_model.size_ranges[:, 1] - patch_model.size_ranges[:, 0])).numpy()

    # 【旧版逻辑】直接对 3x3 矩阵进行 Sigmoid 黑白二值化
    bw_texture = torch.sigmoid(patch_model.grid_texture * 10.0).detach().numpy()  # [17, 1, 3, 3]

    kpt_names = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist",
                 "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"]

    # 3. 生成施工单与高清图片
    print("\n" + "=" * 50)
    print(" 🔥 物理对抗补丁 制作施工单 (非对称版) 🔥")
    print("=" * 50)

    keep_count = 0
    for i in range(17):
        if alphas[i] > 0.5:
            keep_count += 1
            name = kpt_names[i]
            size_ratio = actual_scales[i]

            # 提取 3x3 矩阵并转为 OpenCV 图像格式 (0-255)
            patch_3x3 = bw_texture[i, 0]  # [3, 3]
            patch_img = (patch_3x3 * 255).astype(np.uint8)
            patch_highres = cv2.resize(patch_img, (300, 300), interpolation=cv2.INTER_NEAREST)

            # 保存图片
            img_filename = f"Patch_{name}_SizeRatio_{size_ratio:.2f}.png"
            cv2.imwrite(os.path.join(output_dir, img_filename), patch_highres)

            # 打印物理参数
            print(f"\n🎯 目标部位: {name}")
            print(f"   - Alpha (置信度): {alphas[i]:.2f} (坚决保留)")
            print(f"   - 相对人体宽度比例: {size_ratio:.1%} (例如: 若人宽40cm，此贴图边长应做 {40 * size_ratio:.1f} cm)")
            print(f"   - 高清图纸已保存至: {img_filename}")
            print(f"   - 九宫格矩阵 (1=发热/白, 0=绝热/黑):")
            print(f"     [ {int(patch_3x3[0][0])}  {int(patch_3x3[0][1])}  {int(patch_3x3[0][2])} ]")
            print(f"     [ {int(patch_3x3[1][0])}  {int(patch_3x3[1][1])}  {int(patch_3x3[1][2])} ]")
            print(f"     [ {int(patch_3x3[2][0])}  {int(patch_3x3[2][1])}  {int(patch_3x3[2][2])} ]")

    print("\n" + "=" * 50)
    print(f"🎉 施工单提取完毕！总共需要制作 {keep_count} 块补丁。")
    print("请查看 physical_blueprint 文件夹获取高清图纸。")


if __name__ == '__main__':
    extract_blueprint(weight_path="./weights/adv_patch_epoch_20.pt")