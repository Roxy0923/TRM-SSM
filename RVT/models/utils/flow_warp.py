import torch
import torch.nn.functional as F


def flow_warp(feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    将上一帧特征按照当前光流进行光度扭曲 (photometric warping)。

    Args:
        feature: [B, C, H, W]，上一帧的特征图
        flow:    [B, 2, H, W]，当前预测的光流 (dx, dy)

    Returns:
        warped_feature: [B, C, H, W]，对齐后的特征
    """
    if feature.shape[-2:] != flow.shape[-2:]:
        raise ValueError(
            f"feature spatial size {feature.shape[-2:]} != flow size {flow.shape[-2:]}"
        )

    b, c, h, w = feature.size()

    # 构建基础坐标网格
    yy, xx = torch.meshgrid(
        torch.arange(h, device=feature.device),
        torch.arange(w, device=feature.device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).float()  # [2, H, W]
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, 2, H, W]

    # 加上光流位移
    vgrid = grid + flow

    if torch.isnan(vgrid).any() or torch.isinf(vgrid).any():
        print("[flow_warp] vgrid contains NaN/Inf")
    
    # 提前限制光流值，防止坐标超出合理范围
    # 限制光流位移不超过图像尺寸的1.5倍（更保守的约束）
    # 使用非 inplace 操作，避免梯度计算错误
    max_flow_x = w * 1.5
    max_flow_y = h * 1.5
    vgrid_x = torch.clamp(vgrid[:, 0, :, :], -max_flow_x, max_flow_x)
    vgrid_y = torch.clamp(vgrid[:, 1, :, :], -max_flow_y, max_flow_y)
    vgrid = torch.stack([vgrid_x, vgrid_y], dim=1)

    # 归一化到 [-1, 1]（使用非 inplace 操作）
    vgrid_x_norm = 2.0 * vgrid[:, 0, :, :] / max(w - 1, 1) - 1.0
    vgrid_y_norm = 2.0 * vgrid[:, 1, :, :] / max(h - 1, 1) - 1.0
    vgrid = torch.stack([vgrid_x_norm, vgrid_y_norm], dim=1)
    vgrid = vgrid.permute(0, 2, 3, 1)  # [B, H, W, 2]

    vgrid = torch.clamp(vgrid, -1.5, 1.5)
    warped = F.grid_sample(
        feature,
        vgrid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped

