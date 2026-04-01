import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedCorrelation(nn.Module):
    """
    局部膨胀特征相关性 (Cost Volume) 构建模块。
    参考 EEMFlow 公式 (1)(2)：在中心区域密集采样，外围稀疏采样。
    """

    def __init__(self, radius: int = 4, dilation_step: int = 2):
        super().__init__()
        assert radius >= 1
        assert dilation_step >= 1

        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # 中心区域 (|dx|<=1, |dy|<=1) 保留全部
                if abs(dx) <= 1 and abs(dy) <= 1:
                    offsets.append((dx, dy))
                    continue
                # 外围根据 dilation_step 稀疏采样
                if (abs(dx) % dilation_step == 0) and (abs(dy) % dilation_step == 0):
                    offsets.append((dx, dy))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.out_channels = len(offsets)

    def forward(self, feat_curr: torch.Tensor, feat_prev_warped: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_curr:        [B, C, H, W] 当前帧特征
            feat_prev_warped: [B, C, H, W] Warp 对齐后的上一帧特征
        Returns:
            cost_volume: [B, num_offsets, H, W]
        """
        if feat_curr.shape != feat_prev_warped.shape:
            raise ValueError("feature shapes must match for correlation")

        b, c, h, w = feat_curr.shape
        cost_maps = []
        feat_curr_norm = F.normalize(feat_curr, p=2, dim=1)
        feat_prev_norm = F.normalize(feat_prev_warped, p=2, dim=1)

        for dx, dy in self.offsets.tolist():
            # 对上一帧特征进行平移 (pad + crop)
            pad_l = max(0, -dx)
            pad_r = max(0, dx)
            pad_t = max(0, -dy)
            pad_b = max(0, dy)
            shifted = F.pad(feat_prev_norm, (pad_l, pad_r, pad_t, pad_b))
            y_start = pad_t + dy
            y_end = y_start + h
            x_start = pad_l + dx
            x_end = x_start + w
            shifted = shifted[:, :, y_start:y_end, x_start:x_end]
            if shifted.shape[-2:] != (h, w):
                shifted = F.pad(
                    shifted,
                    (
                        0,
                        max(0, w - shifted.shape[-1]),
                        0,
                        max(0, h - shifted.shape[-2]),
                    ),
                )
                shifted = shifted[:, :, :h, :w]

            corr = (feat_curr_norm * shifted).sum(dim=1)
            cost_maps.append(corr)

        cost_volume = torch.stack(cost_maps, dim=1)
        return cost_volume

