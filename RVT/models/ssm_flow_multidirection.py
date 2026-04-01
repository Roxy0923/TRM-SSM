"""
多方向SSM2DRefiner实现 - 参考Mamba-2D / Vision Mamba
用于更强的2D空间建模能力
"""

import torch
import torch.nn as nn
from einops import rearrange
from models.layers.s5.s5_model import S5Block


class MultiDirectionSSM2DRefiner(nn.Module):
    """
    使用4个方向的SSM扫描（水平前向、水平后向、垂直前向、垂直后向）
    来更好地保留2D图像的边缘细节
    """

    def __init__(self, channels, state_channels=None, dropout=0.0):
        super().__init__()
        state_dim = state_channels or channels

        # 使用4个独立的SSM，每个处理一个方向
        self.ssm_h = S5Block(  # 水平扫描
            dim=channels,
            state_dim=state_dim,
            bidir=True,  # 包含前向和后向
            glu=False,
            ff_mult=1.0,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )
        self.ssm_v = S5Block(  # 垂直扫描
            dim=channels,
            state_dim=state_dim,
            bidir=True,
            glu=False,
            ff_mult=1.0,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )

        # 融合4个方向的输出
        self.fusion = nn.Conv2d(channels * 4, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 水平扫描 (行优先)
        seq_h = rearrange(x, "b c h w -> b (h w) c")
        state_h = self.ssm_h.s5.initial_state(batch_size=b).to(x.device)
        y_h, _ = self.ssm_h(seq_h, state_h)
        y_h = rearrange(y_h, "b (h w) c -> b c h w", h=h, w=w)

        # 2. 垂直扫描 (列优先)
        # 转置 -> 扫描 -> 转置回来
        x_transposed = rearrange(x, "b c h w -> b c w h")
        seq_v = rearrange(x_transposed, "b c w h -> b (w h) c")
        state_v = self.ssm_v.s5.initial_state(batch_size=b).to(x.device)
        y_v, _ = self.ssm_v(seq_v, state_v)
        y_v = rearrange(y_v, "b (w h) c -> b c w h", w=w, h=h)
        y_v = rearrange(y_v, "b c w h -> b c h w")

        # 3. 融合4个方向 (bidir=True 已经包含了前向+后向，所以有4个方向)
        # y_h 包含: 水平前向 + 水平后向
        # y_v 包含: 垂直前向 + 垂直后向
        # 注意：bidir=True 时输出维度会自动变成 2*channels
        # 所以这里需要调整融合逻辑

        # 简化版本：直接concat + 1x1 conv
        y_fused = torch.cat([y_h, y_v], dim=1)  # [B, 2*C, H, W] if bidir
        y_fused = self.fusion(y_fused)  # [B, C, H, W]

        return y_fused
