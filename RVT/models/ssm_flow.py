import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.layers.s5.s5_model import S5Block
from models.utils.flow_warp import flow_warp
from models.utils.dilated_correlation import DilatedCorrelation


class SSM2DRefiner(nn.Module):
    """
    使用 S5Block 对二维特征做线性扫描，替换原 CDC 的 Self-attention 分支。
    """

    def __init__(self, channels, state_channels=None, dropout=0.0):
        super().__init__()
        state_dim = state_channels or channels
        self.ssm = S5Block(
            dim=channels,
            state_dim=state_dim,
            bidir=False,  # ⚠️ 保持False：bidir=True有状态维度bug
            glu=False,
            ff_mult=1.0,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        seq = rearrange(x, "b c h w -> b (h w) c")
        state = self.ssm.s5.initial_state(batch_size=b).to(x.device)
        y, _ = self.ssm(seq, state)
        y = rearrange(y, "b (h w) c -> b c h w", h=h, w=w)
        return y


class CDCModule(nn.Module):
    """
    Confidence-induced Detail Completion，包含自校正 (CNN) 与自相关 (SSM) 两个分支。
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        hidden_channels=None,
        ssm_state_channels=None,
        dropout=0.0,
    ):
        super().__init__()
        hidden = hidden_channels or max(in_channels // 2, 32)
        self.corrector = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels + 1, kernel_size=3, padding=1),
        )
        self.att_branch = SSM2DRefiner(
            channels=in_channels,
            state_channels=ssm_state_channels or in_channels,
            dropout=dropout,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, coarse_feat, skip_feat, return_merged=False):
        x_up = F.interpolate(
            coarse_feat, scale_factor=2, mode="bilinear", align_corners=True
        )
        if x_up.shape[2:] != skip_feat.shape[2:]:
            x_up = F.interpolate(
                x_up, size=skip_feat.shape[2:], mode="bilinear", align_corners=True
            )

        fused = torch.cat([x_up, skip_feat], dim=1)
        delta_conf = self.corrector(fused)
        delta, conf = torch.split(delta_conf, [x_up.shape[1], 1], dim=1)

        refined = x_up + delta
        attn_feat = self.att_branch(x_up)

        conf = self.sigmoid(conf)
        merged = conf * refined + (1.0 - conf) * attn_feat

        if return_merged:
            return merged
        # 输出仍与 skip_feat 拼接，方便后续卷积融合
        return torch.cat([merged, skip_feat], dim=1)


class FlowUpsampler(nn.Module):
    """基于 CDC 的光流上采样与细节补全模块。"""

    def __init__(self, skip_channels, hidden_ratio=0.5, dropout=0.0):
        super().__init__()
        hidden_channels = max(int(skip_channels * hidden_ratio), 32)
        self.cdc = CDCModule(
            in_channels=2,
            skip_channels=skip_channels,
            hidden_channels=hidden_channels,
            ssm_state_channels=2,
            dropout=dropout,
        )

    def forward(self, flow, skip_feat):
        flow_up = F.interpolate(
            flow, size=skip_feat.shape[-2:], mode="bilinear", align_corners=True
        ) * 2.0
        return self.cdc(flow_up, skip_feat, return_merged=True)


class FlowDecoderBlock(nn.Module):
    """接收 Cost Volume + 当前特征 + Flow，预测残差光流。"""

    def __init__(self, feat_channels, corr_channels, hidden_channels):
        super().__init__()
        in_channels = feat_channels + corr_channels + 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cost_volume, feat_curr, flow):
        inp = torch.cat([cost_volume, feat_curr, flow], dim=1)
        delta = self.net(inp)
        return flow + delta

class SSMOpticalFlow(nn.Module):
    def __init__(self, backbone, stage_dims=None, use_temporal_states=False):
        super().__init__()
        self.backbone = backbone
        self.prev_features = None
        self.prev_states = None  # 保存 SSM Backbone 的状态，用于长时记忆
        self.use_temporal_states = use_temporal_states  # 🔥 控制是否使用SSM时序状态

        # --- 冻结 Backbone 参数 (关键策略) ---
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # --- 定义解码器通道数 (根据你的 Log 调整) ---
        if stage_dims is None:
            stage_dims = (64, 128, 256, 512)
        assert len(stage_dims) == 4, "stage_dims 应包含四个阶段的通道数"
        c1, c2, c3, c4 = stage_dims
        
        self.corr = DilatedCorrelation(radius=4, dilation_step=2)
        corr_channels = self.corr.out_channels

        self.flow_up3 = FlowUpsampler(skip_channels=c3)
        self.flow_up2 = FlowUpsampler(skip_channels=c2)
        self.flow_up1 = FlowUpsampler(skip_channels=c1)
        
        self.decoder3 = FlowDecoderBlock(c3, corr_channels, hidden_channels=256)
        self.decoder2 = FlowDecoderBlock(c2, corr_channels, hidden_channels=128)
        self.decoder1 = FlowDecoderBlock(c1, corr_channels, hidden_channels=64)
        
        self.flow_head = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def train(self, mode: bool = True):
        """
        重写 train，确保骨干始终保持推理模式，以保护 BN 统计与关闭 Dropout。
        """
        super().train(mode)
        if mode:
            # 仅骨干保持 eval，解码器依旧训练
            self.backbone.eval()
            for m in self.backbone.modules():
                if isinstance(
                    m,
                    (
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.Dropout,
                        nn.Dropout2d,
                        nn.Dropout3d,
                    ),
                ):
                    m.eval()
        return self

    def reset_state(self):
        """重置模型状态，包括特征缓存和 SSM 状态"""
        self.prev_features = None
        self.prev_states = None

    @staticmethod
    def _print_flow_stats(name, flow):
        # 只在训练初期或特定条件下打印，避免输出过多
        if torch.rand(1).item() > 0.99:  # 只打印 1% 的迭代
            stats = flow.abs()
            print(f"[Flow Stage] {name}: mean={stats.mean().item():.4f}, max={stats.max().item():.4f}")

    def forward(self, x, return_stages: bool = False):
        """
        前向传播，支持真实的双帧输入以消除自相关悖论。
        
        Args:
            x: 事件输入，支持两种格式：
               - [T=2, B, C, H, W]: 双帧序列（推荐）- events[0]=prev, events[1]=curr
               - [1, B, C, H, W] 或 [B, C, H, W]: 单帧（兼容旧模式）
            return_stages: 是否返回多尺度光流
        
        Returns:
            flow_full: [B, 2, H, W] 全分辨率光流
        """
        # 0. 判断输入格式并分离 prev/curr
        if x.dim() == 5 and x.shape[0] == 2:
            # ✅ 新模式：双帧输入 [T=2, B, C, H, W]
            events_prev = x[0:1]  # [1, B, C, H, W] - 前一帧
            events_curr = x[1:2]  # [1, B, C, H, W] - 当前帧
            use_real_prev = True
        else:
            # ⚠️ 兼容模式：单帧输入，使用缓存的 prev_features（可能自相关）
            events_prev = None
            events_curr = x if x.dim() == 5 else x.unsqueeze(0)  # 确保是 5D
            use_real_prev = False
        
        # 1. 填充输入到合适的尺寸
        original_size = None
        _, _, _, H, W = events_curr.shape
        target_multiple = 160  # 4 * 2^3 * 5
        pad_h = (target_multiple - H % target_multiple) % target_multiple
        pad_w = (target_multiple - W % target_multiple) % target_multiple
        
        if pad_h > 0 or pad_w > 0:
            events_curr = F.pad(events_curr, (0, pad_w, 0, pad_h, 0, 0, 0, 0, 0, 0), mode="constant", value=0)
            if events_prev is not None:
                events_prev = F.pad(events_prev, (0, pad_w, 0, pad_h, 0, 0, 0, 0, 0, 0), mode="constant", value=0)
            original_size = (H, W)
        
        # 2. 提取当前帧特征
        # 🔥 根据use_temporal_states决定是否传递历史状态
        with torch.no_grad():
            if self.use_temporal_states:
                # ✅ 启用时序：使用历史状态（充分利用SSM长时记忆）
                curr_feats, new_states = self.backbone(events_curr, prev_states=self.prev_states)
                self.prev_states = new_states  # 保存状态供下一帧使用
            else:
                # ❌ 禁用时序：每帧独立处理（兼容HREM离散帧对训练）
                curr_feats, _ = self.backbone(events_curr, prev_states=None)
        
        # 清理当前帧特征维度
        clean_curr_feats = []
        for f in curr_feats:
            if f.dim() == 5:
                f = f.squeeze(0) if f.shape[0] == 1 else f.squeeze(1)
            clean_curr_feats.append(f.detach())
        curr_c1, curr_c2, curr_c3, curr_c4 = clean_curr_feats
        
        # 3. 提取前一帧特征
        if use_real_prev and events_prev is not None:
            # ✅ 真实的前一帧：独立提取特征（不使用状态，避免污染）
            with torch.no_grad():
                prev_feats, _ = self.backbone(events_prev, prev_states=None)
            clean_prev_feats = []
            for f in prev_feats:
                if f.dim() == 5:
                    f = f.squeeze(0) if f.shape[0] == 1 else f.squeeze(1)
                clean_prev_feats.append(f.detach())
            prev_c1, prev_c2, prev_c3, prev_c4 = clean_prev_feats
        else:
            # ⚠️ 兼容模式：使用缓存特征（首帧会自相关）
            if self.prev_features is None:
                self.prev_features = [curr_c1.detach(), curr_c2.detach(), curr_c3.detach(), curr_c4.detach()]
            prev_c1, prev_c2, prev_c3, prev_c4 = self.prev_features

        batch_size = curr_c1.shape[0]
        device = curr_c1.device

        # 4. Decoder - 多尺度迭代光流估计
        # Stage 4: 初始零流
        flow_s4 = torch.zeros(batch_size, 2, curr_c4.shape[-2], curr_c4.shape[-1], device=device)

        # Stage 3
        flow_s3 = self.flow_up3(flow_s4, curr_c3)
        self._print_flow_stats("flow_s3_up", flow_s3)
        prev_c3_warp = flow_warp(prev_c3, flow_s3)
        cost3 = self.corr(curr_c3, prev_c3_warp)  # ← 现在是真实的 correlation！
        flow_s3 = self.decoder3(cost3, curr_c3, flow_s3)
        self._print_flow_stats("flow_s3_decoded", flow_s3)

        # Stage 2
        flow_s2 = self.flow_up2(flow_s3, curr_c2)
        self._print_flow_stats("flow_s2_up", flow_s2)
        prev_c2_warp = flow_warp(prev_c2, flow_s2)
        cost2 = self.corr(curr_c2, prev_c2_warp)
        flow_s2 = self.decoder2(cost2, curr_c2, flow_s2)
        self._print_flow_stats("flow_s2_decoded", flow_s2)

        # Stage 1
        flow_s1 = self.flow_up1(flow_s2, curr_c1)
        self._print_flow_stats("flow_s1_up", flow_s1)
        prev_c1_warp = flow_warp(prev_c1, flow_s1)
        cost1 = self.corr(curr_c1, prev_c1_warp)
        flow_s1 = self.decoder1(cost1, curr_c1, flow_s1)
        self._print_flow_stats("flow_s1_decoded", flow_s1)

        # 5. 裁剪回原始尺寸
        if original_size is not None:
            target_h, target_w = original_size

            def crop_to_size(flow, divisor: int):
                h = math.ceil(target_h / divisor)
                w = math.ceil(target_w / divisor)
                return flow[:, :, :h, :w]

            flow_s1 = crop_to_size(flow_s1, 4)
            flow_s2 = crop_to_size(flow_s2, 8)
            flow_s3 = crop_to_size(flow_s3, 16)

        # 6. 上采样到全分辨率
        flow_full = (
            F.interpolate(flow_s1, scale_factor=4, mode="bilinear", align_corners=True)
            * 4.0
        )

        # 7. 更新缓存（用于兼容模式）
        self.prev_features = [curr_c1.detach(), curr_c2.detach(), curr_c3.detach(), curr_c4.detach()]
        
        if return_stages:
            stage_flows = {
                "stage1": flow_s1,
                "stage2": flow_s2,
                "stage3": flow_s3,
            }
            return flow_full, stage_flows
        
        return flow_full
