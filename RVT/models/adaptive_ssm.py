"""
Adaptive SSM Modules for Frequency-Adaptive Optical Flow

This module provides SSM-based refinement layers with frequency-adaptive
step_scale modulation for event-based optical flow estimation.

Author: Frequency-Adaptive SSM Integration
"""

import math
import torch
import torch.nn as nn
from einops import rearrange

from RVT.models.layers.s5.s5_model import S5Block


class AdaptiveSSM2DRefiner(nn.Module):
    """
    Frequency-adaptive 2D SSM refiner for spatial feature refinement.

    This module applies State Space Model (SSM) processing to 2D features
    with frequency-adaptive temporal discretization via step_scale modulation.

    **Two Modes**:
    1. **Fixed Mode** (default): step_scale is a fixed hyperparameter from dataset config
       - dt1 (60Hz) → step_scale = 0.7 (high freq, fine temporal resolution)
       - dt4 (15Hz) → step_scale = 1.5 (low freq, coarse, more smoothing)

    2. **Adaptive Mode** (future): step_scale computed from input frequency score
       - Requires freq_score input in forward()

    Args:
        channels (int): Number of input/output feature channels
        state_channels (int, optional): SSM state dimension.
            Defaults to `channels` if None.
        dropout (float): Dropout probability. Default: 0.0
        step_scale (float or str, optional): Fixed step_scale value or 'adaptive'.
            - If float: Use fixed value (e.g., 0.7 for dt1, 1.5 for dt4)
            - If 'adaptive': Compute from freq_score input
            Default: 1.0 (baseline, no frequency adaptation)
        step_scale_range (tuple, optional): (min, max) range for adaptive mode.
            Only used if step_scale='adaptive'. Default: (0.7, 1.5)

    Input (Fixed Mode):
        x: [B, C, H, W] - 2D feature map

    Input (Adaptive Mode):
        x: [B, C, H, W] - 2D feature map
        freq_score: [B, 1] - Frequency score in [0, 1]

    Output:
        y: [B, C, H, W] - Refined feature map
    """

    def __init__(
        self,
        channels,
        state_channels=None,
        dropout=0.0,
        step_scale=1.0,  # Fixed step_scale from dataset config
        anti_alias: bool = False,
        blur_kernel_size: int = 3,
        blur_sigma: float = 1.0,
        h2_reg_weight: float = 0.0,
        h2_mode: str = "proxy",  # "proxy" (fast) or "freq" (frequency integral)
        h2_omega_min: float = 1.0,
        h2_omega_max: float = 100.0,
        h2_num_points: int = 64,
    ):
        super().__init__()
        self.channels = channels
        self.state_channels = state_channels or channels
        self.step_scale_value = step_scale
        self.h2_reg_weight = h2_reg_weight
        self.h2_mode = h2_mode
        self.h2_omega_min = h2_omega_min
        self.h2_omega_max = h2_omega_max
        self.h2_num_points = h2_num_points
        self.last_h2_reg = torch.tensor(0.0)

        # S5Block: LayerNorm → S5 → FFN
        # 参数设置与论文一致：
        # - glu=True: 使用GEGLU激活（论文默认）
        # - ff_dropout=0.0, attn_dropout=0.0: 论文默认（dropout参数仅用于其他模块）
        self.s5_block = S5Block(
            dim=channels,
            state_dim=self.state_channels,
            bidir=False,  # Avoid bidirectional bug
            bandlimit=0.5,  # Static bandlimit (论文推荐)
            glu=True,  # 使用GEGLU激活（论文默认）
            ff_mult=1.0,  # No expansion in FFN
            ff_dropout=0.0,  # 论文默认
            attn_dropout=0.0,  # 论文默认
        )

        # Spatial anti-alias (disabled by default; paper focuses on temporal freq bandlimit)
        self.anti_alias = anti_alias
        if self.anti_alias:
            k = blur_kernel_size
            assert k % 2 == 1, "blur_kernel_size must be odd"
            radius = k // 2
            coords = torch.arange(-radius, radius + 1)
            grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
            kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * blur_sigma**2))
            kernel = kernel / kernel.sum()
            self.register_buffer(
                "blur_kernel",
                kernel.view(1, 1, k, k).repeat(self.channels, 1, 1, 1),
            )
            self.blur = torch.nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=k,
                padding=radius,
                groups=self.channels,
                bias=False,
            )
            with torch.no_grad():
                self.blur.weight.copy_(self.blur_kernel)
            self.blur.requires_grad_(False)

    def forward(self, x, step_scale=None, return_reg: bool = False):
        """
        Apply SSM refinement with step_scale.

        Args:
            x: [B, C, H, W] - Input feature map
            step_scale (float or torch.Tensor, optional): Dynamic step_scale value.
                If None, use self.step_scale_value (fixed mode).
                If provided, use this value (allows per-batch or per-sample scaling).

        Returns:
            y: [B, C, H, W] - Refined feature map
        """
        b, c, h, w = x.shape

        if self.anti_alias:
            # Optional spatial low-pass (off by default)
            x = self.blur(x)

        # Use dynamic step_scale if provided, otherwise use fixed value
        if step_scale is None:
            step_scale = self.step_scale_value

        # Flatten 2D features to 1D sequence (row-major order)
        # [B, C, H, W] → [B, (H*W), C]
        seq = rearrange(x, "b c h w -> b (h w) c")

        # Initialize SSM state (zero state for stateless processing)
        state = self.s5_block.s5.initial_state(batch_size=b).to(x.device)

        # Use S5Block's forward method (it handles LayerNorm, S5, and FFN internally)
        # Note: S5Block expects (x, states) and returns (y, new_state)
        # We need to pass step_scale to the underlying S5 module
        # For now, we'll use the S5Block as-is and handle step_scale separately if needed
        # Ensure step_scale is applied to S5 for correct discretization at new freq
        y, new_state = self.s5_block(seq, state, step_scale=step_scale)

        # Reshape back to 2D: [B, (H*W), C] → [B, C, H, W]
        y = rearrange(y, "b (h w) c -> b c h w", h=h, w=w)

        # Cache H2-like regularization (simple Frobenius proxy on B,C)
        if self.h2_reg_weight > 0:
            if self.h2_mode == "freq":
                h2_raw = self._compute_h2_freq()
                self.last_h2_reg = h2_raw * self.h2_reg_weight
            else:
                h2_raw = self._compute_h2_proxy()
                self.last_h2_reg = h2_raw * self.h2_reg_weight
            # Debug: Print first few forward passes to verify H2 computation
            if not hasattr(self, '_h2_debug_count'):
                self._h2_debug_count = 0
            if self._h2_debug_count < 3:
                import sys
                raw_val = h2_raw.item() if torch.is_tensor(h2_raw) else h2_raw
                reg_val = self.last_h2_reg.item()
                print(
                    f"[AdaptiveSSM2DRefiner] Forward #{self._h2_debug_count}: "
                    f"h2_reg_weight={self.h2_reg_weight}, h2_raw={raw_val:.3e}, "
                    f"last_h2_reg={reg_val:.3e}, step_scale={step_scale}",
                    flush=True,
                )
                self._h2_debug_count += 1
        else:
            self.last_h2_reg = torch.tensor(0.0, device=y.device)

        if return_reg:
            return y, self.last_h2_reg
        return y

    def extra_repr(self):
        """String representation for debugging."""
        return (
            f"channels={self.channels}, state_channels={self.state_channels}, "
            f"step_scale={self.step_scale_value}, anti_alias={self.anti_alias}, "
            f"h2_reg_weight={self.h2_reg_weight}, h2_mode={self.h2_mode}"
        )

    def _safe_view_as_complex(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Safely convert a real tensor with shape [..., 2] to complex.
        Handles storage_offset issues in DataParallel/vmap contexts.
        IMPORTANT: Does NOT detach - preserves gradients for H2 regularization.
        """
        t = tensor
        # Only clone if necessary (non-contiguous or misaligned storage_offset)
        if not t.is_contiguous() or t.storage_offset() % 2 != 0:
            t = t.contiguous().clone()  # clone preserves gradients, detach does not
        else:
            t = t.contiguous()  # contiguous() alone preserves gradients
        return torch.view_as_complex(t)

    def _compute_h2_proxy(self) -> torch.Tensor:
        """
        Lightweight proxy for H2 norm: ||B||_F^2 + ||C||_F^2 on the SSM.
        This avoids solving a Lyapunov equation and keeps training simple.
        """
        seq = self.s5_block.s5.seq

        # B is already real-valued, just compute norm
        B_norm = seq.B.pow(2).sum()

        # For C, use safe complex conversion
        C = self._safe_view_as_complex(seq.C)
        C_norm = C.abs().pow(2).sum()

        reg = B_norm + C_norm
        return reg

    def _compute_h2_freq(self) -> torch.Tensor:
        """
        Frequency-domain approximation of H2 norm (integral over [omega_min, omega_max]).
        Uses a simple rectangle rule; more costly than proxy but closer to paper.
        """
        seq = self.s5_block.s5.seq

        # Use safe complex conversion for all parameters
        A = self._safe_view_as_complex(seq.Lambda)  # [P]
        B = self._safe_view_as_complex(seq.B)       # [P, H]
        C = self._safe_view_as_complex(seq.C)       # [H, P]

        omegas = torch.linspace(
            self.h2_omega_min,
            self.h2_omega_max,
            self.h2_num_points,
            device=A.device,
        )
        h2_norm_sq = 0.0
        for omega in omegas:
            resolvent = 1.0 / (1j * omega - A)       # [P]
            G_omega = C @ (resolvent[:, None] * B)   # [H, H]
            h2_norm_sq = h2_norm_sq + (G_omega.abs() ** 2).sum()
        d_omega = (self.h2_omega_max - self.h2_omega_min) / self.h2_num_points
        h2_norm = torch.sqrt(h2_norm_sq * d_omega / math.pi)
        return h2_norm


class MultiDirectionAdaptiveSSM2DRefiner(nn.Module):
    """
    Multi-directional frequency-adaptive SSM refiner.

    This advanced variant applies SSM scanning in multiple directions
    (horizontal, vertical, and optionally diagonal) to capture richer
    spatial dependencies.

    Note: This is an advanced module for Phase 4 (optional optimization).
          Not used in MVP to keep initial implementation simple.

    Args:
        channels (int): Number of input/output channels
        state_channels (int, optional): SSM state dimension
        num_directions (int): Number of scanning directions (2 or 4)
        dropout (float): Dropout probability
        step_scale_range (tuple): Range for adaptive step_scale

    Input:
        x: [B, C, H, W] - 2D feature map
        freq_score: [B, 1] - Frequency score

    Output:
        y: [B, C, H, W] - Refined feature map
    """

    def __init__(
        self,
        channels,
        state_channels=None,
        num_directions=4,
        dropout=0.0,
        step_scale_range=(0.7, 1.5),
    ):
        super().__init__()
        self.channels = channels
        self.num_directions = num_directions

        # Create separate S5Blocks for each scanning direction
        # Directions: horizontal (LR, RL), vertical (TB, BT)
        self.s5_blocks = nn.ModuleList([
            S5Block(
                dim=channels,
                state_dim=state_channels or channels,
                bidir=False,
                bandlimit=0.5,
                glu=False,
                ff_mult=1.0,
                ff_dropout=dropout,
                attn_dropout=dropout,
            )
            for _ in range(num_directions)
        ])

        # Fusion layer: Combine outputs from all directions
        # Input: channels * num_directions → Output: channels
        self.fusion = nn.Conv2d(
            channels * num_directions, channels, kernel_size=1, bias=False
        )

        self.step_range = step_scale_range

    def forward(self, x, freq_score):
        """
        Apply multi-directional SSM refinement.

        Args:
            x: [B, C, H, W] - Input features
            freq_score: [B, 1] - Frequency score

        Returns:
            y: [B, C, H, W] - Refined features
        """
        b, c, h, w = x.shape
        outputs = []

        # Horizontal scanning (left-to-right and right-to-left)
        if self.num_directions >= 2:
            # Left-to-right
            seq_h = rearrange(x, "b c h w -> (b h) w c")
            state_h = self.s5_blocks[0].s5.initial_state(batch_size=b * h).to(x.device)
            y_h, _ = self.s5_blocks[0](seq_h, state_h)
            y_h = rearrange(y_h, "(b h) w c -> b c h w", b=b, h=h)
            outputs.append(y_h)

            # Right-to-left (flip horizontally)
            seq_h_rev = rearrange(x.flip(dims=[3]), "b c h w -> (b h) w c")
            state_h_rev = self.s5_blocks[1].s5.initial_state(batch_size=b * h).to(x.device)
            y_h_rev, _ = self.s5_blocks[1](seq_h_rev, state_h_rev)
            y_h_rev = rearrange(y_h_rev, "(b h) w c -> b c h w", b=b, h=h)
            y_h_rev = y_h_rev.flip(dims=[3])  # Flip back
            outputs.append(y_h_rev)

        # Vertical scanning (top-to-bottom and bottom-to-top)
        if self.num_directions >= 4:
            # Top-to-bottom
            seq_v = rearrange(x, "b c h w -> (b w) h c")
            state_v = self.s5_blocks[2].s5.initial_state(batch_size=b * w).to(x.device)
            y_v, _ = self.s5_blocks[2](seq_v, state_v)
            y_v = rearrange(y_v, "(b w) h c -> b c h w", b=b, w=w)
            outputs.append(y_v)

            # Bottom-to-top (flip vertically)
            seq_v_rev = rearrange(x.flip(dims=[2]), "b c h w -> (b w) h c")
            state_v_rev = self.s5_blocks[3].s5.initial_state(batch_size=b * w).to(x.device)
            y_v_rev, _ = self.s5_blocks[3](seq_v_rev, state_v_rev)
            y_v_rev = rearrange(y_v_rev, "(b w) h c -> b c h w", b=b, w=w)
            y_v_rev = y_v_rev.flip(dims=[2])  # Flip back
            outputs.append(y_v_rev)

        # Fuse all directional outputs
        fused = torch.cat(outputs, dim=1)  # [B, C*num_directions, H, W]
        y = self.fusion(fused)  # [B, C, H, W]

        return y

    def extra_repr(self):
        """String representation for debugging."""
        return (
            f"channels={self.channels}, num_directions={self.num_directions}, "
            f"step_scale_range={self.step_range}"
        )
