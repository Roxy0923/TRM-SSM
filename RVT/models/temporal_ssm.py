"""
Temporal SSM Module for Event Camera Optical Flow

This module provides temporal modeling over event bins (e.g., 15 bins)
using State Space Models. The step_scale parameter here has direct
physical meaning - it corresponds to the temporal resolution of the data.

step_scale Calculation (based on S5 paper):
    step_scale = reference_frequency / actual_frequency

    Example (using dt1 as reference, step_scale=1.0):
    - dt1 (60Hz): step_scale = 60/60 = 1.0 (reference)
    - dt4 (15Hz): step_scale = 60/15 = 4.0

The key insight: lower frequency data has larger time intervals between
samples, so SSM needs larger step_scale to account for this.

Author: SSM Event Camera Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from einops import rearrange

from RVT.models.layers.s5.s5_model import S5Block


# ============ Frequency Constants ============
FREQ_DT1 = 60.0  # Hz (high frequency)
FREQ_DT4 = 15.0  # Hz (low frequency)
FREQ_REFERENCE = 60.0  # Reference frequency (dt1 as baseline)


def compute_step_scale(data_frequency: float, reference_frequency: float = FREQ_REFERENCE) -> float:
    """
    Compute step_scale based on data frequency ratio.

    According to S5 paper, when data frequency changes, the discretization
    step should scale proportionally to maintain correct temporal dynamics.

    Args:
        data_frequency: Actual data sampling frequency in Hz
        reference_frequency: Reference frequency (default: 60Hz = dt1)

    Returns:
        step_scale: Scaling factor for SSM discretization
            - 1.0 for reference frequency
            - >1.0 for lower frequency (larger time steps)
            - <1.0 for higher frequency (smaller time steps)

    Examples:
        >>> compute_step_scale(60.0)  # dt1 (60Hz)
        1.0
        >>> compute_step_scale(15.0)  # dt4 (15Hz)
        4.0
        >>> compute_step_scale(30.0)  # 30Hz
        2.0
        >>> compute_step_scale(120.0)  # 120Hz
        0.5
    """
    if data_frequency <= 0:
        raise ValueError(f"data_frequency must be positive, got {data_frequency}")
    return reference_frequency / data_frequency


# Pre-computed step_scale values for convenience
STEP_SCALE_DT1 = compute_step_scale(FREQ_DT1)  # 1.0
STEP_SCALE_DT4 = compute_step_scale(FREQ_DT4)  # 4.0


class TemporalSSMPatchwise(nn.Module):
    """
    Memory-efficient Temporal SSM using Patchify strategy.

    Instead of processing each pixel independently (H*W sequences),
    this version:
    1. Divides image into patches (reduces spatial resolution)
    2. Projects patches to d_model dimensions (preserves information)
    3. Applies temporal SSM on patch tokens
    4. Unpatchifies back to original resolution

    This reduces sequence count by patch_size^2 while preserving information.

    Args:
        in_channels (int): Number of input channels (e.g., 15 bins or 1)
        out_channels (int): Output feature channels
        patch_size (int): Size of patches (e.g., 8 means 8x8 patches)
        d_model (int): Internal feature dimension for SSM
        state_dim (int): SSM hidden state dimension
        dropout (float): Dropout probability

    Example:
        For H=480, W=640, patch_size=8:
        - N_tokens = 60 * 80 = 4,800
        - With B=32: 32 * 4,800 = 153,600 sequences (manageable)
        - Compare to per-pixel: 32 * 480 * 640 = 9.8M sequences (OOM)
    """

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 16,
        patch_size: int = 8,
        d_model: int = 64,
        state_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels  # T (num_bins) or 1
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.state_dim = state_dim
        # Debug counters for bandlimit sanity (non-vmap only)
        self._bl_debug_cnt = 0
        self._bl_debug_max = 5

        # Patch embedding: project each patch to d_model
        # Input: [B, T, H, W] -> patches: [B, T, H/ps, W/ps, ps*ps]
        # Then linear: [B, T, N_tokens, ps*ps] -> [B, T, N_tokens, d_model]
        self.patch_embed = nn.Linear(patch_size * patch_size, d_model)

        # Temporal S5 Block: processes sequence of length T
        self.temporal_s5 = S5Block(
            dim=d_model,
            state_dim=state_dim,
            bidir=False,  # Causal: only look at past bins
            bandlimit=1.0,  # Increased from 0.5 to reduce frequency masking for dt4
            glu=True,
            ff_mult=1.0,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )

        # Output projection: d_model -> out_channels * patch_size^2
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_channels * patch_size * patch_size),
        )

    def forward(self, x: torch.Tensor, step_scale: float = 1.0) -> torch.Tensor:
        """
        Forward pass with patchwise temporal SSM.

        Args:
            x: [B, T, H, W] - Event tensor
                B = batch size
                T = num_bins (e.g., 15)
                H, W = spatial dimensions (must be divisible by patch_size)
            step_scale: float - Temporal step scale
                - 1.0 for dt1 (60Hz, reference)
                - 4.0 for dt4 (15Hz)

        Returns:
            y: [B, out_channels, H, W] - Temporally encoded features
        """
        b, t, h, w = x.shape
        ps = self.patch_size

        assert h % ps == 0 and w % ps == 0, \
            f"H={h}, W={w} must be divisible by patch_size={ps}"

        h_patches = h // ps
        w_patches = w // ps
        n_tokens = h_patches * w_patches

        # Step 1: Patchify
        # [B, T, H, W] -> [B, T, H/ps, W/ps, ps, ps] -> [B, T, N_tokens, ps*ps]
        x_patches = rearrange(
            x, 'b t (h ph) (w pw) -> b t (h w) (ph pw)',
            ph=ps, pw=ps, h=h_patches, w=w_patches
        )
        # x_patches: [B, T, N_tokens, ps*ps]

        # Step 2: Patch embedding
        # [B, T, N_tokens, ps*ps] -> [B, T, N_tokens, d_model]
        x_embed = self.patch_embed(x_patches)

        # Step 3: Reshape for temporal SSM
        # [B, T, N_tokens, d_model] -> [(B * N_tokens), T, d_model]
        x_seq = rearrange(x_embed, 'b t n d -> (b n) t d')

        # Step 4: Initialize SSM state
        batch_seq = b * n_tokens
        state = self.temporal_s5.s5.initial_state(batch_seq).to(x.device)

        # Debug: compute bandlimit mask stats once per call (only when step_scale is scalar to avoid vmap issues)
        if self._bl_debug_cnt < self._bl_debug_max and (not torch.is_tensor(step_scale)):
            seq = self.temporal_s5.s5.seq  # S5SSM
            if getattr(seq, "bandlimit", None) is not None:
                with torch.no_grad():
                    step = step_scale * torch.exp(seq.log_step)
                    freqs = step * seq.Lambda[:, 1].abs() / (2 * math.pi)
                    mask = (freqs < seq.bandlimit * 0.5).float()
                    fmax = freqs.reshape(-1).max().float().cpu().item()
                    fmean = freqs.reshape(-1).mean().float().cpu().item()
                    mmean = mask.reshape(-1).mean().float().cpu().item()
                    log_line = (f"[TemporalSSM][debug {self._bl_debug_cnt}] "
                                f"step_scale={float(step_scale):.4f} "
                                f"freqs_max={fmax:.4f} freqs_mean={fmean:.4f} "
                                f"mask_mean={mmean:.4f}")
                    print(log_line, flush=True)
                    dbg_path = os.environ.get("MASK_DEBUG_FILE", None)
                    if dbg_path:
                        try:
                            with open(dbg_path, "a") as f:
                                f.write(log_line + "\n")
                        except Exception:
                            pass
            self._bl_debug_cnt += 1

        # Step 5: Apply temporal SSM with frequency-adaptive step_scale
        x_out, _ = self.temporal_s5(x_seq, state, step_scale=step_scale)
        # x_out: [(B * N_tokens), T, d_model]

        # Step 6: Take last time step (final temporal state)
        x_last = x_out[:, -1, :]  # [(B * N_tokens), d_model]

        # Step 7: Output projection
        # [(B * N_tokens), d_model] -> [(B * N_tokens), out_channels * ps * ps]
        x_proj = self.output_proj(x_last)

        # Step 8: Reshape and unpatchify
        # [(B * N_tokens), out_channels * ps * ps] -> [B, N_tokens, out_channels, ps, ps]
        x_reshape = rearrange(
            x_proj, '(b n) (c ph pw) -> b n c ph pw',
            b=b, n=n_tokens, c=self.out_channels, ph=ps, pw=ps
        )

        # Step 9: Unpatchify to original resolution
        # [B, N_tokens, out_channels, ps, ps] -> [B, out_channels, H, W]
        y = rearrange(
            x_reshape, 'b (h w) c ph pw -> b c (h ph) (w pw)',
            h=h_patches, w=w_patches
        )

        return y

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'patch_size={self.patch_size}, d_model={self.d_model}, '
            f'state_dim={self.state_dim}'
        )


class TemporalSSM(nn.Module):
    """
    Temporal SSM for modeling dependencies across event bins.

    WARNING: This version processes each pixel independently, which can cause
    memory issues for large images. Use TemporalSSMPatchwise instead.

    Processes the temporal dimension (T bins) of event representations using
    S5 State Space Model. Each spatial location (h, w) is treated as having
    an independent temporal sequence of length T.

    The step_scale parameter directly corresponds to physical time:
        - dt1 (60Hz): step_scale = 1.0 (reference)
        - dt4 (15Hz): step_scale = 4.0 (4x larger time intervals)

    Args:
        in_channels (int): Number of temporal bins (e.g., 15)
        out_channels (int): Output feature channels (e.g., 16)
        state_dim (int): SSM hidden state dimension. Default: 16
        dropout (float): Dropout probability. Default: 0.0
        aggregation (str): How to aggregate temporal dimension after SSM.
            - 'last': Use last time step (default, captures final state)
            - 'mean': Average over all time steps
            - 'max': Max pooling over time

    Input:
        x: [B, T, H, W] - Event tensor with T temporal bins
        step_scale: float - Temporal discretization scale

    Output:
        y: [B, out_channels, H, W] - Temporally encoded features
    """

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 16,
        state_dim: int = 16,
        dropout: float = 0.0,
        aggregation: str = 'last'
    ):
        super().__init__()
        self.in_channels = in_channels  # T (num_bins)
        self.out_channels = out_channels
        self.state_dim = state_dim
        self.aggregation = aggregation

        # Input projection: project each spatial location's value to feature dim
        # This creates a learnable embedding for each bin value
        self.input_proj = nn.Linear(1, out_channels)

        # Temporal S5 Block: processes sequence of length T
        self.temporal_s5 = S5Block(
            dim=out_channels,
            state_dim=state_dim,
            bidir=False,  # Causal: only look at past bins
            bandlimit=0.5,
            glu=True,
            ff_mult=1.0,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )

        # Output projection (optional refinement)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, step_scale: float = 1.0) -> torch.Tensor:
        """
        Forward pass with temporal SSM.

        Args:
            x: [B, T, H, W] - Event tensor
                B = batch size
                T = num_bins (e.g., 15)
                H, W = spatial dimensions
            step_scale: float - Temporal step scale
                - 1.0 for dt1 (60Hz, reference)
                - 4.0 for dt4 (15Hz)

        Returns:
            y: [B, out_channels, H, W] - Temporally encoded features
        """
        b, t, h, w = x.shape

        # Step 1: Reshape to process each spatial location independently
        # [B, T, H, W] -> [B*H*W, T, 1]
        x_seq = rearrange(x, 'b t h w -> (b h w) t 1')

        # Step 2: Project each time step to feature dimension
        # [B*H*W, T, 1] -> [B*H*W, T, C]
        x_seq = self.input_proj(x_seq)

        # Step 3: Initialize SSM state
        batch_seq = b * h * w
        state = self.temporal_s5.s5.initial_state(batch_seq).to(x.device)

        # Step 4: Apply temporal SSM with frequency-adaptive step_scale
        # step_scale controls the discretization: larger = coarser temporal resolution
        x_out, _ = self.temporal_s5(x_seq, state, step_scale=step_scale)
        # x_out: [B*H*W, T, C]

        # Step 5: Temporal aggregation
        if self.aggregation == 'last':
            x_agg = x_out[:, -1, :]  # [B*H*W, C] - final state
        elif self.aggregation == 'mean':
            x_agg = x_out.mean(dim=1)  # [B*H*W, C]
        elif self.aggregation == 'max':
            x_agg = x_out.max(dim=1)[0]  # [B*H*W, C]
        else:
            x_agg = x_out[:, -1, :]

        # Step 6: Output projection
        x_agg = self.output_proj(x_agg)

        # Step 7: Reshape back to spatial format
        # [B*H*W, C] -> [B, C, H, W]
        y = rearrange(x_agg, '(b h w) c -> b c h w', b=b, h=h, w=w)

        return y

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'state_dim={self.state_dim}, aggregation={self.aggregation}'
        )


class TemporalSSMEfficient(nn.Module):
    """
    Efficient Temporal SSM with reduced computation.

    Instead of processing each spatial location independently (H*W sequences),
    this version uses spatial downsampling to reduce computation while still
    capturing temporal dynamics.

    Process:
        1. Downsample spatially: [B, T, H, W] -> [B, T, h', w'] where h'*w' << H*W
        2. Apply temporal SSM on downsampled features
        3. Upsample back to original resolution

    Args:
        in_channels (int): Number of temporal bins
        out_channels (int): Output channels
        state_dim (int): SSM state dimension
        downsample_size (int): Spatial size after downsampling. Default: 16
        dropout (float): Dropout probability

    Note: This trades some spatial precision for significant speedup.
    """

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 16,
        state_dim: int = 16,
        downsample_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_size = downsample_size

        # Spatial downsampling
        self.downsample = nn.AdaptiveAvgPool2d((downsample_size, downsample_size))

        # Input projection
        self.input_proj = nn.Linear(1, out_channels)

        # Temporal S5
        self.temporal_s5 = S5Block(
            dim=out_channels,
            state_dim=state_dim,
            bidir=False,
            bandlimit=0.5,
            glu=True,
            ff_mult=1.0,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )

        # Output projection
        self.output_proj = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor, step_scale: float = 1.0) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W]
            step_scale: float

        Returns:
            y: [B, out_channels, H, W]
        """
        b, t, h_orig, w_orig = x.shape
        h_down, w_down = self.downsample_size, self.downsample_size

        # Downsample spatially: [B, T, H, W] -> [B, T, h', w']
        x_down = self.downsample(x)

        # Reshape: [B, T, h', w'] -> [B*h'*w', T, 1]
        x_seq = rearrange(x_down, 'b t h w -> (b h w) t 1')

        # Project: [B*h'*w', T, 1] -> [B*h'*w', T, C]
        x_seq = self.input_proj(x_seq)

        # SSM
        batch_seq = b * h_down * w_down
        state = self.temporal_s5.s5.initial_state(batch_seq).to(x.device)
        x_out, _ = self.temporal_s5(x_seq, state, step_scale=step_scale)

        # Take last time step: [B*h'*w', C]
        x_agg = x_out[:, -1, :]
        x_agg = self.output_proj(x_agg)

        # Reshape: [B*h'*w', C] -> [B, C, h', w']
        y_down = rearrange(x_agg, '(b h w) c -> b c h w', b=b, h=h_down, w=w_down)

        # Upsample back: [B, C, h', w'] -> [B, C, H, W]
        y = nn.functional.interpolate(
            y_down, size=(h_orig, w_orig), mode='bilinear', align_corners=True
        )

        return y


class SpatioTemporalSSM(nn.Module):
    """
    Complete Spatio-Temporal SSM module.

    Combines:
    1. Temporal SSM: Models dependencies across T bins (with step_scale)
    2. Spatial SSM: Models dependencies across H*W locations

    This provides comprehensive spatio-temporal feature extraction where
    step_scale affects the temporal modeling based on data frequency.

    Args:
        in_channels (int): Number of temporal bins (e.g., 15)
        out_channels (int): Output feature channels
        state_dim (int): SSM state dimension for both temporal and spatial
        dropout (float): Dropout probability
        use_spatial_ssm (bool): Whether to apply spatial SSM after temporal
    """

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 16,
        state_dim: int = 16,
        dropout: float = 0.0,
        use_spatial_ssm: bool = True
    ):
        super().__init__()
        self.use_spatial_ssm = use_spatial_ssm

        # Temporal SSM (step_scale is effective here!)
        self.temporal_ssm = TemporalSSM(
            in_channels=in_channels,
            out_channels=out_channels,
            state_dim=state_dim,
            dropout=dropout,
            aggregation='last'
        )

        # Spatial SSM (optional)
        if use_spatial_ssm:
            self.spatial_s5 = S5Block(
                dim=out_channels,
                state_dim=state_dim,
                bidir=False,
                bandlimit=0.5,
                glu=True,
                ff_mult=1.0,
                ff_dropout=dropout,
                attn_dropout=dropout,
            )

    def forward(self, x: torch.Tensor, step_scale: float = 1.0) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W] - Event tensor
            step_scale: float - Temporal step scale (only affects temporal SSM)

        Returns:
            y: [B, out_channels, H, W]
        """
        b = x.shape[0]

        # Temporal SSM (step_scale is physically meaningful here!)
        x_temporal = self.temporal_ssm(x, step_scale=step_scale)
        # x_temporal: [B, C, H, W]

        if not self.use_spatial_ssm:
            return x_temporal

        # Spatial SSM (fixed step_scale=1.0, not frequency-dependent)
        c, h, w = x_temporal.shape[1], x_temporal.shape[2], x_temporal.shape[3]
        x_seq = rearrange(x_temporal, 'b c h w -> b (h w) c')
        state = self.spatial_s5.s5.initial_state(b).to(x.device)
        x_spatial, _ = self.spatial_s5(x_seq, state, step_scale=1.0)
        y = rearrange(x_spatial, 'b (h w) c -> b c h w', h=h, w=w)

        return y


# ============ Convenience Functions ============

def get_step_scale_for_dataset(dataset_name: str) -> float:
    """
    Get appropriate step_scale for a given dataset.

    Args:
        dataset_name: 'dt1', 'dt4', 'dsec', 'mvsec', etc.

    Returns:
        step_scale: Appropriate scaling factor
    """
    dataset_frequencies = {
        'dt1': 60.0,   # HREM dt1
        'dt4': 15.0,   # HREM dt4
        'dsec': 20.0,  # DSEC ~20Hz
        'mvsec': 45.0, # MVSEC ~45Hz
    }

    freq = dataset_frequencies.get(dataset_name.lower(), FREQ_REFERENCE)
    return compute_step_scale(freq)


# ============ Export ============
__all__ = [
    'TemporalSSM',
    'TemporalSSMEfficient',
    'TemporalSSMPatchwise',
    'SpatioTemporalSSM',
    'compute_step_scale',
    'get_step_scale_for_dataset',
    'STEP_SCALE_DT1',
    'STEP_SCALE_DT4',
    'FREQ_DT1',
    'FREQ_DT4',
]
