"""
Frequency Estimator for Event Cameras

This module estimates the global event frequency from event tensor statistics
to enable frequency-adaptive processing in SSM-based optical flow estimation.

Author: Frequency-Adaptive SSM Integration
"""

import torch
import torch.nn as nn


class FrequencyEstimator(nn.Module):
    """
    Estimates global event frequency from event tensor statistics.

    The frequency estimator analyzes temporal event patterns to determine
    whether the current input represents high-frequency motion (fast, dense events)
    or low-frequency motion (slow, sparse events).

    Args:
        num_bins (int): Number of temporal bins in the event representation.
            Default: 15 (for HREM dataset)
        hidden_dim (int): Hidden dimension for the MLP. Default: 64
        dropout (float): Dropout probability. Default: 0.1

    Input:
        event_tensor: [B, num_bins, H, W] - Event representation tensor

    Output:
        frequency_score: [B, 1] - Frequency score in range [0, 1]
            - 0: Low frequency (slow motion, sparse events)
            - 1: High frequency (fast motion, dense events)

    Statistical Features Extracted:
        1. Mean intensity per bin: Indicates average event density
        2. Standard deviation per bin: Indicates temporal variation
        3. Sparsity: Percentage of non-zero pixels
    """

    def __init__(self, num_bins=15, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_bins = num_bins

        # MLP to process concatenated statistics
        # Input dimension: num_bins * 3 (mean + std + sparsity per bin)
        self.mlp = nn.Sequential(
            nn.Linear(num_bins * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Ensure output is in [0, 1]
        )

    def forward(self, event_tensor):
        """
        Estimate frequency score from event tensor.

        Args:
            event_tensor: [B, num_bins, H, W] - Event representation

        Returns:
            frequency_score: [B, 1] - Estimated frequency in [0, 1]
        """
        # Extract statistics via global pooling over spatial dimensions
        # Shape: [B, num_bins]
        mean_per_bin = event_tensor.mean(dim=[2, 3])
        std_per_bin = event_tensor.std(dim=[2, 3])

        # Compute sparsity (percentage of non-zero pixels)
        # Using small threshold to handle numerical precision
        sparsity = (event_tensor.abs() > 1e-5).float().mean(dim=[2, 3])

        # Concatenate all statistical features: [B, num_bins * 3]
        stats = torch.cat([mean_per_bin, std_per_bin, sparsity], dim=1)

        # Pass through MLP to get frequency score
        freq_score = self.mlp(stats)  # [B, 1]

        return freq_score

    def extra_repr(self):
        """String representation for debugging."""
        return f"num_bins={self.num_bins}, hidden_dim={self.mlp[0].out_features}"


class MultiScaleFrequencyEstimator(nn.Module):
    """
    Multi-scale frequency estimator for hierarchical flow estimation.

    This variant can estimate different frequencies at different spatial scales,
    useful for hierarchical decoders where coarse levels need global frequency
    and fine levels need local frequency estimates.

    Note: This is an advanced variant not used in MVP. Included for future work.

    Args:
        num_bins (int): Number of temporal bins
        num_scales (int): Number of spatial scales to process
        hidden_dim (int): Hidden dimension for processing
        dropout (float): Dropout probability
    """

    def __init__(self, num_bins=15, num_scales=3, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_bins = num_bins
        self.num_scales = num_scales

        # Spatial pooling layers for each scale
        self.pooling_scales = nn.ModuleList([
            nn.AdaptiveAvgPool2d((scale_size, scale_size))
            for scale_size in [1, 4, 8]  # Global, coarse, medium
        ])

        # Shared MLP for all scales
        input_dim = num_bins * 3 * num_scales
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_scales),  # One frequency per scale
            nn.Sigmoid()
        )

    def forward(self, event_tensor):
        """
        Estimate multi-scale frequencies.

        Args:
            event_tensor: [B, num_bins, H, W]

        Returns:
            frequency_scores: [B, num_scales] - Frequency at each scale
        """
        all_stats = []

        for pooling in self.pooling_scales:
            # Pool to different scales
            pooled = pooling(event_tensor)  # [B, num_bins, scale_h, scale_w]

            # Compute statistics
            mean_per_bin = pooled.mean(dim=[2, 3])
            std_per_bin = pooled.std(dim=[2, 3])
            sparsity = (pooled.abs() > 1e-5).float().mean(dim=[2, 3])

            scale_stats = torch.cat([mean_per_bin, std_per_bin, sparsity], dim=1)
            all_stats.append(scale_stats)

        # Concatenate stats from all scales: [B, num_bins * 3 * num_scales]
        combined_stats = torch.cat(all_stats, dim=1)

        # Predict frequencies for all scales
        freq_scores = self.mlp(combined_stats)  # [B, num_scales]

        return freq_scores
