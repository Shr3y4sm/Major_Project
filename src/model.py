"""
src/model.py
============
Typed refactor of the Phase 1 3D UNet architecture (BraTS 2021 segmentation).

The layer topology is extracted verbatim from `braintumour-phase-1.ipynb`
(Phase 1) — no architectural changes are made. Only type annotations and
module-level docstrings are added.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ConvBlock3D",
    "Encoder3D",
    "Decoder3D",
    "UNet3D",
    "load_model",
]


class ConvBlock3D(nn.Module):
    """Two stacked (Conv3d -> BatchNorm3d -> ReLU) blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder3D(nn.Module):
    """Convolutional feature extractor followed by 2x max-pooling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return feat, self.pool(feat)


class Decoder3D(nn.Module):
    """Transposed-convolution upsampler with skip-connection fusion.

    Preserves the Phase 1 dynamic padding behaviour: if the upsampled
    tensor does not match the skip tensor's spatial shape, it is padded
    symmetrically (with the extra voxel on the trailing side).
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            diff = [skip.shape[i] - x.shape[i] for i in (2, 3, 4)]
            x = F.pad(
                x,
                [
                    diff[2] // 2,
                    diff[2] - diff[2] // 2,
                    diff[1] // 2,
                    diff[1] - diff[1] // 2,
                    diff[0] // 2,
                    diff[0] - diff[0] // 2,
                ],
            )
        return self.conv(torch.cat([x, skip], dim=1))


class UNet3D(nn.Module):
    """3D U-Net for multi-modal brain tumour segmentation.

    Args:
        in_channels: Number of input modalities (BraTS: 4).
        out_channels: Number of output classes (BraTS remapped: 4).
        base: Base feature multiplier (Phase 1 uses 32).
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base: int = 32) -> None:
        super().__init__()
        self.enc1 = Encoder3D(in_channels, base)
        self.enc2 = Encoder3D(base, base * 2)
        self.enc3 = Encoder3D(base * 2, base * 4)
        self.enc4 = Encoder3D(base * 4, base * 8)
        self.bottleneck = ConvBlock3D(base * 8, base * 16)
        self.dec4 = Decoder3D(base * 16, base * 8, base * 8)
        self.dec3 = Decoder3D(base * 8, base * 4, base * 4)
        self.dec2 = Decoder3D(base * 4, base * 2, base * 2)
        self.dec1 = Decoder3D(base * 2, base, base)
        self.out_conv = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, p1 = self.enc1(x)
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)
        f4, p4 = self.enc4(p3)
        b = self.bottleneck(p4)
        d4 = self.dec4(b, f4)
        d3 = self.dec3(d4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        return self.out_conv(d1)


def load_model(
    checkpoint_path: str = "artifacts/best_unet.pt",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Reconstruct the Phase 1 UNet3D and load its state_dict.

    Args:
        checkpoint_path: Path to the saved state_dict checkpoint.
        device: Torch device to load onto (defaults to CUDA if available).

    Returns:
        The model in evaluation mode on the target device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Train the Phase 1 model first or run `python -m src.export_onnx` "
            "to generate a dummy checkpoint."
        )

    model = UNet3D(in_channels=4, out_channels=4)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()