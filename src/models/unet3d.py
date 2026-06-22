"""
3D U-Net Segmentation Model Architecture in PyTorch
Author: Senior AI Engineer & Medical Imaging Expert
"""

import torch
import torch.nn as nn
from typing import List


class DoubleConv3D(nn.Module):
    """
    Two sequential 3D convolutions, each followed by 3D Batch Normalization and ReLU.
    [Conv3d -> BatchNorm3d -> ReLU] x 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock3D(nn.Module):
    """
    Downscaling block in 3D U-Net:
    3D Max Pooling -> Double Convolution
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pooled = self.pool(x)
        return self.conv(x_pooled)


class UpBlock3D(nn.Module):
    """
    Upscaling block in 3D U-Net:
    3D ConvTranspose3d -> Concatenate Skip Connection -> Double Convolution
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Halves channel count, doubles spatial sizes (D, H, W)
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature maps from lower block
            skip: Skip connection feature maps from encoder block
        """
        x_up = self.up(x)
        
        # Safe concatenation in case of minor dimension mismatches due to odd sizing
        diff_d = skip.size()[2] - x_up.size()[2]
        diff_h = skip.size()[3] - x_up.size()[3]
        diff_w = skip.size()[4] - x_up.size()[4]
        
        if diff_d > 0 or diff_h > 0 or diff_w > 0:
            x_up = nn.functional.pad(
                x_up, 
                [diff_w // 2, diff_w - diff_w // 2,
                 diff_h // 2, diff_h - diff_h // 2,
                 diff_d // 2, diff_d - diff_d // 2]
            )
            
        # Concatenate along the channel dimension (dim 1)
        x_concat = torch.cat([skip, x_up], dim=1)
        return self.conv(x_concat)


class UNet3D(nn.Module):
    """
    Complete 3D U-Net Model.
    Designed for volumetric image segmentation.
    Input shape: (B, in_channels, D, H, W)
    Output shape: (B, out_channels, D, H, W)
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 features: List[int] = None):
        """
        Args:
            in_channels: Number of input channels (typically 1 for CT scans)
            out_channels: Number of output channels (typically 1 for binary mask)
            features: Number of intermediate filters at each layer
        """
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        # Encoder path
        self.inc = DoubleConv3D(in_channels, features[0])
        self.down1 = DownBlock3D(features[0], features[1])
        self.down2 = DownBlock3D(features[1], features[2])
        self.down3 = DownBlock3D(features[2], features[3])

        # Bottom bottleneck
        self.bottleneck = DoubleConv3D(features[3], features[3] * 2)

        # Decoder path
        self.up3 = UpBlock3D(features[3] * 2, features[3])
        self.up2 = UpBlock3D(features[3], features[2])
        self.up1 = UpBlock3D(features[2], features[1])
        self.up0 = UpBlock3D(features[1], features[0])

        # Output final mapping layer
        # Maps local feature map back to target segment channels
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (Contracting Path)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        b = self.bottleneck(x4)

        # Decoder (Expanding Path)
        x = self.up3(b, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)
        x = self.up0(x, x1)

        logits = self.outc(x)
        return self.sigmoid(logits)


# Verification check
if __name__ == "__main__":
    # Test forward pass with typical 3D CT patch (Batch=1, Ch=1, D=16, H=128, W=128)
    model = UNet3D(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 16, 128, 128)
    output = model(x)
    print("3D U-Net Model instantiated successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert x.shape == output.shape, "Input and Output shapes should match."
