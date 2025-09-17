import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJ_DIR = BASE_DIR.parent
RESULT_DIR = BASE_DIR / "results"
VIS_DIR = RESULT_DIR / "visualizations"
DATA_DIR = PROJ_DIR / "datasets"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = conv_bn_relu(1, 32)
        self.block2 = conv_bn_relu(32, 64)
        self.block3 = conv_bn_relu(64, 128)
        self.block4 = conv_bn_relu(128, 256)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.block1(x)
        e1 = self.pool(x1)

        x2 = self.block2(e1)
        e2 = self.pool(x2)

        x3 = self.block3(e2)
        e3 = self.pool(x3)

        x4 = self.block4(e3)
        e4 = self.pool(x4)
        return e1, e2, e3, e4


class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the heatmap regression network.

        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints

        # Encoder (downsampling path)
        # Input: [batch, 1, 128, 128]
        # Progressively downsample to extract features

        # Decoder (upsampling path)
        # Progressively upsample back to heatmap resolution
        # Output: [batch, num_keypoints, 64, 64]

        # Skip connections between encoder and decoder
        self.enc = Encoder()

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(32, num_keypoints, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, 1, 128, 128]

        Returns:
            heatmaps: Tensor of shape [batch, num_keypoints, 64, 64]
        """
        e1, e2, e3, e4 = self.enc(x)

        # 解码 + skip
        u4 = self.deconv4(e4)
        u4 = torch.cat([u4, e3], dim=1)

        u3 = self.deconv3(u4)
        u3 = torch.cat([u3, e2], dim=1)

        u2 = self.deconv2(u3)

        out = self.head(u2)
        return out


class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the direct regression network.

        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints

        # Use same encoder architecture as HeatmapNet
        # But add global pooling and fully connected layers
        # Output: [batch, num_keypoints * 2]
        self.enc = Encoder()
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc3 = nn.Linear(64, num_keypoints * 2)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, 1, 128, 128]

        Returns:
            coords: Tensor of shape [batch, num_keypoints * 2]
                   Values in range [0, 1] (normalized coordinates)
        """
        _, _, _, e4 = self.enc(x)
        z = self.gap(e4).flatten(1)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        coords = self.act(z)
        return coords