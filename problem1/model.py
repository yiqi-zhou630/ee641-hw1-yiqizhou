import torch
import torch.nn as nn

def conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(in_channels)
        self.stem_relu = nn.ReLU(inplace=True)
        out_ch = num_anchors * (5 + num_classes)
        self.pred = nn.Conv2d(in_channels, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.stem_relu(self.stem_bn(self.stem(x)))
        x = self.pred(x)
        return x

class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):
        """
        Initialize the multi-scale detector.

        Args:
            num_classes: Number of object classes (not including background)
            num_anchors: Number of anchors per spatial location
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors


        self.block1a = conv_bn_relu(3, 32, k=3, s=1, p=1)
        self.block1b = conv_bn_relu(32, 64, k=3, s=2, p=1)


        self.block2 = conv_bn_relu(64, 128, k=3, s=2, p=1)


        self.block3 = conv_bn_relu(128, 256, k=3, s=2, p=1)


        self.block4 = conv_bn_relu(256, 512, k=3, s=2, p=1)


        self.head1 = DetectionHead(128, num_anchors, num_classes)
        self.head2 = DetectionHead(256, num_anchors, num_classes)
        self.head3 = DetectionHead(512, num_anchors, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, 3, 224, 224]

        Returns:
            List of 3 tensors (one per scale), each containing predictions
            Shape: [batch, num_anchors * (5 + num_classes), H, W]
            where 5 = 4 bbox coords + 1 objectness score
        """
        x = self.block1a(x)
        x = self.block1b(x)

        f1 = self.block2(x)
        f2 = self.block3(f1)
        f3 = self.block4(f2)

        p1 = self.head1(f1)
        p2 = self.head2(f2)
        p3 = self.head3(f3)

        return [p1, p2, p3]