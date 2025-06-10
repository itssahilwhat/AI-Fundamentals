import torch
import torch.nn as nn


# Depthwise Separable Conv Block (for v1)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),  # depthwise
            nn.BatchNorm2d(in_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c, out_c, 1, bias=False),                         # pointwise
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# Inverted Residual Block (for v2, v3, v4)
class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion):
        super().__init__()
        hidden_c = in_c * expansion
        self.use_res = in_c == out_c and stride == 1

        self.block = nn.Sequential(
            nn.Conv2d(in_c, hidden_c, 1, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_c, hidden_c, 3, stride, 1, groups=hidden_c, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


# Squeeze-and-Excitation Block (for v3, v4)
class SEBlock(nn.Module):
    def __init__(self, in_c, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_c, in_c // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // reduction, in_c),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


# Unified MobileNet Family Model
class MobileNetFamily(nn.Module):
    def __init__(self, version="v1", num_classes=10):
        super().__init__()

        self.version = version.lower()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        layers = []
        in_c = 32

        if version == "v1":
            cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1)]
            for out_c, s in cfg:
                layers.append(DepthwiseSeparableConv(in_c, out_c, s))
                in_c = out_c

        elif version in ["v2", "v3", "v4"]:
            cfg = [
                # out_c, stride, expansion, use_se
                (16, 1, 1, False),
                (24, 2, 6, False),
                (24, 1, 6, False),
                (32, 2, 6, version != "v2"),  # v3/v4 use SE
                (32, 1, 6, version != "v2")
            ]
            for out_c, s, exp, use_se in cfg:
                block = InvertedResidual(in_c, out_c, s, exp)
                layers.append(block)
                if use_se:
                    layers.append(SEBlock(out_c))
                in_c = out_c

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_c, 128),
            nn.Hardswish() if version in ["v3", "v4"] else nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)
