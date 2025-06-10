import torch
import torch.nn as nn


# 🔹 Basic Conv Block (Conv → BN → ReLU)
class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c, k, **kwargs):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


# 🔹 Inception-A Block (used in v1–v4)
class InceptionA(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.b1 = ConvBlock(in_c, 64, 1)
        self.b2 = nn.Sequential(ConvBlock(in_c, 48, 1), ConvBlock(48, 64, 5, padding=2))
        self.b3 = nn.Sequential(ConvBlock(in_c, 64, 1),
                                ConvBlock(64, 96, 3, padding=1),
                                ConvBlock(96, 96, 3, padding=1))
        self.b4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1),
                                ConvBlock(in_c, 32, 1))

    def forward(self, x):
        out = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        return torch.cat(out, dim=1)


# 🔹 Inception-ResNet Block
class InceptionResNetBlock(nn.Module):
    def __init__(self, in_c, scale=1.0):
        super().__init__()
        self.b1 = ConvBlock(in_c, 32, 1)
        self.b2 = nn.Sequential(ConvBlock(in_c, 32, 1), ConvBlock(32, 32, 3, padding=1))
        self.b3 = nn.Sequential(ConvBlock(in_c, 32, 1), ConvBlock(32, 48, 3, padding=1),
                                ConvBlock(48, 64, 3, padding=1))
        self.conv = nn.Conv2d(128, in_c, 1)
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        mixed = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        up = self.conv(mixed)
        return self.relu(x + self.scale * up)


# 🔸 Unified Inception Family Model
class InceptionFamilyNet(nn.Module):
    def __init__(self, version="v1", num_classes=10):
        super().__init__()
        self.stem = ConvBlock(1, 64, 7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, 2, 1)

        if version == "v1":
            self.body = nn.Sequential(InceptionA(64), InceptionA(256))
            final_channels = 256
        elif version in ["v2", "v3"]:
            self.body = nn.Sequential(*[InceptionA(64) for _ in range(3)])
            final_channels = 256
        elif version == "v4":
            self.body = nn.Sequential(*[InceptionA(64) for _ in range(4)])
            final_channels = 256
        elif version in ["resnetv1", "resnetv2"]:
            self.body = nn.Sequential(*[InceptionResNetBlock(64) for _ in range(5)])
            final_channels = 64
        else:
            raise ValueError("Unknown version.")

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(final_channels, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.stem(x))
        x = self.body(x)
        return self.classifier(x)