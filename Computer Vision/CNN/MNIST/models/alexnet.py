import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 11, 4, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 384, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 224)  # Original AlexNet input size
            flatten_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_size, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)