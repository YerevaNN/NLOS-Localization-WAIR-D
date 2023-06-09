import torch
import torch.nn as nn

from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool):
        super().__init__()

        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        y = self.model(x)
        return y
