"""ResNet18 (ImageNet), second best for grade in our image run and selected for grade in image_copy."""
import torch.nn as nn
from torchvision import models

from .common import DEFAULT_HPARAMS, SEARCH_SPACE

__all__ = ["NAME", "build", "DEFAULT_HPARAMS", "SEARCH_SPACE"]
NAME = "resnet18"


def build(n_classes, dropout=0.3, pretrained=True):
    """Return (model, head module name, Grad-CAM layer)."""
    m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, n_classes))
    return m, "fc", m.layer4
