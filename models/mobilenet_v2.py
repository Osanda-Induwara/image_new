"""MobileNetV2 (ImageNet), the best region model in image_copy; light."""
import torch.nn as nn
from torchvision import models

from .common import DEFAULT_HPARAMS, SEARCH_SPACE

__all__ = ["NAME", "build", "DEFAULT_HPARAMS", "SEARCH_SPACE"]
NAME = "mobilenet_v2"


def build(n_classes, dropout=0.3, pretrained=True):
    """Return (model, head module name, Grad-CAM layer)."""
    m = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, n_classes))
    return m, "classifier", m.features[-1]
