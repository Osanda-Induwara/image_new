"""The transfer-learning models of image_new, one file each."""
from . import efficientnet_b0, mobilenet_v2, resnet18

MODELS = {m.NAME: m for m in (efficientnet_b0, resnet18, mobilenet_v2)}
