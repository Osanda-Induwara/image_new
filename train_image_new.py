"""image_new training: grade on grey-masked circle crops, grades balanced by augmentation, three transfer-learning
models (EfficientNet-B0, ResNet18, MobileNetV2), each with default or tuned hyperparameters.

Per (seed, model), on the shared sample-level split:
  * the training photos are balanced by grade with augmentation.balance_training_set (copies only of training
    photos), then shuffled; online_augment() adds random flips and rotation every epoch;
  * two-stage fine-tuning: head only on a frozen backbone, then the whole network (AdamW), early stopping on
    validation loss, label smoothing optional;
  * validation and test: real photos only, softmax averaged per physical sample.
Selection, reports, epoch curves, Grad-CAM and resuming come from common/cnn_train.py.

    python -m unimodels.image_new.train_image_new --config default --tasks grade
    python -m unimodels.image_new.train_image_new --config tuned --tasks grade    # after tune.py
    (add --fast --out <dir> for a smoke test)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..common import config as C
from ..common.cnn_train import (NUM_WORKERS, CropDataset, ImageBranch, epoch_pass, freeze_backbone,
                                history_row, main)
from ..image.preprocess_images import OUT_DIR_MASKED
from .augmentation import balance_training_set, online_augment
from .models import MODELS
from .models.common import DEFAULT_HPARAMS

NEW_DIR = C.UNIMODELS_DIR / "image_new"
TUNING_DIR = NEW_DIR / "results" / "tuning"
_TUNING = {"dir": TUNING_DIR}          # --tuning-dir (e.g. the FAST smoke folder) overrides it


def fast_hparams(hp):
    """Smoke test: at most 1 head-only epoch and 1 fine-tuning epoch."""
    return dict(hp, head_epochs=min(int(hp["head_epochs"]), 1), finetune_epochs=1, patience=1)


def _settings(hparams, fast):
    hp = {m: (fast_hparams(h) if fast else dict(h)) for m, h in hparams.items()}
    return {"hparams": hp, "batch": 32, "image_size": C.IMAGE_SIZE, "cache_side": C.IMAGE_SIZE,
            "max_images_per_sample": 3 if fast else None, "balancing": "grade, augmented copies of training photos",
            "crop": "masked circle (image/preprocess_images.py --crop masked)"}


def default_settings(fast):
    return _settings({m: dict(DEFAULT_HPARAMS) for m in MODELS}, fast)


def load_best_hparams(path=None):
    path = Path(path or _TUNING["dir"] / "best_hparams.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run unimodels.image_new.tune first")
    return json.loads(Path(path).read_text())["best"]


def tuned_settings(fast):
    return _settings(load_best_hparams(), fast)


def train_model(name, arr, tr, va, n_classes, seed, cfg, device):
    """Balance the training photos by grade, then two-stage fine-tuning with this model's hyperparameters.
    Returns (model with the best-validation-loss weights, history, best epoch)."""
    hp = cfg["hparams"][name]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, head, _ = MODELS[name].build(n_classes, dropout=hp["dropout"], pretrained=True)
    model.to(device)
    arr_b, y_b, _, _ = balance_training_set(arr, tr["idx"], tr["y"], tr["keys"], seed)
    g = torch.Generator().manual_seed(seed)
    pin = device.type == "cuda"
    tl = DataLoader(CropDataset(arr_b, np.arange(len(y_b)), y_b, online_augment(), cfg["image_size"]),
                    batch_size=int(hp["batch"]), shuffle=True, generator=g, num_workers=NUM_WORKERS,
                    pin_memory=pin, drop_last=len(y_b) > hp["batch"])
    vl = DataLoader(CropDataset(arr, va["idx"], va["y"], (), cfg["image_size"]), batch_size=32, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=pin)
    train_loss = nn.CrossEntropyLoss(label_smoothing=float(hp["label_smoothing"]))
    val_loss = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    head_params = [p for n, p in model.named_parameters() if n.startswith(head + ".")]
    body_params = [p for n, p in model.named_parameters() if not n.startswith(head + ".")]
    stages = [("head_only", int(hp["head_epochs"]), [{"params": head_params, "lr": hp["lr_head"]}], True),
              ("finetune", int(hp["finetune_epochs"]), [{"params": body_params, "lr": hp["lr_backbone"]},
                                                        {"params": head_params, "lr": hp["lr_head"] / 10}], False)]

    best_loss, best_state, best_epoch, history, epoch = float("inf"), None, 0, [], 0
    for stage, n_epochs, groups, frozen in stages:
        if n_epochs <= 0:
            continue
        freeze_backbone(model, head, frozen)
        opt = torch.optim.AdamW(groups, weight_decay=hp["weight_decay"])
        bad = 0
        for _ in range(n_epochs):
            epoch += 1
            tr_out = epoch_pass(model, tl, device, train_loss, opt, scaler, head, freeze_bn=frozen)
            va_out = epoch_pass(model, vl, device, val_loss)
            history.append(history_row(epoch, stage, tr_out, va_out, va))
            if va_out[0] < best_loss - 1e-4:
                best_loss, best_epoch, bad = va_out[0], epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if stage != "head_only" and bad >= hp["patience"]:
                    break
    model.load_state_dict(best_state)
    freeze_backbone(model, head, False)
    return model, pd.DataFrame(history), best_epoch


def _build(name, n, pretrained):
    return MODELS[name].build(n, pretrained=pretrained)


def make_branch(config):
    return ImageBranch(
        name=f"image_new_{config}", label=f"Image new ({config} hyperparameters)", models=tuple(MODELS),
        fast_models=("efficientnet_b0", "mobilenet_v2"), build_model=_build, train_model=train_model,
        settings=default_settings if config == "default" else tuned_settings, prep_dir=OUT_DIR_MASKED,
        default_out=NEW_DIR / f"results_{config}", doc=__doc__,
    )


DEFAULT_BRANCH = make_branch("default")


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", choices=["default", "tuned"], default="default")
    pre.add_argument("--tuning-dir", default=None, help="where tune.py wrote best_hparams.json")
    known, rest = pre.parse_known_args()
    if known.tuning_dir:
        _TUNING["dir"] = Path(known.tuning_dir)
    sys.argv = [sys.argv[0]] + rest
    main(make_branch(known.config))
