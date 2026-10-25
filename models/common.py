"""Hyperparameters shared by the three transfer-learning models of image_new.

DEFAULT_HPARAMS = the official image recipe (unimodels/image/train_image.py), so "before tuning" is the recipe
already reported. SEARCH_SPACE = what tune.py samples from (random search, validation only).
"""
DEFAULT_HPARAMS = {
    "lr_head": 1e-3,          # AdamW learning rate of the new classification head
    "lr_backbone": 1e-4,      # ... of the pretrained backbone (fine-tuning stage)
    "weight_decay": 1e-4,
    "dropout": 0.3,           # before the final linear layer
    "label_smoothing": 0.0,
    "batch": 32,
    "head_epochs": 5,         # stage 1: frozen backbone, head only (0 = skip)
    "finetune_epochs": 30,    # stage 2: whole network, early stopping on validation loss
    "patience": 7,
}

# (kind, values): "log" = log-uniform between the two bounds, "uniform" = uniform, "choice" = one of the list
SEARCH_SPACE = {
    "lr_head": ("log", 3e-4, 3e-3),
    "lr_backbone": ("log", 1e-5, 3e-4),
    "weight_decay": ("log", 1e-5, 1e-2),
    "dropout": ("uniform", 0.1, 0.5),
    "label_smoothing": ("choice", [0.0, 0.05, 0.1]),
    "batch": ("choice", [16, 32]),
    "head_epochs": ("choice", [0, 3, 5]),
    "patience": ("choice", [5, 7, 10]),
}


def sample_hparams(rng, space=SEARCH_SPACE, base=DEFAULT_HPARAMS):
    """One random configuration from the search space (keys not in the space keep their default)."""
    hp = dict(base)
    for k, spec in space.items():
        kind = spec[0]
        if kind == "log":
            hp[k] = float(10 ** rng.uniform(_log10(spec[1]), _log10(spec[2])))
        elif kind == "uniform":
            hp[k] = float(rng.uniform(spec[1], spec[2]))
        else:
            v = spec[1][int(rng.integers(len(spec[1])))]
            hp[k] = v.item() if hasattr(v, "item") else v
    return hp


def _log10(x):
    import math
    return math.log10(x)
