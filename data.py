"""Data helpers shared by tune.py and learning_curve.py: the same crops, labels, splits and per-sample
predictions as the training run (common/cnn_train.py), for the grade task."""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from ..common.cnn_train import _prep_signature, build_parts, by_sample, load_crops, predict_images
from ..common.metrics import summary_metrics
from ..common.samples import build_sample_table
from ..common.splits import load_or_make_split
from .train_image_new import DEFAULT_BRANCH

TASK = "grade"


def prepare(data_root, fast=False):
    """Crops, labels and helpers for the grade task."""
    samples, _ = build_sample_table(data_root)
    man, arr = load_crops(DEFAULT_BRANCH, samples["sample_key"], DEFAULT_BRANCH.settings(fast)["cache_side"],
                          3 if fast else None)
    enc = LabelEncoder().fit(samples[TASK])
    y_all = enc.transform(man["sample_key"].map(dict(zip(samples["sample_key"], samples[TASK]))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {"samples": samples, "man": man, "arr": arr, "enc": enc, "classes": list(enc.classes_), "y_all": y_all,
            "device": device, "prep": _prep_signature(DEFAULT_BRANCH)}


def parts_for(d, seed):
    return build_parts(d["man"], d["y_all"], load_or_make_split(d["samples"], seed))


def sample_predictions(model, d, part, cfg, seed, label, subset):
    """Per-sample frame (seed, model, subset, sample_key, true, pred, p_<class>) as in all_predictions.csv."""
    probs = predict_images(model, d["arr"], part["idx"], d["device"], cfg)
    sp = by_sample(part["keys"], probs)
    y = pd.Series(part["y"], index=list(part["keys"])).groupby(level=0).first().loc[sp.index]
    frame = pd.DataFrame({"seed": seed, "model": label, "subset": subset, "sample_key": sp.index,
                          "true": d["enc"].inverse_transform(y.to_numpy()),
                          "pred": d["enc"].inverse_transform(sp.to_numpy().argmax(1))})
    for j, c in enumerate(d["classes"]):
        frame[f"p_{c}"] = sp[j].to_numpy()
    return frame


def frame_metrics(frame):
    return summary_metrics(frame["true"], frame["pred"])


def subset_part(part, keep_keys):
    """The rows of a part whose sample key is in keep_keys."""
    sel = np.isin(part["keys"], list(keep_keys))
    return {k: v[sel] for k, v in part.items()}
