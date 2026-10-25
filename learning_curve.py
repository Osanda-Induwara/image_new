"""Training samples vs accuracy for the best tuned image_new model (grade).

The selected model of results_tuned/ (chosen on validation) is retrained with its tuned hyperparameters on
20 / 40 / 60 / 80 / 100% of each split's training samples (physical samples, stratified by grade), for seeds
42-44. Every subset is balanced by augmentation exactly as in training, so the curve uses all the augmented data.
Train (the subset's real photos), validation and test accuracy and macro-F1 are measured per sample.
A rising validation/test curve at 100% means more samples would help; a wide train-test gap means overfitting.
This is a diagnostic: it never changes model selection. Each point is saved, so the run resumes.

    python -m unimodels.image_new.learning_curve
    python -m unimodels.image_new.learning_curve --fast --out <dir> --results <tuned results> --tuning-dir <dir>
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from ..common import config as C  # noqa: E402
from .data import frame_metrics, parts_for, prepare, sample_predictions, subset_part  # noqa: E402
from .train_image_new import NEW_DIR, TUNING_DIR, fast_hparams, load_best_hparams, train_model  # noqa: E402

FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def choose_samples(part, frac, seed):
    """frac of the training samples (physical samples), stratified by grade when possible."""
    keys = pd.Series(part["y"], index=list(part["keys"])).groupby(level=0).first()
    if frac >= 1.0:
        return list(keys.index)
    n = max(int(round(frac * len(keys))), len(set(keys)))
    for strat in (keys.to_numpy(), None):
        try:
            sub, _ = train_test_split(list(keys.index), train_size=n, random_state=seed, stratify=strat)
            return sub
        except ValueError:
            continue
    raise RuntimeError("could not subsample")


def run(data_root, out_dir, results_dir, tuning_dir, fractions, seeds, fast):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = json.loads((Path(results_dir) / "grade" / "best_model.json").read_text())["best_model"]
    hp = load_best_hparams(Path(tuning_dir) / "best_hparams.json")[best]
    hp = fast_hparams(hp) if fast else hp
    cfg = {"hparams": {best: hp}, "batch": 32, "image_size": C.IMAGE_SIZE}
    d = prepare(data_root, fast)
    rows = []
    for seed in seeds:
        parts = parts_for(d, seed)
        for frac in fractions:
            path = out_dir / f"seed{seed}_frac{int(frac * 100):03d}.json"
            signature = {"model": best, "hparams": hp, "seed": seed, "frac": frac, "prep": d["prep"]}
            if path.exists() and json.loads(path.read_text()).get("signature") == json.loads(json.dumps(signature)):
                rows.append(json.loads(path.read_text())["row"])
                continue
            t0 = time.time()
            tr = subset_part(parts["train"], choose_samples(parts["train"], frac, seed))
            counts = np.bincount(tr["y"], minlength=len(d["classes"]))
            n_balanced = int((counts > 0).sum() * counts.max())   # every present grade padded to the largest
            model, _, best_epoch = train_model(best, d["arr"], tr, parts["val"], len(d["classes"]), seed, cfg,
                                               d["device"])
            row = {"seed": seed, "fraction": frac, "n_train_samples": int(len(set(tr["keys"]))),
                   "n_train_photos": int(len(tr["idx"])), "n_balanced_photos": n_balanced,
                   "best_epoch": int(best_epoch)}
            for name, part in (("train", tr), ("val", parts["val"]), ("test", parts["test"])):
                m = frame_metrics(sample_predictions(model, d, part, cfg, seed, best, name))
                row.update({f"{name}_accuracy": m["accuracy"], f"{name}_macro_f1": m["macro_f1"]})
            row["minutes"] = round((time.time() - t0) / 60, 2)
            path.write_text(json.dumps({"signature": signature, "row": row}, default=str))
            rows.append(row)
            print(f"    seed {seed} {int(frac * 100):3d}%: {row['n_train_samples']} samples "
                  f"({row['n_balanced_photos']} balanced photos)  test accuracy {row['test_accuracy']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "learning_curve.csv", index=False)
    plot(table, best, out_dir / "learning_curve.png")
    return best, table


def plot(table, model, path):
    g = table.groupby("fraction")
    x = g["n_train_samples"].mean()
    balanced = g["n_balanced_photos"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, metric, ylabel in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        for subset, colour, style in (("train", "#4C72B0", "--"), ("val", "#DD8452", "-"), ("test", "#55A868", "-")):
            m, s = g[f"{subset}_{metric}"].mean(), g[f"{subset}_{metric}"].std().fillna(0)
            ax.plot(x, m, style, marker="o", color=colour, label={"train": "training samples", "val": "validation",
                                                                    "test": "test"}[subset])
            ax.fill_between(x, m - s, m + s, color=colour, alpha=0.12)
        for xi, b in zip(x, balanced):
            ax.annotate(f"{int(b)} photos", (xi, 0.02), ha="center", fontsize=7, color="grey")
        ax.set_xlabel("Number of training samples (physical tea samples)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Grade - {model} (tuned): training samples vs score, mean ± std over seeds "
                 f"(grey: balanced training photos incl. augmented copies)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(C.DATA_ROOT))
    p.add_argument("--out", default=str(NEW_DIR / "results" / "learning_curve"))
    p.add_argument("--results", default=str(NEW_DIR / "results_tuned"), help="tuned training results (selection)")
    p.add_argument("--tuning-dir", default=str(TUNING_DIR))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true", help="1 seed, 2 points, 1 epoch; smoke test only")
    a = p.parse_args()
    fractions = (0.5, 1.0) if a.fast else FRACTIONS
    seeds = [42] if a.fast else a.seeds
    best, table = run(Path(a.data_root), Path(a.out), Path(a.results), Path(a.tuning_dir), fractions, seeds, a.fast)
    print(table.groupby("fraction")[["n_train_samples", "n_balanced_photos", "train_accuracy", "val_accuracy",
                                     "test_accuracy"]].mean().round(3).to_string())
    print(f"Model: {best}. Plot: {Path(a.out).resolve() / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
