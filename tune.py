"""Hyperparameter tuning for image_new (grade): random search on validation data only.

For each model (EfficientNet-B0, ResNet18, MobileNetV2):
  * trial 0 = the default hyperparameters, trials 1.. = random configurations from models/common.SEARCH_SPACE
    (so the tuned setting can only match or beat the default on validation);
  * every trial is trained on the tuning seeds (42-44) with the same balanced training and evaluated on those
    seeds' validation samples only; test data are never used here;
  * the best trial is chosen like every model in this project (common/selection.py): pooled validation macro-F1,
    trials within 0.01 tied, ties to the lowest validation Brier score.
Each finished trial is saved, so an interrupted search resumes.

    python -m unimodels.image_new.tune                      # 12 trials per model, seeds 42-44
    python -m unimodels.image_new.tune --fast --out <dir>   # smoke test only (2 trials, 1 seed, 1 epoch)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import config as C
from ..common.selection import rule_text, selection_table
from .data import parts_for, prepare, sample_predictions
from .models import MODELS
from .models.common import DEFAULT_HPARAMS, sample_hparams
from .train_image_new import TUNING_DIR, fast_hparams, train_model


def trial_configs(name, n_trials):
    """Deterministic list of configurations for one model: the default, then random samples."""
    rng = np.random.default_rng(1000 + sorted(MODELS).index(name))
    return [dict(DEFAULT_HPARAMS)] + [sample_hparams(rng) for _ in range(n_trials - 1)]


def run_trial(d, name, t, hp, seeds, out_dir, fast):
    tdir = out_dir / name
    tdir.mkdir(parents=True, exist_ok=True)
    meta_path = tdir / f"trial_{t:02d}.json"
    signature = {"model": name, "trial": t, "hparams": hp, "seeds": list(seeds), "prep": d["prep"], "fast": fast}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("signature") == json.loads(json.dumps(signature)):
            return meta, pd.read_csv(tdir / f"trial_{t:02d}_val_predictions.csv")
    t0 = time.time()
    hp_run = fast_hparams(hp) if fast else hp
    cfg = {"hparams": {name: hp_run}, "batch": 32, "image_size": C.IMAGE_SIZE}
    frames, epochs = [], []
    for seed in seeds:
        parts = parts_for(d, seed)
        model, hist, best_epoch = train_model(name, d["arr"], parts["train"], parts["val"], len(d["classes"]),
                                              seed, cfg, d["device"])
        frames.append(sample_predictions(model, d, parts["val"], cfg, seed, f"trial_{t:02d}", "val"))
        epochs.append(int(best_epoch))
    preds = pd.concat(frames, ignore_index=True)
    sel = selection_table(preds, d["classes"], C.SELECTION_TIE_TOLERANCE).iloc[0]
    meta = {"signature": signature, "model": name, "trial": t, **{f"hp_{k}": v for k, v in hp.items()},
            "val_pooled_macro_f1": float(sel["val_pooled_macro_f1"]), "val_pooled_brier": float(sel["val_pooled_brier"]),
            "val_pooled_accuracy": float(sel["val_pooled_accuracy"]), "best_epochs": epochs,
            "minutes": round((time.time() - t0) / 60, 2)}
    preds.to_csv(tdir / f"trial_{t:02d}_val_predictions.csv", index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return meta, preds


def run(data_root, out_dir, models, n_trials, seeds, fast):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = prepare(data_root, fast)
    rows, best, chosen = [], {}, {}
    for name in models:
        configs = trial_configs(name, n_trials)
        all_preds = []
        for t, hp in enumerate(configs):
            meta, preds = run_trial(d, name, t, hp, seeds, out_dir, fast)
            rows.append({k: v for k, v in meta.items() if k != "signature"})
            all_preds.append(preds)
            print(f"    {name:<16} trial {t:2d}{' (default)' if t == 0 else '':<10} val macro-F1 "
                  f"{meta['val_pooled_macro_f1']:.3f}  Brier {meta['val_pooled_brier']:.3f}  ({meta['minutes']} min)",
                  flush=True)
        ranking = selection_table(pd.concat(all_preds, ignore_index=True), d["classes"], C.SELECTION_TIE_TOLERANCE)
        ranking.round(4).to_csv(out_dir / f"{name}_trial_ranking.csv")
        t_best = int(ranking.index[0].split("_")[1])
        best[name], chosen[name] = configs[t_best], t_best
        print(f"  {name}: best = trial {t_best}{' (the default)' if t_best == 0 else ''}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "trials.csv", index=False)
    info = {"best": best, "best_trial": chosen, "n_trials": n_trials, "seeds": list(seeds),
            "rule": rule_text(C.SELECTION_TIE_TOLERANCE, len(seeds)) + " (over trials, validation of the tuning seeds)",
            "smoke_test_only": bool(fast)}
    (out_dir / "best_hparams.json").write_text(json.dumps(info, indent=2, default=str))
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(C.DATA_ROOT))
    p.add_argument("--out", default=str(TUNING_DIR))
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true", help="2 trials, seed 42, 1 epoch; smoke test only")
    a = p.parse_args()
    if a.fast:
        a.trials, a.seeds = 2, [42]
    print(f"Tuning {a.models}: {a.trials} trials each on seeds {a.seeds} (validation only)", flush=True)
    t0 = time.time()
    info = run(Path(a.data_root), Path(a.out), a.models, a.trials, a.seeds, a.fast)
    print(f"Best trials: {info['best_trial']}. Done in {(time.time() - t0) / 60:.1f} min. Written to {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
