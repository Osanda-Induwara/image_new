"""Before vs after hyperparameter tuning (image_new, grade): test accuracy, macro precision, recall and F1 per
model, mean ± std over the 10 seeds, from results_default/ and results_tuned/.

    python -m unimodels.image_new.compare
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .train_image_new import NEW_DIR  # noqa: E402

METRICS = {"accuracy": "Accuracy", "macro_precision": "Precision", "macro_recall": "Recall", "macro_f1": "F1"}


def run(default_dir, tuned_dir, out_dir, task="grade"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for config, folder in (("before tuning (default)", default_dir), ("after tuning", tuned_dir)):
        summary = pd.read_csv(Path(folder) / task / "summary_by_model.csv", index_col=0)
        selected = json.loads((Path(folder) / task / "best_model.json").read_text())["best_model"]
        for model in summary.index:
            if model == "majority_baseline":
                continue
            row = {"model": model, "config": config, "selected": model == selected}
            for m in METRICS:
                row[f"{m}_mean"] = summary.loc[model, f"test_{m}_mean"]
                row[f"{m}_std"] = summary.loc[model, f"test_{m}_std"]
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(["model", "config"], ascending=[True, False])
    table.round(4).to_csv(out_dir / "before_after.csv", index=False)
    pretty = table[["model", "config", "selected"]].copy()
    for m, label in METRICS.items():
        pretty[label] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(table[f"{m}_mean"], table[f"{m}_std"])]
    pretty.to_csv(out_dir / "before_after_formatted.csv", index=False, encoding="utf-8-sig")
    _plot(table, out_dir / "before_after.png", task)
    return pretty


def _plot(table, path, task):
    models = sorted(table["model"].unique())
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.8), sharey=True)
    for ax, (m, label) in zip(axes, METRICS.items()):
        x = np.arange(len(models))
        for k, (config, colour) in enumerate((("before tuning (default)", "#A0A0A0"), ("after tuning", "#4C72B0"))):
            t = table[table["config"] == config].set_index("model").reindex(models)
            ax.bar(x + (k - 0.5) * 0.38, t[f"{m}_mean"], 0.38, yerr=t[f"{m}_std"], capsize=3, color=colour,
                   label=config)
        ax.set_xticks(x, models, rotation=15, fontsize=8)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Image new - {task}: test scores before vs after hyperparameter tuning (mean ± std over seeds)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--default", default=str(NEW_DIR / "results_default"))
    p.add_argument("--tuned", default=str(NEW_DIR / "results_tuned"))
    p.add_argument("--out", default=str(NEW_DIR / "results" / "comparison"))
    a = p.parse_args()
    print(run(Path(a.default), Path(a.tuned), Path(a.out)).to_string(index=False))


if __name__ == "__main__":
    main()
