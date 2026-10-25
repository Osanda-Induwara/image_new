"""Data augmentation for image_new: geometric only (rotation, flip), as the
development plan requires (no colour changes: liquor colour is the signal).

Two uses:
  * online_augment(): random horizontal/vertical flip and +/-10 degree rotation,
    applied to every training photo every epoch;
  * balance_training_set(): grade balancing. The grades with fewer training
    photos get augmented copies until every grade has as many training photos as
    the largest grade (BOPF). Copies are made ONLY from the training photos of
    the current split and keep their source photo's sample key, so they can never
    reach validation or test (no leakage). Validation and test keep only real
    photos.

Copies of a photo are, in order: the 7 lossless dihedral variants of the disc
(90/180/270 degree rotations, the mirror image and its rotations), then seeded
random-angle rotations (with a random flip) if even more copies are needed. The
crops are grey-masked circles (image/preprocess_images.py --crop masked), so
rotating them never brings bowl or background into view; the uncovered corners
are filled with the same grey.

Report (counts before / after balancing per seed, example grid):
    python -m unimodels.image_new.augmentation --data-root <dataset>
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torchvision import transforms

from ..common import config as C
from ..image.preprocess_images import MASKED_PARAMS

GREY_RGB = tuple(reversed(MASKED_PARAMS["fill_bgr"]))     # crops are cached as RGB
N_DIHEDRAL = 7


def online_augment():
    """Random geometric augmentation applied every epoch to the (balanced) training set."""
    return [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10, fill=list(GREY_RGB))]


def variant(img, v, rng):
    """Copy number v (1, 2, ...) of an RGB square crop: dihedral variants first, then random rotations."""
    if v <= N_DIHEDRAL:
        out = img[:, ::-1] if v >= 4 else img                 # 4-7: mirror image ...
        return np.ascontiguousarray(np.rot90(out, k=v % 4))   # ... and the 0/90/180/270 rotations
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-180, 180)), 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=GREY_RGB)
    return out[:, ::-1].copy() if rng.random() < 0.5 else out


def balance_training_set(arr, idx, y, keys, seed):
    """Balance the training photos of one split by grade.

    arr: all cached crops (N x S x S x 3, uint8); idx / y / keys: the training photos' rows, labels and sample keys.
    Returns (arr_b, y_b, keys_b, info): the originals followed by the augmented copies, and a table with one row per
    photo (source row in arr, copy number, 0 = original). Every grade ends with max(count) photos.
    """
    idx, y, keys = np.asarray(idx), np.asarray(y), np.asarray(keys)
    counts = Counter(y.tolist())
    target = max(counts.values())
    rng = np.random.default_rng(seed)
    imgs, ys, ks, src, ver = [arr[idx]], [y], [keys], [idx], [np.zeros(len(idx), int)]
    for c in sorted(counts):
        members = np.where(y == c)[0]
        need = target - len(members)
        if need <= 0:
            continue
        order = rng.permutation(len(members))
        copies = []
        for j in range(need):
            m = members[order[j % len(members)]]
            v = 1 + j // len(members)
            copies.append(variant(arr[idx[m]], v, rng))
            src.append([idx[m]])
            ver.append([v])
            ys.append([c])
            ks.append([keys[m]])
        imgs.append(np.stack(copies))
    arr_b = np.concatenate(imgs)
    info = pd.DataFrame({"source_row": np.concatenate(src), "copy": np.concatenate(ver), "label": np.concatenate(ys)})
    return arr_b, np.concatenate(ys), np.concatenate(ks), info


def class_counts(y, classes, info=None):
    """Photos per class; with info, split into originals and copies."""
    rows = []
    for i, c in enumerate(classes):
        if info is None:
            rows.append({"grade": c, "photos": int((np.asarray(y) == i).sum())})
        else:
            sel = info["label"].to_numpy() == i
            rows.append({"grade": c, "originals": int((info["copy"].to_numpy()[sel] == 0).sum()),
                         "copies": int((info["copy"].to_numpy()[sel] > 0).sum()), "photos": int(sel.sum())})
    return pd.DataFrame(rows)


def save_examples(arr, row, path, rng_seed=0, n=8):
    """A strip: one training crop and its first n-1 copies."""
    rng = np.random.default_rng(rng_seed)
    tiles = [arr[row]] + [variant(arr[row], v, rng) for v in range(1, n)]
    tiles = [cv2.resize(t, (160, 160), interpolation=cv2.INTER_AREA) for t in tiles]
    for i, t in enumerate(tiles):
        cv2.putText(t, "original" if i == 0 else f"copy {i}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))


def report(data_root, out_dir, seeds, max_per_sample=None):
    """Counts before / after balancing for every seed (training split), plus an example strip."""
    from sklearn.preprocessing import LabelEncoder

    from ..common.cnn_train import build_parts, load_crops
    from ..common.samples import build_sample_table
    from ..common.splits import load_or_make_split
    from .train_image_new import DEFAULT_BRANCH

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, _ = build_sample_table(data_root)
    man, arr = load_crops(DEFAULT_BRANCH, samples["sample_key"], C.IMAGE_SIZE, max_per_sample)
    enc = LabelEncoder().fit(samples["grade"])
    y_all = enc.transform(man["sample_key"].map(dict(zip(samples["sample_key"], samples["grade"]))))
    rows = []
    for seed in seeds:
        parts = build_parts(man, y_all, load_or_make_split(samples, seed))
        tr = parts["train"]
        _, y_b, _, info = balance_training_set(arr, tr["idx"], tr["y"], tr["keys"], seed)
        t = class_counts(y_b, enc.classes_, info)
        for s in ("val", "test"):
            t[f"{s}_photos (real only)"] = class_counts(parts[s]["y"], enc.classes_)["photos"]
        t.insert(0, "seed", seed)
        rows.append(t)
        if seed == seeds[0]:
            smallest = int(np.argmin(np.bincount(tr["y"], minlength=len(enc.classes_))))
            save_examples(arr, int(tr["idx"][tr["y"] == smallest][0]), out_dir / "examples.jpg")
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out_dir / "class_counts.csv", index=False)
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(C.DATA_ROOT))
    p.add_argument("--out", default=str(C.UNIMODELS_DIR / "image_new" / "results" / "augmentation"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    a = p.parse_args()
    t = report(Path(a.data_root), Path(a.out), a.seeds)
    print(t[t["seed"] == a.seeds[0]].to_string(index=False))
    print(f"Written to {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
