# image_new: grade from masked-circle crops, balanced by augmentation, tuned transfer learning

Built 2026-10-26 at the user's request. **Grade only** (the main focus). Run `train_image_new.ipynb` on a Colab
GPU: a FAST smoke test first, then the real run, about 6 GPU-hours, resumable.

## What it does

1. **Crops:** the grey-masked circle from `image/preprocess_images.py --crop masked`: the whole liquor, outside
   painted the ImageNet-mean grey.
2. **Augmentation** (`augmentation.py`, geometric only, as the development plan requires):
   - `online_augment()`: random horizontal and vertical flip and ±10° rotation, every epoch;
   - `balance_training_set()`: the grades with fewer training photos get augmented copies until every grade has as
     many training photos as the largest grade (BOPF). Copies are, in order: the 7 lossless variants of the disc
     (90/180/270° rotations, the mirror image and its rotations), then seeded random-angle rotations.
   - **No leakage:** copies are made only from the training photos of each split and keep their source photo's
     sample key. Validation and test keep only real photos, evaluated per physical sample.
3. **Models** (`models/`, one file each, ImageNet-pretrained): `efficientnet_b0.py`, `resnet18.py`,
   `mobilenet_v2.py`. They are trained in two stages: head only on a frozen backbone, then the whole network
   (AdamW), with early stopping on validation loss.
4. **Before vs after tuning:**
   - before: the default hyperparameters, i.e. the official image recipe (`models/common.py`);
   - tuning (`tune.py`): 12 random trials per model; trial 0 is the default, so tuning can only match or beat it
     on validation. Each trial runs on seeds 42-44 and is scored on **validation only**: pooled macro-F1 with the
     Brier tie-break. Search space: learning rates, weight decay, dropout, label smoothing, batch size, head-only
     epochs, patience;
   - after: the best trial's setting, retrained on all 10 seeds;
   - `compare.py` reports test accuracy, macro precision, recall and F1 for both.
5. **Learning curve** (`learning_curve.py`): the best tuned model, trained on 20-100% of the training samples, each
   subset balanced the same way, seeds 42-44. It plots training samples vs train / validation / test accuracy and
   macro-F1.

## Outputs

```
results_default/grade/   default hyperparameters: selection, summary_by_model, reports, epoch curves, Grad-CAM
results_tuned/grade/     tuned hyperparameters: the same files
results/augmentation/    class_counts.csv (per seed: originals, copies, after balancing; val/test real only), examples.jpg
results/tuning/          trials.csv, <model>_trial_ranking.csv, best_hparams.json, one file per trial
results/comparison/      before_after.csv, before_after_formatted.csv, before_after.png
results/learning_curve/  learning_curve.csv, learning_curve.png
```

## Notes for the thesis

- **Copies are not new samples.** Balancing equalises the number of training *photos* per grade, but the
  augmented copies come from the same physical samples (OP still has only 11 samples). It reduces bias towards
  BOPF; it does not add new information.
- **Tuning seeds are also evaluation seeds.** Tuning used the validation data of seeds 42-44, which are also among
  the 10 evaluation seeds. Their validation scores are therefore slightly optimistic. Test data were never used
  for any choice, so the test results stay honest.

## Results

Not run yet.
