# pairwise-bcs

Reference code for the article

> Dijkinga, F.J., Garrido, L.F., Daros, R., Vale, M., Laroca dos Santos, R.B.
> **A Continuous Geometry of Dairy Cow Body Condition: Pairwise Comparisons versus Ordinal Scores as Supervision for Deep Vision Models.**
> *Computers and Electronics in Agriculture* (in revision).

The study asks whether a vision model learns body condition better from **pairwise comparisons**
("which of these two cows has greater body condition?") than from conventional **ordinal body
condition scores (BCS)**, when both are given by the same expert for the same animals. This
repository implements the methodology:

| Step | Module | Article |
|---|---|---|
| Adaptive selection of informative pairs | `pairwise_bcs/pair_selection.py` | Section 2.1 |
| Bradley-Terry latent scale (ridge MAP, Fisher SE, reliability) | `pairwise_bcs/bradley_terry.py` | Section 2.2 |
| Image-disjoint test, Bridge-Holdout, Edge-Holdout | `pairwise_bcs/splits.py` | Section 2.7 |
| DINOv2 ViT-B/14 + attention pooling; Bradley-Terry and CORAL objectives | `pairwise_bcs/model.py` | Sections 2.4-2.5 |
| Three-phase fine-tuning | `pairwise_bcs/train.py` | Section 2.6, Table S1 |
| Rank agreement, calibration to BCS units, QWK, paired bootstrap | `pairwise_bcs/evaluate.py` | Section 2.11 |

## What is not included

* **Images and annotations.** The images were collected on commercial dairy farms whose owners
  authorised their use for this research but not their release to third parties, to protect the
  privacy of the farms and producers. They cannot be shared.
* **The annotation interface** used to collect the comparisons. `pair_selection.py` is a reference
  implementation of its pair-selection criterion: the most informative pair is the one with the
  highest value of either the entropy of the Bradley-Terry win probability or the mean Beta
  uncertainty of the two images, with a penalty for repeated pairs.
* **Object detection.** Images are expected to be already cropped to the animal (the study used a
  YOLO detector for the posterior region).

The code is a cleaned version of the research pipeline, with the same methods and parameters as
the article. Results on other data will differ.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.1+. The DINOv2 backbone is downloaded through `torch.hub` on first use;
a GPU is recommended for training (the study used an NVIDIA RTX 3060, 12 GB).

## Quick start (synthetic data, CPU, ~2 min)

```bash
python examples/synthetic_demo.py
```

This creates a small synthetic dataset (drawn shapes whose width depends on a hidden trait),
simulates an annotator answering adaptively selected pairs, fits the Bradley-Terry scale, builds
the evaluation splits, trains a pairwise and an ordinal model with a small CNN backbone, and
evaluates them. It only shows that every step runs; its numbers say nothing about cattle.

## Your own data

```
comparisons.csv   winner_id,loser_id     one row per pairwise decision
ordinal.csv       image_id,bcs           ordinal BCS (1-5 in 0.25 steps)
images/           <image_id>.jpg | .png  images cropped to the animal
```

Train and evaluate on the image-disjoint protocol (15% of animals held out, stratified by BCS):

```bash
python -m pairwise_bcs.train --images images --comparisons comparisons.csv --ordinal ordinal.csv \
    --objective pairwise --seed 42 --out runs/pairwise_s42
python -m pairwise_bcs.train --images images --comparisons comparisons.csv --ordinal ordinal.csv \
    --objective ordinal --seed 42 --out runs/ordinal_s42
python -m pairwise_bcs.evaluate --ordinal ordinal.csv --run runs/pairwise_s42 --compare runs/ordinal_s42
```

Add `--budget-matched` to train the pairwise model on as many comparisons as there are ordinal
training scores. The article used seeds 42, 123 and 2026.

Fit the Bradley-Terry scale only:

```python
from pairwise_bcs.data import load_comparisons
from pairwise_bcs.bradley_terry import fit_bradley_terry
bt = fit_bradley_terry(load_comparisons("comparisons.csv"), ridge=0.05)
print(bt.reliability, bt.as_dict())
```

## Definitions used in the article

* **Comparisons per image**: number of comparisons in which an image took part, counting each
  pairwise decision for both of its images (2 x decisions / images). **Decisions per image** is half
  this value.
* **BT latent score**: scale estimated from the comparisons. **Model score**: continuous output
  s(x) of a trained model. **Calibrated BCS estimate**: model score mapped to BCS units.
* **Target-population calibration**: linear mapping fitted by 10-fold cross-validation over the
  test animals, QWK categories from the test-set BCS distribution. **Training-fixed calibration**:
  mapping and thresholds estimated on the training animals only.

## Main configuration (Table S1)

| Setting | Value |
|---|---|
| Backbone | DINOv2 ViT-B/14, 224 x 224 input, ImageNet normalisation |
| Head | 8-head attention pooling + [CLS] fusion; LayerNorm - Linear(768-256) - GELU - Linear(256-1) |
| Losses | Bradley-Terry NLL (pairwise); CORAL, 16 thresholds (ordinal) |
| Schedule | 30 epochs: 5 (backbone frozen) - 5 (last 2 blocks) - 20 (last 6 blocks + final norm) |
| Learning rates | head 3e-4 / 2e-4 / 1e-4; backbone 0 / 2e-5 / 1e-5 |
| Optimiser | AdamW, weight decay 0.05, 5% linear warm-up, cosine decay, batch 64 |
| Augmentation | horizontal flip; brightness gain U(0.8, 1.2), offset U(-0.02, 0.02) |
| Model selection | every 2 epochs; best validation pair agreement (pairwise) or Spearman with BCS (ordinal) |
| Bradley-Terry | ridge 0.05, L-BFGS-B |

## Citation

If you use this code, please cite the article above.

## License

MIT (see `LICENSE`).
