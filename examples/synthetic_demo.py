"""End-to-end demonstration on synthetic data (runs on a CPU in a few minutes).

The images of the study cannot be distributed, so this script creates a small
synthetic dataset: each "animal" is a drawn shape whose width depends on a
hidden trait. It then runs every step of the method:

  1. adaptive pair selection with a simulated annotator,
  2. Bradley-Terry scale estimation,
  3. image-disjoint split and structural holdouts,
  4. training of a pairwise (Bradley-Terry) and an ordinal (CORAL) model,
  5. evaluation against the ordinal score and a paired bootstrap.

It uses the small "tiny" CNN backbone and a short schedule. The numbers it
prints say nothing about body condition; they only show that the code runs.

    python examples/synthetic_demo.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairwise_bcs.bradley_terry import fit_bradley_terry, in_sample_consistency, win_probability  # noqa: E402
from pairwise_bcs.data import ImageStore  # noqa: E402
from pairwise_bcs.evaluate import evaluate_run, paired_bootstrap  # noqa: E402
from pairwise_bcs.pair_selection import AdaptivePairSelector  # noqa: E402
from pairwise_bcs.splits import bridge_holdout, edge_holdout, image_disjoint_split  # noqa: E402
from pairwise_bcs.train import Config, score_images, train  # noqa: E402

OUT = Path(__file__).resolve().parent / "synthetic"
N_IMAGES, DECISIONS_PER_IMAGE, SEED = 160, 10, 0


def make_dataset():
    rng = np.random.default_rng(SEED)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    trait = rng.normal(0, 1, N_IMAGES)
    ids = [f"img{k:03d}" for k in range(N_IMAGES)]
    for i, t in zip(ids, trait):
        img = Image.new("RGB", (224, 224), tuple(int(c) for c in rng.integers(90, 160, 3)))
        d = ImageDraw.Draw(img)
        half_width = 55 + 18 * t + rng.normal(0, 4)          # "body condition" -> width
        d.ellipse([112 - half_width, 40, 112 + half_width, 200], fill=(30, 30, 30))
        d.rectangle([90, 60, 134, 200], fill=(235, 235, 235))
        img.save(OUT / "images" / f"{i}.png")
    # ordinal score of the same "expert": noisy and rounded to 0.25
    bcs = np.clip(np.round((3.0 + 0.6 * trait + rng.normal(0, 0.15, N_IMAGES)) / 0.25) * 0.25, 1, 5)
    with open(OUT / "ordinal.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "bcs"]); w.writerows(zip(ids, bcs))

    # simulated annotator answering the pairs proposed by the adaptive selector
    selector = AdaptivePairSelector(ids, n_candidates=200, refit_every=100, seed=SEED)
    true = dict(zip(ids, 2.0 * trait))
    for _ in range(N_IMAGES * DECISIONS_PER_IMAGE // 2):     # 10 comparisons per image
        a, b = selector.next_pair()
        if rng.random() < win_probability(true[a], true[b]):
            selector.record(winner=a, loser=b)
        else:
            selector.record(winner=b, loser=a)
    with open(OUT / "comparisons.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["winner_id", "loser_id"]); w.writerows(selector.history)
    return ids, selector.history, dict(zip(ids, bcs.tolist()))


def main():
    ids, comps, ordinal = make_dataset()
    print(f"{len(comps)} decisions; {2 * len(comps) / len(ids):.1f} comparisons per image "
          f"({len(comps) / len(ids):.1f} decisions per image)")

    bt = fit_bradley_terry(comps)
    scores = bt.as_dict()
    from scipy.stats import spearmanr
    print(f"BT: reliability {bt.reliability:.3f}, in-sample consistency {in_sample_consistency(comps, scores):.3f}, "
          f"Spearman with ordinal {spearmanr([scores[i] for i in ids], [ordinal[i] for i in ids])[0]:.3f}")

    bridge = bridge_holdout(comps, seed=SEED)
    edge = edge_holdout(comps, scores)
    print(f"Bridge-Holdout withholds {len(bridge['held_pairs'])} comparisons; "
          f"Edge-Holdout withholds {len(edge['held_pairs'])} ({len(edge['extremes'])} extreme images)")

    split = image_disjoint_split(comps, ordinal, seed=SEED)
    store = ImageStore(OUT / "images", ids)
    cfg = Config(backbone="tiny", phase_epochs=(2, 2, 4), batch_size=16, eval_every=2, seed=SEED)
    runs = {}
    for objective in ("pairwise", "ordinal"):
        print(f"\n--- training {objective} model ---")
        model, history = train(store, objective, split["train_pairs"], split["val_pairs"], ordinal,
                               split["train_images"], cfg, device="cpu")
        sc = score_images(model, store, split["train_images"] + split["test_images"], "cpu")
        runs[objective] = {"scores": sc, "split": split, "history": history}
        print(objective, evaluate_run(runs[objective], ordinal))
    print("\npaired bootstrap (pairwise - ordinal):",
          paired_bootstrap(runs["pairwise"]["scores"], runs["ordinal"]["scores"], ordinal,
                           split["test_images"], n_boot=500))


if __name__ == "__main__":
    main()
