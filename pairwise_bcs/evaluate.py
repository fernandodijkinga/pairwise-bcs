"""Evaluation on unseen animals (Sections 2.7 and 2.11 of the article).

* Rank agreement with the ordinal BCS (Spearman rho, Kendall tau): no calibration,
  no BCS label is used for fitting. This is the primary measure.
* BCS units, two calibrations of the linear mapping BCS = a + b * s:
    - target-population: 10-fold cross-validation over the test animals;
      QWK categories from the test-set BCS distribution (quantile matching);
    - training-fixed: mapping and category thresholds estimated on the training
      animals and applied unchanged to the test animals (no test label used).
* Paired bootstrap over test animals (2,000 resamples, same animals for both
  models) for the difference in Spearman rho between two models.

Command line:
    python -m pairwise_bcs.evaluate --ordinal ordinal.csv --run runs/pairwise_s42 \
        [--compare runs/ordinal_s42]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score

from .data import LEVELS, level_index, load_ordinal


def qwk(true_idx, pred_idx):
    return float(cohen_kappa_score(true_idx, pred_idx, weights="quadratic", labels=list(range(len(LEVELS)))))


def quantile_categories(s_fit, idx_fit, s_apply):
    """Thresholds on the score chosen so that category frequencies match the fit set."""
    freq = np.bincount(idx_fit, minlength=len(LEVELS)).astype(float)
    cut = np.cumsum(freq / freq.sum())[:-1]
    return np.digitize(s_apply, np.quantile(s_fit, np.clip(cut, 0, 1)))


def calibration_cv(s, y, folds=10, seed=0):
    """Out-of-fold predictions of the linear mapping BCS = a + b * s."""
    fold = np.random.default_rng(seed).permutation(len(s)) % folds
    pred = np.empty_like(y, dtype=float)
    for f in range(folds):
        b, a = np.polyfit(s[fold != f], y[fold != f], 1)
        pred[fold == f] = a + b * s[fold == f]
    return pred


def bcs_metrics(pred, y):
    res = y - pred
    return {"mae": float(np.mean(np.abs(res))), "rmse": float(np.sqrt(np.mean(res ** 2))),
            "within_0.25": float(np.mean(np.abs(res) <= 0.25))}


def evaluate_run(run: dict, ordinal: dict) -> dict:
    sc, split = run["scores"], run["split"]
    te = [i for i in split["test_images"] if i in ordinal]
    tr = [i for i in split["train_images"] if i in ordinal]
    s_te = np.array([sc[i] for i in te]); y_te = np.array([ordinal[i] for i in te])
    s_tr = np.array([sc[i] for i in tr]); y_tr = np.array([ordinal[i] for i in tr])
    i_te = np.array([level_index(v) for v in y_te]); i_tr = np.array([level_index(v) for v in y_tr])
    b, a = np.polyfit(s_tr, y_tr, 1)
    out = {
        "n_test": len(te),
        "spearman_bcs": float(spearmanr(s_te, y_te)[0]),
        "kendall_bcs": float(kendalltau(s_te, y_te)[0]),
        "target_population": {**bcs_metrics(calibration_cv(s_te, y_te), y_te),
                              "qwk": qwk(i_te, quantile_categories(s_te, i_te, s_te))},
        "training_fixed": {**bcs_metrics(a + b * s_te, y_te),
                           "qwk": qwk(i_te, quantile_categories(s_tr, i_tr, s_te))},
    }
    pairs = [(w, l) for w, l in split["test_pairs"] if w in sc and l in sc]
    if pairs:
        out["test_pair_agreement"] = float(np.mean([sc[w] > sc[l] for w, l in pairs]))
    return out


def paired_bootstrap(scores_a: dict, scores_b: dict, ordinal: dict, ids, n_boot=2000, seed=0):
    """Difference in Spearman rho with BCS (model A - model B), resampling test animals."""
    ids = [i for i in ids if i in ordinal and i in scores_a and i in scores_b]
    sa = np.array([scores_a[i] for i in ids]); sb = np.array([scores_b[i] for i in ids])
    y = np.array([ordinal[i] for i in ids])
    delta = spearmanr(sa, y)[0] - spearmanr(sb, y)[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(ids), len(ids))
        boots.append(spearmanr(sa[k], y[k])[0] - spearmanr(sb[k], y[k])[0])
    boots = np.array(boots)
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"delta_rho": float(delta), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p": float(min(p, 1.0)), "n_animals": len(ids), "n_boot": n_boot}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ordinal", required=True)
    ap.add_argument("--run", required=True, help="folder written by pairwise_bcs.train")
    ap.add_argument("--compare", help="second run folder (same seed) for the paired bootstrap")
    args = ap.parse_args()
    ordinal = load_ordinal(args.ordinal)
    run = json.load(open(f"{args.run}/run.json"))
    result = {"run": evaluate_run(run, ordinal)}
    if args.compare:
        other = json.load(open(f"{args.compare}/run.json"))
        result["compare"] = evaluate_run(other, ordinal)
        result["paired_bootstrap"] = paired_bootstrap(run["scores"], other["scores"], ordinal, run["split"]["test_images"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
