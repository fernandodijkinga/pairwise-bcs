"""Data partitions used for evaluation (Section 2.7 of the article).

1. Image-disjoint test (unseen animals): 15% of the images, stratified by the
   ordinal score, form the test block. Every comparison involving a test image
   is removed from training. Comparisons between two test images form the
   pairwise test set; comparisons linking a test and a training image are
   discarded. The remaining comparisons are split at the comparison level into
   training and validation (validation = unseen relations, used only for model
   selection).
2. Bridge-Holdout (unseen relations): the comparison graph is split into two
   communities by breadth-first search from a random node, and every comparison
   between them is withheld from training.
3. Edge-Holdout (extrapolation): every comparison involving an image in the
   lower or upper 5% tail of the Bradley-Terry scale is withheld, so those
   images receive no supervision.

For the structural holdouts, the withheld comparisons are excluded from
training but kept in the full Bradley-Terry reference used for evaluation, so
that reference is not independent of them.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def image_disjoint_split(comparisons, ordinal: dict, seed: int, test_fraction: float = 0.15,
                         val_fraction: float = 0.15, n_bins: int = 10) -> dict:
    """comparisons: list of (winner, loser); ordinal: {image_id: BCS}."""
    rng = np.random.default_rng(seed)
    ids = sorted({x for c in comparisons for x in c} & set(ordinal), key=str)
    values = np.array([ordinal[i] for i in ids])
    # Stratify by quantile bins of the ordinal score.
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(values, edges)
    test = set()
    for b in np.unique(bins):
        members = [ids[k] for k in np.where(bins == b)[0]]
        n = int(round(test_fraction * len(members)))
        test.update(rng.choice(members, size=n, replace=False).tolist())
    train_images = [i for i in ids if i not in test]

    test_pairs, inner, discarded = [], [], 0
    for w, l in comparisons:
        a, b = w in test, l in test
        if a and b:
            test_pairs.append((w, l))
        elif a or b:
            discarded += 1
        else:
            inner.append((w, l))
    order = rng.permutation(len(inner))
    n_val = int(round(val_fraction * len(inner)))
    val_pairs = [inner[k] for k in order[:n_val]]
    train_pairs = [inner[k] for k in order[n_val:]]
    return {"train_images": train_images, "test_images": sorted(test, key=str),
            "train_pairs": train_pairs, "val_pairs": val_pairs, "test_pairs": test_pairs,
            "n_discarded": discarded}


def bridge_holdout(comparisons, seed: int) -> dict:
    """Split the graph in two BFS communities and withhold all edges between them."""
    adj = defaultdict(set)
    for w, l in comparisons:
        adj[w].add(l); adj[l].add(w)
    nodes = sorted(adj, key=str)
    rng = random.Random(seed)
    start = rng.choice(nodes)
    target = len(nodes) // 2
    seen, queue = {start}, [start]
    while queue and len(seen) < target:
        node = queue.pop(0)
        for nb in sorted(adj[node], key=str):
            if nb not in seen:
                seen.add(nb); queue.append(nb)
                if len(seen) >= target:
                    break
    part_a = seen
    train, held = [], []
    for w, l in comparisons:
        (held if (w in part_a) != (l in part_a) else train).append((w, l))
    return {"train_pairs": train, "held_pairs": held, "community_a": sorted(part_a, key=str)}


def edge_holdout(comparisons, bt_scores: dict, quantile: float = 0.05) -> dict:
    """Withhold every comparison that involves an image in the 5% tails of the BT scale."""
    ids = sorted(bt_scores, key=str)
    vals = np.array([bt_scores[i] for i in ids])
    lo, hi = np.quantile(vals, [quantile, 1 - quantile])
    extremes = {i for i in ids if bt_scores[i] <= lo or bt_scores[i] >= hi}
    train, held = [], []
    for w, l in comparisons:
        (held if (w in extremes or l in extremes) else train).append((w, l))
    still_seen = extremes & {x for c in train for x in c}
    assert not still_seen, "extreme images must not appear in training"
    return {"train_pairs": train, "held_pairs": held, "extremes": sorted(extremes, key=str)}
