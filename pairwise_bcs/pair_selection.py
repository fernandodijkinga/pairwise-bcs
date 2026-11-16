"""Adaptive pair selection (reference implementation of Section 2.1).

At each annotation step the most informative candidate pair is proposed to the
annotator, who then chooses which of the two images shows greater body
condition. "Most informative" is the pair with the highest value of EITHER of
two criteria (no weighted combination):

  1. Entropy of the current Bradley-Terry win probability,
         H(p) = -p log2 p - (1 - p) log2 (1 - p),
     which is maximal (1 bit) when the two images are hardest to tell apart.
  2. Mean uncertainty of the two images, each tracked by a Beta distribution
     Beta(1 + wins, 1 + losses) updated after every comparison (here, its
     standard deviation).

A repetition penalty proportional to the number of previous comparisons of the
same pair discourages redundant queries. At the start, when all win
probabilities are close to 0.5, the entropy is maximal for every pair and the
uncertainty criterion (which favours images with few comparisons) drives the
selection.

The annotation interface used in the study is not part of this repository;
this module reproduces the selection criterion so that it can be reused.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from .bradley_terry import fit_bradley_terry, win_probability


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class BetaTracker:
    """Per-image Beta(1 + wins, 1 + losses) uncertainty."""

    def __init__(self, image_ids):
        self.wins = defaultdict(int)
        self.losses = defaultdict(int)
        for i in image_ids:
            self.wins[i]; self.losses[i]

    def update(self, winner, loser):
        self.wins[winner] += 1
        self.losses[loser] += 1

    def uncertainty(self, image_id) -> float:
        a, b = 1 + self.wins[image_id], 1 + self.losses[image_id]
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))   # SD of Beta(a, b)


class AdaptivePairSelector:
    """Propose informative pairs; record the annotator's decisions.

    Example
    -------
    >>> sel = AdaptivePairSelector(image_ids)
    >>> a, b = sel.next_pair()
    >>> sel.record(winner=a, loser=b)        # decision made by the annotator
    """

    def __init__(self, image_ids, n_candidates: int = 500, refit_every: int = 200,
                 repetition_penalty: float = 0.5, ridge: float = 0.05, seed: int = 0):
        self.ids = list(image_ids)
        self.n_candidates = n_candidates          # random candidate pairs scored per step
        self.refit_every = refit_every            # refit the BT scale every N decisions
        self.repetition_penalty = repetition_penalty
        self.ridge = ridge
        self.rng = random.Random(seed)
        self.beta = BetaTracker(self.ids)
        self.history = []                         # list of (winner, loser)
        self.pair_counts = Counter()
        self.scores = {i: 0.0 for i in self.ids}  # BT scores (all equal before any data)

    def _criteria(self, a, b):
        entropy = binary_entropy(win_probability(self.scores[a], self.scores[b]))
        uncertainty = 0.5 * (self.beta.uncertainty(a) + self.beta.uncertainty(b))
        # Put both criteria on [0, 1]: entropy is in bits; the Beta SD is at most 1/sqrt(12).
        return entropy, uncertainty * math.sqrt(12)

    def next_pair(self):
        best, best_value = None, -float("inf")
        for _ in range(self.n_candidates):
            a, b = self.rng.sample(self.ids, 2)
            entropy, uncertainty = self._criteria(a, b)
            value = max(entropy, uncertainty)                             # either criterion
            value -= self.repetition_penalty * self.pair_counts[frozenset((a, b))]
            if value > best_value:
                best, best_value = (a, b), value
        return best

    def record(self, winner, loser):
        self.history.append((winner, loser))
        self.pair_counts[frozenset((winner, loser))] += 1
        self.beta.update(winner, loser)
        if len(self.history) % self.refit_every == 0:
            fit = fit_bradley_terry(self.history, ridge=self.ridge)
            self.scores.update(fit.as_dict())
