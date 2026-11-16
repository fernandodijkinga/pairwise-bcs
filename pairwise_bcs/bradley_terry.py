"""Bradley-Terry latent scale from pairwise comparisons.

Model: P(i beats j) = sigmoid(beta_i - beta_j).
The scores are estimated by maximum a posteriori (MAP) estimation with a
Gaussian (ridge) prior, which fixes the otherwise arbitrary scale and keeps
the estimates finite for images that won or lost all their comparisons.
Settings follow Section 2.2 of the article (ridge lambda = 0.05, L-BFGS).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class BTResult:
    ids: list            # image ids, in the order of `beta`
    beta: np.ndarray     # latent scores, centred at zero
    se: np.ndarray       # standard errors from the Fisher information
    reliability: float   # 1 - mean(SE^2) / var(beta)
    n_comparisons: int

    def as_dict(self) -> dict:
        """{image_id: score}."""
        return dict(zip(self.ids, self.beta.tolist()))


def fit_bradley_terry(comparisons, ridge: float = 0.05, max_iter: int = 10000) -> BTResult:
    """Fit a ridge-regularised Bradley-Terry model.

    Parameters
    ----------
    comparisons : iterable of (winner_id, loser_id)
    ridge : strength of the Gaussian prior, penalty = ridge / 2 * sum(beta^2)
    """
    comparisons = list(comparisons)
    ids = sorted({x for pair in comparisons for x in pair}, key=str)
    index = {k: i for i, k in enumerate(ids)}

    # Count identical directed outcomes once, with a weight.
    counts = Counter((index[w], index[l]) for w, l in comparisons)
    pairs = np.array(list(counts.keys()), dtype=np.int64)
    weights = np.array(list(counts.values()), dtype=np.float64)
    win, lose = pairs[:, 0], pairs[:, 1]
    lam = max(float(ridge), 1e-4)

    def neg_log_posterior(beta):
        diff = beta[win] - beta[lose]
        # -log sigmoid(diff) = log(1 + exp(-diff)), computed stably
        return float(np.sum(weights * np.logaddexp(0.0, -diff)) + 0.5 * lam * np.sum(beta ** 2))

    def gradient(beta):
        diff = beta[win] - beta[lose]
        p_loss = 1.0 / (1.0 + np.exp(diff))          # probability of the observed outcome NOT happening
        g = weights * p_loss
        grad = lam * beta
        np.add.at(grad, win, -g)
        np.add.at(grad, lose, g)
        return grad

    res = minimize(neg_log_posterior, np.zeros(len(ids)), jac=gradient, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-11, "gtol": 1e-8})
    beta = res.x - res.x.mean()

    # Diagonal of the Fisher information -> approximate standard errors.
    diff = beta[win] - beta[lose]
    p = 1.0 / (1.0 + np.exp(-diff))
    w_ij = weights * p * (1.0 - p)
    info = np.full(len(ids), lam)
    np.add.at(info, win, w_ij)
    np.add.at(info, lose, w_ij)
    se = 1.0 / np.sqrt(info)

    var_beta = float(np.var(beta))
    reliability = max(var_beta - float(np.mean(se ** 2)), 0.0) / max(var_beta, 1e-12)
    return BTResult(ids=ids, beta=beta, se=se, reliability=reliability, n_comparisons=len(comparisons))


def in_sample_consistency(comparisons, scores: dict) -> float:
    """Share of comparisons whose winner has the higher fitted score.

    Note: this is computed on the same comparisons used to fit the scale, so it
    measures the fit of the model to the data, not annotator reliability.
    """
    ok = [scores[w] > scores[l] for w, l in comparisons if w in scores and l in scores]
    return float(np.mean(ok)) if ok else float("nan")


def win_probability(score_a: float, score_b: float) -> float:
    """P(a beats b) under the Bradley-Terry model."""
    return float(1.0 / (1.0 + np.exp(-(score_a - score_b))))
