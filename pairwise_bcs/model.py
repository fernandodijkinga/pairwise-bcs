"""Scoring model (Sections 2.4-2.5 of the article).

Backbone: DINOv2 ViT-B/14 (self-supervised pre-training). The patch tokens are
aggregated by a learned 8-head attention-pooling layer, averaged with the mean
patch token, fused with the [CLS] token, and mapped to a scalar score s(x) by
LayerNorm -> Linear(768 -> 256) -> GELU -> Linear(256 -> 1).

Two training objectives share this model:
  * pairwise: Bradley-Terry loss,  P(A > B) = sigmoid(s_A - s_B)
  * ordinal : CORAL head, 16 ordered thresholds on the same scalar s(x)

A small CNN backbone ("tiny") is provided only to run the synthetic example
quickly on a CPU; it is not the model used in the article.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LEVELS = 17          # BCS 1.0 to 5.0 in 0.25 steps
HIDDEN = 256


class AttentionPool(nn.Module):
    """Single learned query attending over the patch tokens, fused with [CLS]."""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.q = nn.Parameter(torch.randn(1, heads, 1, self.dh) * 0.02)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, tokens, cls):
        b, n, d = tokens.shape
        k = self.k(tokens).view(b, n, self.heads, self.dh).transpose(1, 2)
        v = self.v(tokens).view(b, n, self.heads, self.dh).transpose(1, 2)
        attn = torch.softmax((self.q * k).sum(-1) / math.sqrt(self.dh), dim=-1)   # [b, heads, n]
        pooled = self.proj(torch.einsum("bhn,bhnd->bhd", attn, v).reshape(b, d))
        merged = 0.5 * pooled + 0.5 * tokens.mean(dim=1)
        return self.fuse(torch.cat([merged, cls], dim=-1)), attn.mean(1)


class CoralHead(nn.Module):
    """CORAL: logit_k = s(x) - theta_k with theta_1 < ... < theta_16 (softplus increments)."""

    def __init__(self, n_thresholds: int = N_LEVELS - 1):
        super().__init__()
        self.b0 = nn.Parameter(torch.zeros(1))
        self.deltas = nn.Parameter(torch.full((n_thresholds - 1,), -1.0))

    def thresholds(self):
        return torch.cat([self.b0, self.b0 + torch.cumsum(F.softplus(self.deltas), 0)])

    def forward(self, s):
        return s.unsqueeze(1) - self.thresholds().unsqueeze(0)


class TinyBackbone(nn.Module):
    """Small CNN with a ViT-like interface (tokens + cls). For the CPU example only."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.embed_dim = dim
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, 5, 4, 2), nn.GELU()),
            nn.Sequential(nn.Conv2d(32, dim, 3, 2, 1), nn.GELU()),
            nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.GELU()),
        ])
        self.norm = nn.LayerNorm(dim)

    def forward_features(self, x):
        for blk in self.blocks:
            x = blk(x)
        tokens = self.norm(x.flatten(2).transpose(1, 2))
        return {"x_norm_patchtokens": tokens, "x_norm_clstoken": tokens.mean(1)}


class ScoreModel(nn.Module):
    def __init__(self, backbone: str = "dinov2_vitb14", objective: str = "pairwise"):
        super().__init__()
        if backbone == "tiny":
            self.backbone = TinyBackbone()
        else:
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone)
        dim = self.backbone.embed_dim
        self.pool = AttentionPool(dim)
        self.trunk = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, HIDDEN), nn.GELU())
        self.score = nn.Linear(HIDDEN, 1)
        self.coral = CoralHead() if objective == "ordinal" else None

    def forward(self, x):
        out = self.backbone.forward_features(x)
        h, attn = self.pool(out["x_norm_patchtokens"], out["x_norm_clstoken"])
        return self.score(self.trunk(h)).squeeze(-1), attn


def set_trainable_blocks(model: ScoreModel, n_blocks: int) -> int:
    """Freeze the backbone, then unfreeze its last `n_blocks` blocks and final norm.

    Phase 1: 0 blocks; Phase 2: last 2; Phase 3: last 6 (of 12). Within an
    unfrozen block all parameters (attention, MLP, LayerNorm, LayerScale) train.
    Returns the number of trainable backbone parameters.
    """
    for p in model.backbone.parameters():
        p.requires_grad = False
    if n_blocks > 0:
        for blk in list(model.backbone.blocks)[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in model.backbone.norm.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)


def bradley_terry_loss(score_a, score_b, a_wins):
    """Negative log-likelihood of the Bradley-Terry model."""
    return F.binary_cross_entropy_with_logits(score_a - score_b, a_wins)


def coral_loss(model: ScoreModel, s, level_idx):
    """Mean binary cross-entropy of the cumulative events BCS > level k."""
    logits = model.coral(s)                                        # [B, 16]
    k = torch.arange(logits.shape[1], device=s.device).unsqueeze(0)
    target = (level_idx.unsqueeze(1) > k).float()
    return F.binary_cross_entropy_with_logits(logits, target)


def coral_predict_level(model: ScoreModel, s):
    """Predicted level index (0..16) from the model's own CORAL thresholds."""
    return (model.coral(s) > 0).sum(dim=1)
