"""Three-phase fine-tuning (Section 2.6 and Table S1 of the article).

Phase 1 (epochs 1-5):   backbone frozen; head lr 3e-4
Phase 2 (epochs 6-10):  last 2 transformer blocks unfrozen; head 2e-4, backbone 2e-5
Phase 3 (epochs 11-30): last 6 blocks + final LayerNorm unfrozen; head 1e-4, backbone 1e-5
AdamW (weight decay 0.05), linear warm-up over the first 5% of steps, cosine decay,
batch size 64, mixed precision on GPU. Validation every 2 epochs; the checkpoint
with the best validation metric is kept (pairwise: share of validation pairs
ordered correctly; ordinal: Spearman correlation with validation BCS).

Command line (image-disjoint protocol, one seed):
    python -m pairwise_bcs.train --images images/ --comparisons comparisons.csv \
        --ordinal ordinal.csv --objective pairwise --seed 42 --out runs/pairwise_s42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .data import ImageStore, level_index, load_comparisons, load_ordinal
from .model import ScoreModel, bradley_terry_loss, coral_loss, set_trainable_blocks
from .splits import image_disjoint_split


@dataclass
class Config:
    backbone: str = "dinov2_vitb14"
    phase_epochs: tuple = (5, 5, 20)
    unfreeze_blocks: tuple = (0, 2, 6)
    lr_head: tuple = (3e-4, 2e-4, 1e-4)
    lr_backbone: tuple = (0.0, 2e-5, 1e-5)
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    batch_size: int = 64
    eval_every: int = 2
    seed: int = 42


def score_images(model, store, ids, device, batch_size=128):
    model.eval()
    out = {}
    with torch.no_grad():
        for k in range(0, len(ids), batch_size):
            chunk = ids[k:k + batch_size]
            s, _ = model(store.batch(chunk, device))
            out.update(zip(chunk, s.float().cpu().numpy().tolist()))
    return out


def pair_accuracy(scores, pairs):
    return float(np.mean([scores[w] > scores[l] for w, l in pairs]))


def train(store, objective, train_pairs, val_pairs, ordinal=None, train_images=None, cfg=Config(), device=None):
    """Train one model. Returns (model, history)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = ScoreModel(cfg.backbone, objective).to(device)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": head, "lr": cfg.lr_head[0], "weight_decay": cfg.weight_decay},
                             {"params": list(model.backbone.parameters()), "lr": 0.0,
                              "weight_decay": cfg.weight_decay}])
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if objective == "pairwise":
        units = list(train_pairs)
    else:
        units = [(i, level_index(ordinal[i])) for i in train_images if i in ordinal]
    steps_per_epoch = max(1, len(units) // cfg.batch_size)
    total_steps = steps_per_epoch * sum(cfg.phase_epochs)
    warmup = int(total_steps * cfg.warmup_frac)
    val_images = sorted({x for p in val_pairs for x in p}, key=str)

    best, history, step, epoch = (-math.inf, None), [], 0, 0
    for phase, n_epochs in enumerate(cfg.phase_epochs):
        set_trainable_blocks(model, cfg.unfreeze_blocks[phase])
        for _ in range(n_epochs):
            epoch += 1
            model.train()
            order = rng.permutation(len(units))
            losses = []
            for bi in range(steps_per_epoch):
                batch = [units[k] for k in order[bi * cfg.batch_size:(bi + 1) * cfg.batch_size]]
                if len(batch) < 2:
                    continue
                # learning rate: phase base rate x (warm-up, then cosine) over all steps
                scale = step / max(warmup, 1) if step < warmup else \
                    0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))
                opt.param_groups[0]["lr"] = cfg.lr_head[phase] * max(scale, 1e-3)
                opt.param_groups[1]["lr"] = cfg.lr_backbone[phase] * max(scale, 1e-3)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    if objective == "pairwise":
                        # random order within each pair, so the model cannot use position
                        flip = rng.random(len(batch)) < 0.5
                        a = [l if f else w for (w, l), f in zip(batch, flip)]
                        b = [w if f else l for (w, l), f in zip(batch, flip)]
                        y = torch.tensor(~flip, dtype=torch.float32, device=device)
                        sa, _ = model(store.batch(a, device, augment=True, rng=rng))
                        sb, _ = model(store.batch(b, device, augment=True, rng=rng))
                        loss = bradley_terry_loss(sa.float(), sb.float(), y)
                    else:
                        x = store.batch([i for i, _ in batch], device, augment=True, rng=rng)
                        lv = torch.tensor([lvl for _, lvl in batch], device=device)
                        s, _ = model(x)
                        loss = coral_loss(model, s.float(), lv)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                losses.append(float(loss)); step += 1

            if epoch % cfg.eval_every == 0 or epoch == sum(cfg.phase_epochs):
                vs = score_images(model, store, val_images, device)
                if objective == "pairwise":
                    metric = pair_accuracy(vs, val_pairs)
                else:
                    ids = [i for i in val_images if i in ordinal]
                    metric = float(spearmanr([vs[i] for i in ids], [ordinal[i] for i in ids])[0])
                history.append({"epoch": epoch, "phase": phase + 1, "loss": float(np.mean(losses)), "val_metric": metric})
                print(f"epoch {epoch:2d} phase {phase + 1} loss {np.mean(losses):.4f} val {metric:.4f}", flush=True)
                if metric > best[0]:
                    best = (metric, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model, history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--comparisons", required=True)
    ap.add_argument("--ordinal", required=True)
    ap.add_argument("--objective", choices=["pairwise", "ordinal"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--epochs", type=int, nargs=3, default=(5, 5, 20), metavar=("P1", "P2", "P3"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--budget-matched", action="store_true",
                    help="pairwise only: train on as many comparisons as there are ordinal training scores")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    comps = load_comparisons(args.comparisons)
    ordinal = load_ordinal(args.ordinal)
    split = image_disjoint_split(comps, ordinal, seed=args.seed)
    train_pairs = split["train_pairs"]
    if args.budget_matched:
        rnd = random.Random(args.seed * 7919)
        train_pairs = rnd.sample(train_pairs, len(split["train_images"]))
    ids = sorted({x for c in comps for x in c} & set(ordinal), key=str)
    store = ImageStore(args.images, ids)
    cfg = Config(backbone=args.backbone, phase_epochs=tuple(args.epochs), batch_size=args.batch_size, seed=args.seed)
    model, history = train(store, args.objective, train_pairs, split["val_pairs"], ordinal,
                           split["train_images"], cfg)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    scores = score_images(model, store, split["train_images"] + split["test_images"], str(device))
    torch.save(model.state_dict(), out / "model.pt")
    json.dump({"config": asdict(cfg), "history": history, "split": split,
               "scores": scores}, open(out / "run.json", "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
