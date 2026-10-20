"""Learning from noisy labels on CIFAR-10N (Inter-IIT CV bootcamp). Every task uses this file.

    python nl.py train --tag t1_noisy --labels noisy --method ce --track
    python nl.py cv --labels noisy
    python nl.py summarize

See README.md for the full sequence. Raw outputs (per-epoch logits, histories) go to WORK_DIR;
`summarize` turns them into predictions/, results.json and stats used in report.md.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import ResNet18  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", HERE.parents[1] / "data"))
WORK = Path(os.environ.get("WORK_DIR", HERE.parents[1] / "work"))
SUB = HERE.parent
PRED = SUB / "predictions"

# Shared training setup: identical for the baseline and every method.
SEED = 0
EPOCHS = 60
BATCH = 128
LR = 0.1
WD = 5e-4
MOMENTUM = 0.9
WARMUP_EPOCHS = 1
CV_FOLDS = 4
CV_EPOCHS = 20
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
DEV = "cuda"
# bf16 on Ampere and newer; fp16 with loss scaling on older GPUs (e.g. Colab's T4)
AMP = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
COLS = {"noisy": 1, "synthetic": 2, "clean": 3}


# ---------------------------------------------------------------- data
def load_data():
    lab = np.loadtxt(DATA / "labels.csv", delimiter=",", skiprows=1, dtype=np.int64)
    assert (lab[:, 0] == np.arange(len(lab))).all()
    d = {k: lab[:, c] for k, c in COLS.items()}
    d["train_idx"] = np.loadtxt(DATA / "train_idx.txt", dtype=np.int64)
    d["val_idx"] = np.loadtxt(DATA / "val_idx.txt", dtype=np.int64)
    return d


def to_gpu(images):
    x = torch.from_numpy(images).to(DEV).permute(0, 3, 1, 2).float().div_(255)
    return ((x - MEAN.to(DEV)) / STD.to(DEV)).half().contiguous()


def augment(x):
    """Random 32x32 crop from the zero-padded (4px) image plus horizontal flip, done on the GPU."""
    b = x.shape[0]
    x = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (b,), device=DEV)
    j = torch.randint(0, 9, (b,), device=DEV)
    ar = torch.arange(32, device=DEV)
    rows = (i[:, None] + ar)[:, None, :, None]
    cols = (j[:, None] + ar)[:, None, None, :]
    x = x[torch.arange(b, device=DEV)[:, None, None, None], torch.arange(3, device=DEV)[None, :, None, None],
          rows, cols]
    flip = torch.rand(b, device=DEV) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


# ---------------------------------------------------------------- losses
class CE:
    def __call__(self, out, y, idx):
        return F.cross_entropy(out, y)


class GCE:
    """Generalized cross-entropy (Zhang & Sabuncu 2018): (1 - p_y^q) / q."""

    def __init__(self, q=0.7):
        self.q = q

    def __call__(self, out, y, idx):
        p = F.softmax(out.float(), 1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
        return ((1 - p ** self.q) / self.q).mean()


class ELR:
    """Early-learning regularization (Liu et al. 2020): CE + lam * log(1 - <t_i, p_i>),
    where t_i is a running average of the model's own predictions for sample i."""

    def __init__(self, n, lam=3.0, beta=0.7):
        self.t = torch.zeros(n, 10, device=DEV)
        self.lam, self.beta = lam, beta

    def __call__(self, out, y, idx):
        p = F.softmax(out.float(), 1).clamp(1e-4, 1 - 1e-4)
        self.t[idx] = self.beta * self.t[idx] + (1 - self.beta) * (p / p.sum(1, keepdim=True)).detach()
        reg = torch.log(1 - (self.t[idx] * p).sum(1)).mean()
        return F.cross_entropy(out, y) + self.lam * reg


# ---------------------------------------------------------------- training
@torch.no_grad()
def predict(model, x, bs=1000):
    model.eval()
    outs = []
    for k in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=AMP):
            outs.append(model(x[k:k + bs].contiguous(memory_format=torch.channels_last)).float())
    return torch.cat(outs)


def fit(x_train, y_train, loss_fn, eval_sets, epochs=EPOCHS, seed=SEED, log=print):
    """Train a fresh ResNet-18. Returns {name: (epochs, n, 10) float16 logits} for every eval set,
    evaluated after each epoch, plus the per-epoch mean training loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(x_train)
    steps_per_epoch = n // BATCH
    total = epochs * steps_per_epoch
    warm = WARMUP_EPOCHS * steps_per_epoch
    model = ResNet18(10).to(DEV, memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, nesterov=True, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP == torch.float16)
    logits = {k: [] for k in eval_sets}
    losses = []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=DEV)
        run = torch.zeros((), device=DEV)
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            xb = augment(x_train[idx]).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=AMP):
                out = model(xb)
            loss = loss_fn(out.float(), y_train[idx], idx)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.detach()
        losses.append(run.item() / steps_per_epoch)
        msg = []
        for k, (xe, ye) in eval_sets.items():
            out = predict(model, xe)
            logits[k].append(out.half().cpu().numpy())
            if ye is not None:
                msg.append(f"{k} {(out.argmax(1) == ye).float().mean().item():.4f}")
        log(f"ep {ep + 1:2d}/{epochs} loss {losses[-1]:.4f} {' '.join(msg)} ({time.time() - t0:.1f}s)")
    return {k: np.stack(v) for k, v in logits.items()}, losses


def make_logger(path):
    f = open(path, "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        f.write(m + "\n")
        f.flush()
    return log


# ---------------------------------------------------------------- noise scores (Task 3)
def cv_probs(labels):
    """Held-out softmax for every training image from a CV_FOLDS-fold split (mean of the last 5 epochs)."""
    p = np.load(WORK / f"cv_{labels}" / "heldout_probs.npy")
    return p


def noise_scores(d, labels, ce_tag):
    """Final Task 3 score, fixed in advance: mean of the rank-normalised
    (a) cross-validated probability that the given label is wrong, 1 - p_cv[y], and
    (b) per-sample cross-entropy of the given label averaged over all epochs of the CE run on the same labels."""
    tr = d["train_idx"]
    y = d[labels][tr]
    p = cv_probs(labels)
    s_cv = 1 - p[np.arange(len(tr)), y]
    s_loss = mean_train_loss(ce_tag, y)
    return 0.5 * (rank01(s_cv) + rank01(s_loss)), s_cv, s_loss


def mean_train_loss(tag, y):
    lg = np.load(WORK / tag / "train_logits.npy").astype(np.float32)  # (E, N, 10)
    lsm = lg - np.log(np.exp(lg - lg.max(2, keepdims=True)).sum(2, keepdims=True)) - lg.max(2, keepdims=True)
    return -lsm[:, np.arange(len(y)), y].mean(0)


def rank01(s):
    r = np.empty(len(s))
    r[np.argsort(s, kind="stable")] = np.arange(len(s))
    return r / (len(s) - 1)


def relabeled_set(d, labels, ce_tag, conf_thr=0.5):
    """Flag the top-r scoring images, where r is the CV disagreement rate (an estimate of the noise rate
    from noisy labels only). Flagged images whose CV prediction is confident are relabelled to it; the rest
    are dropped. Returns (positions into train_idx to keep, their new labels, info)."""
    tr = d["train_idx"]
    y = d[labels][tr].copy()
    score, _, _ = noise_scores(d, labels, ce_tag)
    p = cv_probs(labels)
    pred, conf = p.argmax(1), p.max(1)
    rate = float((pred != y).mean())
    flagged = np.argsort(-score)[:int(round(rate * len(tr)))]
    relabel = flagged[(conf[flagged] >= conf_thr) & (pred[flagged] != y[flagged])]
    drop = np.setdiff1d(flagged, relabel)
    y[relabel] = pred[relabel]
    keep = np.setdiff1d(np.arange(len(tr)), drop)
    return keep, y[keep], {"est_noise_rate": rate, "flagged": len(flagged), "relabelled": len(relabel),
                           "dropped": len(drop)}


# ---------------------------------------------------------------- commands
def cmd_train(a):
    d = load_data()
    out = WORK / a.tag
    out.mkdir(parents=True, exist_ok=True)
    log = make_logger(out / "log.txt")
    imgs = to_gpu(np.load(DATA / "train_images.npy"))
    test = to_gpu(np.load(DATA / "test_images.npy"))
    tr, va = d["train_idx"], d["val_idx"]
    pos = np.arange(len(tr))
    y_np = d[a.labels][tr]
    info = {}
    if a.method == "relabel":
        pos, y_np, info = relabeled_set(d, a.labels, a.ce_tag)
        log(f"relabel: {info}")
    x_train = imgs[torch.from_numpy(tr[pos]).to(DEV)]
    y_train = torch.from_numpy(y_np).to(DEV)
    loss_fn = {"ce": lambda: CE(), "relabel": lambda: CE(), "gce": lambda: GCE(),
               "elr": lambda: ELR(len(pos))}[a.method]()
    sel = "synthetic" if a.labels == "synthetic" else "noisy"
    eval_sets = {f"val_{sel}": (imgs[torch.from_numpy(va).to(DEV)], torch.from_numpy(d[sel][va]).to(DEV)),
                 "test": (test, None)}
    if a.track:
        eval_sets["train"] = (imgs[torch.from_numpy(tr).to(DEV)], None)
    log(f"== {a.tag}: method={a.method} labels={a.labels} n_train={len(pos)} epochs={EPOCHS}")
    logits, losses = fit(x_train, y_train, loss_fn, eval_sets, log=log)
    np.save(out / "val_logits.npy", logits[f"val_{sel}"])
    np.save(out / "test_logits.npy", logits["test"])
    if a.track:
        np.save(out / "train_logits.npy", logits["train"])
    json.dump({"method": a.method, "labels": a.labels, "train_loss": losses, "info": info},
              open(out / "meta.json", "w"), indent=1)


def cmd_cv(a):
    d = load_data()
    out = WORK / f"cv_{a.labels}"
    out.mkdir(parents=True, exist_ok=True)
    log = make_logger(out / "log.txt")
    imgs = to_gpu(np.load(DATA / "train_images.npy"))
    tr = d["train_idx"]
    y = d[a.labels][tr]
    rng = np.random.RandomState(SEED)
    fold = rng.permutation(len(tr)) % CV_FOLDS
    probs = np.zeros((len(tr), 10), np.float32)
    for k in range(CV_FOLDS):
        fit_pos, held = np.where(fold != k)[0], np.where(fold == k)[0]
        log(f"== cv_{a.labels} fold {k + 1}/{CV_FOLDS}: train {len(fit_pos)} held-out {len(held)}")
        x_fit = imgs[torch.from_numpy(tr[fit_pos]).to(DEV)]
        x_held = imgs[torch.from_numpy(tr[held]).to(DEV)]
        logits, _ = fit(x_fit, torch.from_numpy(y[fit_pos]).to(DEV), CE(),
                        {"heldout": (x_held, torch.from_numpy(y[held]).to(DEV))}, epochs=CV_EPOCHS,
                        seed=SEED + k, log=log)
        lg = torch.from_numpy(logits["heldout"][-5:].astype(np.float32))
        probs[held] = F.softmax(lg, 2).mean(0).numpy()
    np.save(out / "heldout_probs.npy", probs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--tag", required=True)
    t.add_argument("--labels", choices=["noisy", "synthetic", "clean"], required=True)
    t.add_argument("--method", choices=["ce", "gce", "elr", "relabel"], default="ce")
    t.add_argument("--track", action="store_true", help="also log logits on the whole training set every epoch")
    t.add_argument("--ce-tag", help="CE run (with --track) on the same labels; used by --method relabel")
    c = sp.add_parser("cv")
    c.add_argument("--labels", choices=["noisy", "synthetic"], required=True)
    sp.add_parser("summarize")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    if args.cmd == "summarize":
        from summarize import main as summarize_main
        summarize_main()
    else:
        {"train": cmd_train, "cv": cmd_cv}[args.cmd](args)
