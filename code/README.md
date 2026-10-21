# CIFAR-10N: learning from noisy labels

All code is in `nl.py` (data, GPU augmentation, training loop, losses, Task 3 scores) and
`summarize.py` (writes `predictions/`, `results.json` and `analysis.md`). `model.py` is the provided ResNet-18, unchanged.

## Setup

- Python 3.13, PyTorch 2.13 (CUDA 13.0), NumPy. One NVIDIA GPU (tested on an RTX 4050 Laptop, 6 GB).
- Data: the provided `data/` folder. The code uses `data/` at the submission root if it exists (as in the GitHub
  repo), otherwise `data/` one level above the submission folder; override with `DATA_DIR=/path/to/data`.
- Raw run outputs (per-epoch logits) go to `work/`, located the same way (override with `WORK_DIR`).

## Shared training setup (identical for every run)

ResNet-18 from scratch, SGD (momentum 0.9, Nesterov, weight decay 5e-4), batch 128, 60 epochs,
linear warm-up to lr 0.1 over 1 epoch then per-step cosine decay to 0, bf16 autocast,
random crop (4 px zero padding) + horizontal flip, seed 0. After every epoch the model is evaluated on the
validation and test sets, and the logits are saved, so any epoch can be selected afterwards.

## Reproduce everything

```bash
bash run_all.sh        # about 3 h on an RTX 4050 laptop GPU
```

or step by step:

| Task | Command | Output |
|---|---|---|
| 1 | `python nl.py train --tag t1_noisy --labels noisy --method ce --track` | noisy CE baseline; `--track` also logs training-set logits per epoch (Task 2, Task 3) |
| 1 | `python nl.py train --tag t1_clean --labels clean --method ce` | clean-label reference |
| 2 | (from `t1_noisy`) | `t2_dynamics.csv`; early-stop epoch = argmax of noisy val accuracy |
| 3 | `python nl.py cv --labels noisy` | 4-fold CV (4 runs, 20 epochs each), held-out softmax for all 45k images |
| 4 | `python nl.py train --tag t4_elr --labels noisy --method elr` | Early-Learning Regularization (λ=3, β=0.7) |
| 4 | `python nl.py train --tag t4_relabel --labels noisy --method relabel --ce-tag t1_noisy` | filter/relabel with Task 3 scores, then CE retrain |
| 4 | `python nl.py train --tag t4_gce --labels noisy --method gce` | Generalized CE (q=0.7) |
| 5 | `python nl.py train --tag t5_ce --labels synthetic --method ce --track` | CE baseline on synthetic labels |
| 5 | `python nl.py train --tag t5_best --labels synthetic --method <best>` | best Task 4 method on synthetic labels (for `relabel`, first run `python nl.py cv --labels synthetic` and pass `--ce-tag t5_ce`) |
| all | `python nl.py summarize` | `predictions/*.csv`, `results.json`, `analysis.md` |

## Method details

- **Model selection.** Task 1 predictions come from the final (60th) epoch. Task 2, Task 4 and the Task 5 runs use the epoch
  with the highest validation accuracy, measured with noisy labels for Tasks 2 and 4 and with synthetic labels for Task 5.
- **Task 3 score**, fixed before looking at any clean-label numbers: the mean of the rank-normalised
  (a) `1 - p_cv(noisy label)` from 4-fold cross-validation (mean softmax over the last 5 of 20 epochs) and
  (b) the per-sample CE of the noisy label averaged over all 60 epochs of the Task 1 noisy run.
- **Relabel method.** Flag the top-r images by score, where r is the CV disagreement rate (an estimate of the noise rate
  that uses only noisy labels). A flagged image whose CV prediction has confidence ≥ 0.5 is relabelled to that
  prediction; the other flagged images are dropped. Then train CE on the result with the shared setup.
- **Clean labels** are used only to train `t1_clean` and to *measure* things in `summarize.py`:
  the confusion matrix, the Task 2 curves, Task 3 AUROC / precision@k, and clean validation accuracy reported in
  `analysis.md`. They never select, filter or relabel anything.
