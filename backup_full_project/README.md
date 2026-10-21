# Full project backup (not part of the submission)

Roll number 24B1534. The repo's top level holds **only the submission, in the exact format the problem statement
asks for**: `code/`, `predictions/`, `results.json` and `report.md`. This folder holds everything else, so nothing
is lost: the dataset, every raw training output, the provided materials and the submission zip.

## Contents

| Path | Contents |
|---|---|
| `24b1534_cv_bootcamp.zip` | The submission zip (the repo's top-level `code/`, `predictions/`, `results.json`, `report.md`) |
| `data/` | The provided dataset (images as `.npy`, labels, splits) |
| `work/` | Raw outputs of every training run: per-epoch val/test (and train) logits, logs, CV held-out probabilities, `stats.json` |
| `CV Problem Statement.docx`, `check_format.py`, `results_template.json` | Provided materials |

The `.npy` files are stored with [Git LFS](https://git-lfs.com). Install it before cloning:

```bash
git lfs install
git clone https://github.com/Raghav789-code/24b1534-cv-bootcamp.git
```

## Check the submission

```bash
cd backup_full_project
python check_format.py 24b1534_cv_bootcamp.zip      # prints "All good."
```

## Reproduce

The code in the top-level `code/` folder finds `backup_full_project/data` and `backup_full_project/work` on its own.

```bash
cd code
python nl.py summarize       # rebuild predictions/ and results.json from the saved logits (seconds, no GPU)
bash run_all.sh              # or rerun all training, about 3 h on an RTX 4050 laptop GPU
```

To run on Google Colab, open `code/CV_Bootcamp_Colab.ipynb`.

## Results

| | Result |
|---|---|
| Noise rate (training split) | 40.2% |
| T1 val acc, noisy labels (clean-ref / noisy CE, final epoch) | 59.3% / 51.0% |
| T2 memorization start / early stop | epoch 32 / epoch 25 (55.6%) |
| T3 noise-score AUROC / precision@k | 0.956 / 0.856 |
| T4 noisy val acc: ELR (best) / filter+relabel / GCE | 58.3% / 57.4% / 57.1% |
| T5 synthetic val acc: CE / ELR | 50.4% / 54.0% |
