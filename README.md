# Inter-IIT Tech Meet 15.0 CV bootcamp: learning from noisy labels (CIFAR-10N)

Roll number 24B1534. Everything is in this repo: the provided materials, the dataset, the code, every raw
training output and the final submission.

## What to submit

**`24b1534_cv_bootcamp.zip`**: `code/`, `predictions/`, `results.json` and `report.md`, as the problem statement asks.
It passes `python check_format.py 24b1534_cv_bootcamp.zip` ("All good").

## Layout

| Path | Contents |
|---|---|
| `24b1534_cv_bootcamp.zip` | The submission |
| `code/` | `nl.py` (training, losses, noise scores), `summarize.py` (predictions + results), `model.py` (provided ResNet-18), `run_all.sh`, `README.md` (exact commands), `analysis.md` (all numbers), `CV_Bootcamp_Colab.ipynb` |
| `predictions/` | The 10 prediction files for Tasks 1–5 |
| `results.json`, `report.md` | Filled-in results and answers to Q1–Q5 |
| `data/` | The provided dataset (images as `.npy`, labels, splits) |
| `work/` | Raw outputs of every training run: per-epoch val/test (and train) logits, logs, CV held-out probabilities |
| `CV Problem Statement.docx`, `check_format.py`, `results_template.json` | Provided materials |

The `.npy` files are stored with [Git LFS](https://git-lfs.com). Install it before cloning:

```bash
git lfs install
git clone https://github.com/Raghav789-code/24b1534-cv-bootcamp.git
```

## Reproduce

```bash
cd code
bash run_all.sh              # all training runs, about 3 h on an RTX 4050 laptop GPU
python nl.py summarize       # rebuild predictions/ and results.json from work/ (seconds)
```

`run_all.sh` runs every task in order. If you only want to rebuild the outputs, `python nl.py summarize` regenerates
everything from the saved logits in `work/`, with no GPU needed. To run on Google Colab, open
`code/CV_Bootcamp_Colab.ipynb`.

## Results

| | Result |
|---|---|
| Noise rate (training split) | 40.2% |
| T1 val acc, noisy labels (clean-ref / noisy CE, final epoch) | 59.3% / 51.0% |
| T2 memorization start / early stop | epoch 32 / epoch 25 (55.6%) |
| T3 noise-score AUROC / precision@k | 0.956 / 0.856 |
| T4 noisy val acc: ELR (best) / filter+relabel / GCE | 58.3% / 57.4% / 57.1% |
| T5 synthetic val acc: CE / ELR | 50.4% / 54.0% |
