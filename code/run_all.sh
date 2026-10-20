#!/usr/bin/env bash
# Reproduces every task in order (about 3 h on an RTX 4050 laptop GPU). Run from this folder.
set -e
cd "$(dirname "$0")"

# Task 1 (+ Task 2 dynamics, logged with --track)
python nl.py train --tag t1_noisy --labels noisy --method ce --track
python nl.py train --tag t1_clean --labels clean --method ce
# Task 3: 4-fold cross-validated confidence on the noisy labels
python nl.py cv --labels noisy
# Task 4: robust methods, same budget and setup as the noisy baseline
python nl.py train --tag t4_elr --labels noisy --method elr
python nl.py train --tag t4_relabel --labels noisy --method relabel --ce-tag t1_noisy
# Task 5 baseline: CE on synthetic (symmetric) noise
python nl.py train --tag t5_ce --labels synthetic --method ce --track
python nl.py train --tag t4_gce --labels noisy --method gce
# Task 5 best: rerun the Task 4 method with the highest noisy validation accuracy on synthetic labels
BEST=$(python -c "import summarize; print(summarize.best_t4_method())")
echo "best Task 4 method: $BEST"
if [ "$BEST" = "relabel" ]; then
  python nl.py cv --labels synthetic
  python nl.py train --tag t5_best --labels synthetic --method relabel --ce-tag t5_ce
else
  python nl.py train --tag t5_best --labels synthetic --method "$BEST"
fi
python nl.py summarize
