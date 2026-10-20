# CV Bootcamp report: learning from noisy labels (CIFAR-10N "worst")

Supporting tables (confusion matrix, per-epoch dynamics, all metrics) are in `code/analysis.md`.

Q1. Which class pairs account for most of the noise?

The noise rate is 40.2% on the 45k training split. The noise is concentrated in visually similar pairs: automobile↔truck accounts for 13.1% of all wrong labels (truck→automobile 1,247, automobile→truck 1,126), cat↔dog for 9.7%, deer↔horse for 5.1% and airplane↔ship for 4.1%. Deer→dog (491) and bird→frog/deer add more. These pairs are strongly asymmetric across classes: cat and deer images are mislabeled 51% of the time, horse only 28%. So the noise is class-dependent and systematic, not uniform.

Q2. At which epoch does memorization of wrong labels start, and what is the evidence?

Epoch 32. From epoch 4 to 30, accuracy on the mislabeled subset measured against the wrong label stays flat at 12–15%, and 60–66% of those images are predicted as their true class. From epoch 32 the wrong-label fit rises steadily to 69% at epoch 60, while true-class predictions fall to 21%. Clean validation accuracy peaks at 79.4% (epoch 41) and drops to 69.3%. Noisy validation accuracy peaks at epoch 25 (55.6%), which we use for early stopping (clean val 78.7%), and decays to 51.0%.

Q3. Which noise-score signal worked best? Describe one failure mode.

Per-sample loss averaged over all 60 epochs of the noisy CE run was best (AUROC 0.958, precision@k 0.862), slightly ahead of 4-fold cross-validated confidence (0.942 / 0.837). The submitted score averages both ranks (0.956 / 0.856). Final-epoch loss alone is poor (0.737) because wrong labels are memorized by then. Failure mode: systematic confusions are missed. Only 64% of mislabeled automobile↔truck and 60% of cat↔dog images are flagged, against about 90% for deer↔dog or bird↔frog. Annotators mislabel ambiguous images consistently, so the model learns to agree with them.

Q4. Why did your best Task 4 method help, or not help?

ELR was best: 58.3% noisy validation accuracy, against 55.6% for early-stopped CE (clean validation 86.7% vs 78.7%, measured only). Early in training the network fits mostly correct labels. ELR keeps a running average of each image's predictions and penalises moving away from it, cancelling a wrong label's gradient once the model is confident in the true class. Memorization never sets in: its best epoch is the last. Filter+relabel also helped (57.4%), but it dropped 9.4k images and inherits Task 3's errors on automobile/truck and cat/dog.

Q5. Did the ranking of methods change between real and synthetic noise? Why?

No. ELR beats CE on both: 54.0% vs 50.4% synthetic validation accuracy, against 58.3% vs 55.6% on real noise. Clean validation accuracy (measured only) is 90.6% vs 83.7% on synthetic and 86.7% vs 78.7% on real. At the same 40% rate, synthetic noise is easier: CE starts memorizing later (epoch 42 vs 32) and fits fewer wrong labels (48% vs 69%). Random flips are inconsistent across similar images, so their gradients cancel. Human errors repeat on the same ambiguous images (automobile/truck, cat/dog), so they look like learnable signal.
