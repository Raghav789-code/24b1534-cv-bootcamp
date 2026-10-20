"""Turn the raw run outputs in WORK_DIR into predictions/, results.json and code/analysis.md.

Clean labels are used here only to *measure* (confusion matrix, Task 2 curves, Task 3 AUROC, and
clean-label validation accuracy reported for analysis). Every selection uses noisy (Tasks 1-4) or
synthetic (Task 5) validation labels.
"""
import csv
import json

import numpy as np

from nl import BATCH, EPOCHS, PRED, SUB, WORK, load_data, noise_scores

T4_METHODS = ["elr", "relabel", "gce"]
T4_NAMES = {"elr": "ELR (early-learning regularization)",
            "relabel": "CV-score filter + relabel, then CE retrain",
            "gce": "GCE (generalized cross-entropy, q=0.7)"}
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
_D = None


def data():
    global _D
    if _D is None:
        _D = load_data()
    return _D


def has(tag):
    return (WORK / tag / "val_logits.npy").exists()


def val_curve(tag, labels):
    """Per-epoch validation accuracy of run `tag` against the given label column."""
    d = data()
    lg = np.load(WORK / tag / "val_logits.npy")
    return (lg.argmax(2) == d[labels][d["val_idx"]]).mean(1)


def best_t4_method():
    done = [m for m in T4_METHODS if has(f"t4_{m}")]
    return max(done, key=lambda m: val_curve(f"t4_{m}", "noisy").max())


def test_preds(tag, epoch):
    """Test predictions after `epoch` (1-based)."""
    return np.load(WORK / tag / "test_logits.npy")[epoch - 1].astype(np.float32).argmax(1)


def write_preds(name, pred):
    with open(PRED / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pred"])
        w.writerows(zip(range(len(pred)), pred.tolist()))


def auroc(score, pos):
    r = np.empty(len(score))
    order = np.argsort(score, kind="stable")
    r[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    s_sorted = score[order]
    _, start, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    for st, c in zip(start, counts):
        if c > 1:
            r[order[st:st + c]] = st + (c + 1) / 2
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def prec_at_k(score, pos):
    k = int(pos.sum())
    return float(pos[np.argsort(-score)[:k]].mean())


def memorization_start(acc_m, margin=0.03):
    """First epoch (1-based) from which the 3-epoch moving average of the accuracy on the *wrong* labels of
    the mislabeled subset stays more than `margin` above its early plateau (median of epochs 5-20)."""
    s = np.convolve(acc_m, np.ones(3) / 3, "same")
    s[0], s[-1] = acc_m[:2].mean(), acc_m[-2:].mean()
    above = s > np.median(acc_m[4:20]) + margin
    return next(i for i in range(len(s)) if above[i:].all()) + 1


def r4(x):
    return None if x is None else round(float(x), 4)


def main():
    d = data()
    tr, va = d["train_idx"], d["val_idx"]
    PRED.mkdir(exist_ok=True)
    res = json.load(open(SUB / "results.json")) if (SUB / "results.json").exists() else {}
    res.setdefault("name", "YOUR NAME")
    res.setdefault("roll_no", "YOUR_ROLL_NO")
    res["seed"] = 0
    res["training_setup"] = {
        "optimizer": "SGD, momentum 0.9 (Nesterov), weight decay 5e-4, bf16 autocast",
        "learning_rate_schedule": "linear warmup to 0.1 over 1 epoch, then per-step cosine decay to 0",
        "batch_size": BATCH,
        "epochs": EPOCHS,
        "augmentation": "random 32x32 crop from 4px zero-padded image + random horizontal flip",
        "gpu_used": "NVIDIA GeForce RTX 4050 Laptop GPU (6 GB)",
    }
    A = []  # analysis.md lines
    stats = {}

    # ---- Task 1
    noisy, clean = d["noisy"][tr], d["clean"][tr]
    wrong = noisy != clean
    rate = float(wrong.mean())
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (clean, noisy), 1)
    off = [(cm[i, j], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10) if i != j]
    off.sort(reverse=True)
    pair = {}
    for c, a, b in off:
        pair[tuple(sorted((a, b)))] = pair.get(tuple(sorted((a, b))), 0) + c
    pairs = sorted(pair.items(), key=lambda kv: -kv[1])
    stats["noise_rate_train"] = rate
    stats["noise_rate_all50k"] = float((d["noisy"] != d["clean"]).mean())
    stats["top_directed"] = [(int(c), a, b, round(c / wrong.sum(), 4)) for c, a, b in off[:10]]
    stats["top_pairs"] = [(a, b, int(c), round(c / wrong.sum(), 4)) for (a, b), c in pairs[:8]]
    A += ["# Analysis (supporting numbers for report.md)", "",
          f"## Task 1: noise rate {rate:.4f} on the 45k training split "
          f"({stats['noise_rate_all50k']:.4f} on all 50k)", "",
          "Confusion matrix, rows = clean label, columns = noisy label (training split):", "",
          "| clean \\ noisy | " + " | ".join(CLASSES) + " |", "|---" * 11 + "|"]
    for i in range(10):
        A.append(f"| {CLASSES[i]} | " + " | ".join(str(v) for v in cm[i]) + " |")
    A += ["", "Per-class noise rate (fraction of each clean class that got a wrong label): " +
          ", ".join(f"{CLASSES[i]} {1 - cm[i, i] / cm[i].sum():.3f}" for i in range(10)), "",
          "Largest undirected confusion pairs (share of all wrong labels): " +
          "; ".join(f"{a}-{b} {c} ({s:.1%})" for a, b, c, s in stats["top_pairs"]), ""]
    t1 = {"noise_rate": r4(rate)}
    for tag, key, fname in [("t1_clean", "val_acc_noisy_labels__clean_ref_model", "t1_clean_ref.csv"),
                            ("t1_noisy", "val_acc_noisy_labels__noisy_ce_model", "t1_noisy_ce.csv")]:
        if has(tag):
            write_preds(fname, test_preds(tag, EPOCHS))
            t1[key] = r4(val_curve(tag, "noisy")[-1])
            stats[tag] = {"val_noisy_final": t1[key], "val_clean_final": r4(val_curve(tag, "clean")[-1])}
    res["task1"] = {**res.get("task1", {}), **t1}
    A += ["Final-epoch validation accuracy: " + json.dumps({k: v for k, v in stats.items() if k.startswith("t1_")}),
          ""]

    # ---- Task 2
    if (WORK / "t1_noisy" / "train_logits.npy").exists():
        tp = np.load(WORK / "t1_noisy" / "train_logits.npy").argmax(2)  # (E, N)
        acc_c = (tp[:, ~wrong] == noisy[~wrong]).mean(1)
        acc_m = (tp[:, wrong] == noisy[wrong]).mean(1)
        acc_m_true = (tp[:, wrong] == clean[wrong]).mean(1)
        vn, vc = val_curve("t1_noisy", "noisy"), val_curve("t1_noisy", "clean")
        with open(PRED / "t2_dynamics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"])
            for e in range(len(vn)):
                w.writerow([e + 1, f"{acc_c[e]:.4f}", f"{acc_m[e]:.4f}", f"{vn[e]:.4f}", f"{vc[e]:.4f}"])
        mem = memorization_start(acc_m)
        es = int(np.argmax(vn)) + 1
        write_preds("t2_early_stop.csv", test_preds("t1_noisy", es))
        res["task2"] = {"memorization_start_epoch": mem, "early_stop_epoch": es,
                        "val_acc_noisy_labels_at_early_stop": r4(vn[es - 1])}
        stats["t2"] = {"mem_epoch": mem, "acc_mis_at_mem": r4(acc_m[mem - 1]), "acc_mis_final": r4(acc_m[-1]),
                       "acc_mis_true_at_mem": r4(acc_m_true[mem - 1]), "acc_mis_true_final": r4(acc_m_true[-1]),
                       "acc_correct_at_mem": r4(acc_c[mem - 1]), "acc_correct_final": r4(acc_c[-1]),
                       "val_clean_peak_epoch": int(np.argmax(vc)) + 1, "val_clean_peak": r4(vc.max()),
                       "val_clean_final": r4(vc[-1]), "early_stop": es, "val_noisy_es": r4(vn[es - 1]),
                       "val_clean_es": r4(vc[es - 1]), "val_noisy_final": r4(vn[-1])}
        A += ["## Task 2", "", "| epoch | train_acc_correct | train_acc_mislabeled (fits wrong label) | "
              "mislabeled predicted as clean label | val_noisy | val_clean |", "|---" * 6 + "|"]
        A += [f"| {e + 1} | {acc_c[e]:.4f} | {acc_m[e]:.4f} | {acc_m_true[e]:.4f} | {vn[e]:.4f} | {vc[e]:.4f} |"
              for e in range(len(vn))]
        A += ["", json.dumps(stats["t2"]), ""]

    # ---- Task 3
    if (WORK / "cv_noisy" / "heldout_probs.npy").exists() and (WORK / "t1_noisy" / "train_logits.npy").exists():
        score, s_cv, s_loss = noise_scores(d, "noisy", "t1_noisy")
        with open(PRED / "t3_noise_scores.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "score"])
            w.writerows((int(i), f"{s:.6f}") for i, s in zip(tr, score))
        res["task3"] = {"method": "mean of rank-normalised (a) 1 - cross-validated softmax prob of the noisy label "
                                  "(4-fold CV, 20 epochs per fold, mean of last 5 epochs) and (b) per-sample CE loss "
                                  "averaged over all 60 epochs of the Task 1 noisy CE run"}
        stats["t3"] = {}
        for name, s in [("combined (submitted)", score), ("cv_confidence", s_cv), ("mean_train_loss", s_loss)]:
            stats["t3"][name] = {"auroc": r4(auroc(s, wrong)), "prec_at_k": r4(prec_at_k(s, wrong))}
        lg = np.load(WORK / "t1_noisy" / "train_logits.npy")
        for e in [stats.get("t2", {}).get("early_stop"), EPOCHS]:
            if e:
                l = lg[e - 1].astype(np.float32)
                l = l - l.max(1, keepdims=True)
                loss_e = -(l[np.arange(len(tr)), noisy] - np.log(np.exp(l).sum(1)))
                stats["t3"][f"loss_at_epoch_{e}"] = {"auroc": r4(auroc(loss_e, wrong)),
                                                     "prec_at_k": r4(prec_at_k(loss_e, wrong))}
        # failure mode: where the score misses (mislabeled images ranked low) by class pair
        k = int(wrong.sum())
        flagged = np.zeros(len(tr), bool)
        flagged[np.argsort(-score)[:k]] = True
        miss = wrong & ~flagged
        recall_pair = {}
        for (a, b), _ in pairs[:6]:
            ia, ib = CLASSES.index(a), CLASSES.index(b)
            m = wrong & (((clean == ia) & (noisy == ib)) | ((clean == ib) & (noisy == ia)))
            recall_pair[f"{a}-{b}"] = r4(flagged[m].mean())
        stats["t3"]["recall_at_k_overall"] = r4(flagged[wrong].mean())
        stats["t3"]["recall_at_k_by_pair"] = recall_pair
        stats["t3"]["missed_mislabeled"] = int(miss.sum())
        A += ["## Task 3 (clean labels used only to measure)", "", json.dumps(stats["t3"], indent=1), ""]

    # ---- Task 4
    methods = []
    rows = []
    if "t2" in stats:
        rows.append(("CE baseline (early-stopped, Task 2)", stats["t2"]["val_noisy_es"], stats["t2"]["val_clean_es"],
                     stats["t1_noisy"]["val_noisy_final"], stats["t1_noisy"]["val_clean_final"]))
    for m in T4_METHODS:
        tag = f"t4_{m}"
        if not has(tag):
            continue
        vn, vc = val_curve(tag, "noisy"), val_curve(tag, "clean")
        e = int(np.argmax(vn)) + 1
        write_preds(f"t4_{m}.csv", test_preds(tag, e))
        methods.append({"name": m, "file": f"t4_{m}.csv", "val_acc_noisy_labels": r4(vn[e - 1])})
        rows.append((T4_NAMES[m] + f" [best epoch {e}]", r4(vn[e - 1]), r4(vc[e - 1]), r4(vn[-1]), r4(vc[-1])))
        meta = json.load(open(WORK / tag / "meta.json"))
        if meta.get("info"):
            stats["relabel_info"] = meta["info"]
    if methods:
        best = max(methods, key=lambda x: x["val_acc_noisy_labels"])["name"]
        res["task4"] = {"methods": methods, "best_method": best}
        stats["t4_best"] = best
    stats["t4_rows"] = rows
    A += ["## Task 4 (noisy labels; epoch picked by noisy val acc)", "",
          "| method | val noisy @sel | val clean @sel | val noisy final | val clean final |", "|---" * 5 + "|"]
    A += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows]
    if "relabel_info" in stats:
        A += ["", "Relabel step: " + json.dumps(stats["relabel_info"])]
    A.append("")

    # ---- Task 5
    t5 = {}
    rows5 = []
    for tag, key, fname in [("t5_ce", "val_acc_synthetic_labels__baseline", "t5_baseline_synth.csv"),
                            ("t5_best", "val_acc_synthetic_labels__best_method", "t5_best_synth.csv")]:
        if has(tag):
            vs, vc = val_curve(tag, "synthetic"), val_curve(tag, "clean")
            e = int(np.argmax(vs)) + 1
            write_preds(fname, test_preds(tag, e))
            t5[key] = r4(vs[e - 1])
            rows5.append((tag + f" [best epoch {e}]", r4(vs[e - 1]), r4(vc[e - 1]), r4(vs[-1]), r4(vc[-1])))
    if (WORK / "t5_ce" / "train_logits.npy").exists():
        syn_wrong = d["synthetic"][tr] != clean
        tp = np.load(WORK / "t5_ce" / "train_logits.npy").argmax(2)
        acc_m5 = (tp[:, syn_wrong] == d["synthetic"][tr][syn_wrong]).mean(1)
        stats["t5_synth_noise_rate"] = r4(syn_wrong.mean())
        stats["t5_mem_epoch"] = memorization_start(acc_m5)
        stats["t5_acc_mis_final"] = r4(acc_m5[-1])
        if (WORK / "cv_synthetic" / "heldout_probs.npy").exists():
            sc, scv, sl = noise_scores(d, "synthetic", "t5_ce")
            stats["t5_noise_score_auroc"] = r4(auroc(sc, syn_wrong))
    res["task5"] = {**res.get("task5", {}), **t5}
    stats["t5_rows"] = rows5
    A += ["## Task 5 (synthetic labels; epoch picked by synthetic val acc)", "",
          "| run | val synth @sel | val clean @sel | val synth final | val clean final |", "|---" * 5 + "|"]
    A += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows5]
    A += ["", json.dumps({k: v for k, v in stats.items() if k.startswith("t5_") and "rows" not in k}), ""]

    json.dump(res, open(SUB / "results.json", "w"), indent=2)
    json.dump(stats, open(WORK / "stats.json", "w"), indent=1)
    (SUB / "code" / "analysis.md").write_text("\n".join(A), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
