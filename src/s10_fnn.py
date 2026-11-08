"""Optional small FNN (secondary experiment). It takes no part in model selection and is never deployed.

Fixed settings: the same preprocessing as Logistic Regression (age standardised, the other 15 columns unchanged);
Linear(16,16) -> ReLU -> Dropout(0.2) -> Linear(16,8) -> ReLU -> Linear(8,1); BCEWithLogitsLoss; Adam (lr 1e-3,
weight decay 1e-4); batch size 32; at most 200 epochs; early-stopping patience 20; seeds 0 to 4; CPU only;
threshold 0.5 on the sigmoid; no hyperparameter search.

Training protocol, identical for every outer CV fold and for the final test evaluation ("this fold" = the outer
training fold in CV, the full training set of the view in the final evaluation):
  Phase 1 (choose the epoch count): split this fold by profile group into inner-training and inner-validation parts
    (first split of StratifiedGroupKFold(5, shuffle=True, random_state=seed)). Fit the age scaler on the inner-training
    part only. Train with early stopping on the inner-validation BCE loss (eval mode). An epoch counts as an improvement
    only when its loss is strictly lower than the previous best; ties keep the earliest epoch; training stops after 20
    consecutive epochs without improvement, at most 200 epochs. The selected epoch count E is the epoch of the best loss.
  Phase 2 (final fit): re-seed, build a fresh network and a fresh age scaler fitted on the whole of this fold, and train
    on all of it for exactly E epochs (no early stopping, no inner split, no weight restoring).
  Evaluation: only the phase-2 network, only on the untouched outer validation fold (CV) or the untouched test set.

Reproducibility: for every run the Python, NumPy and PyTorch seeds are set to the current seed; a separately seeded
torch.Generator drives the shuffled DataLoader; CPU only; torch.use_deterministic_algorithms(True); one thread. A
determinism self-check repeats one run and aborts if the result is not bit-identical.

The test rows are read only in the final-evaluation part, after the CV part has finished and written its CSV.

Run with:  <env python> -I -B src/s10_fnn.py
Writes:    reports/fnn_cv_results.csv, reports/fnn_results.json
"""
from __future__ import annotations

import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch import nn  # noqa: E402

from mlcommon import (BOOTSTRAP_REPS, BOOTSTRAP_SEED, FEATURES, METRICS, REPORTS, ROOT, VIEWS, auc_score,  # noqa: E402
                      bootstrap_summary, confusion_counts, draw_group_weights, jsonable, load_training, make_splits,
                      metrics_from_counts, roc_points, summarize, verify_inputs, view_frame)

SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS, PATIENCE, BATCH, LR, WEIGHT_DECAY = 200, 20, 32, 1e-3, 1e-4
SPLIT_ASSIGN = ROOT / "data" / "splits" / "split_assignment.csv"
CLEANED = ROOT / "data" / "processed" / "cleaned_all_rows.csv"
TEST_INPUT_SHA256 = {
    SPLIT_ASSIGN: "8f0077e57e4b59b868929413446b3b445a3be9807b4bb1f4645dad221817b31e",
    CLEANED: "2678581eedfbffae3d1d2a4668b3cbfffca78bd5369bcbc2a6981ac4cd810ff8",
}

torch.use_deterministic_algorithms(True)       # raises if an operation has no deterministic implementation
torch.set_num_threads(1)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(X: np.ndarray, y: np.ndarray, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)                        # separate generator for the shuffled batches
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, generator=g, drop_last=False)


def transform(arr: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    out = np.array(arr, dtype=float, copy=True)
    out[:, 0] = scaler.transform(out[:, [0]])[:, 0]
    return out.astype(np.float32)


def train_one_epoch(net, opt, loss_fn, loader) -> None:
    net.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(net(xb), yb).backward()
        opt.step()


def phase1_epochs(Xtr, ytr, Xva, yva, seed: int) -> int:
    seed_everything(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = make_loader(Xtr, ytr, seed)
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_one_epoch(net, opt, loss_fn, loader)
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(Xv), yv).item()
        if val_loss < best_loss:               # an improvement only if strictly lower; ties keep the earliest epoch
            best_loss, best_epoch, stale = val_loss, epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best_epoch


def phase2_net(Xall, yall, epochs: int, seed: int):
    seed_everything(seed)
    net = Net()                                # a fresh network
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = make_loader(Xall, yall, seed)
    for _ in range(epochs):
        train_one_epoch(net, opt, loss_fn, loader)
    net.eval()
    return net


def inner_split(y: np.ndarray, groups: np.ndarray, seed: int):
    tr, va = next(iter(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups)))
    assert len(set(y[tr])) == 2 and len(set(y[va])) == 2, "inner split lacks a class"
    assert not (set(groups[tr]) & set(groups[va])), "a profile is on both sides of the inner split"
    return tr, va


def run_protocol(X_fold: pd.DataFrame, y_fold: np.ndarray, g_fold: np.ndarray, X_eval: pd.DataFrame, seed: int):
    """Phase 1 (epoch count E) and phase 2 (fresh network, E epochs, whole fold). Returns (E, probabilities on X_eval)."""
    arr, yf = X_fold.to_numpy(dtype=float), y_fold.astype(np.float32)
    tr, va = inner_split(y_fold, g_fold, seed)
    sc1 = StandardScaler().fit(arr[tr][:, [0]])                      # inner-training part only
    epochs = phase1_epochs(transform(arr[tr], sc1), yf[tr], transform(arr[va], sc1), yf[va], seed)
    sc2 = StandardScaler().fit(arr[:, [0]])                          # the whole fold
    net = phase2_net(transform(arr, sc2), yf, epochs, seed)
    with torch.no_grad():
        prob = torch.sigmoid(net(torch.from_numpy(transform(X_eval.to_numpy(dtype=float), sc2)))).numpy().astype(float)
    return epochs, prob


def metrics_for(y, prob) -> dict:
    m = metrics_from_counts(*confusion_counts(y, (prob > 0.5).astype(int)))
    m["roc_auc"] = auc_score(y, prob)
    return m


def load_test(train_row_ids) -> dict:
    """Test rows are read only here, in the final-evaluation part."""
    verify_inputs(TEST_INPUT_SHA256)
    assign = pd.read_csv(SPLIT_ASSIGN)
    cleaned = pd.read_csv(CLEANED)
    test = cleaned[cleaned["row_id"].isin(set(assign.loc[assign["partition"] == "test", "row_id"]))].reset_index(drop=True)
    assert test.columns.tolist()[4:20] == FEATURES
    assert len(test) == 104 and test["profile_id"].nunique() == 49 and test["label"].nunique() == 2
    assert not (set(test["row_id"]) & set(train_row_ids))
    sens = test[test["is_first_copy"] == 1].reset_index(drop=True)
    assert len(sens) == 49 and sens["profile_id"].nunique() == 49 and sens["label"].nunique() == 2
    return {"primary": test, "sensitivity": sens}


t0 = time.time()
frame = load_training()

# determinism self-check: the same run twice must be bit-identical
vf0 = view_frame(frame, "primary")
_r, _k, tr0, va0 = make_splits(vf0)[0]
y0, g0 = vf0["label"].to_numpy(), vf0["profile_id"].to_numpy()
a = run_protocol(vf0[FEATURES].iloc[tr0], y0[tr0], g0[tr0], vf0[FEATURES].iloc[va0], 0)
b = run_protocol(vf0[FEATURES].iloc[tr0], y0[tr0], g0[tr0], vf0[FEATURES].iloc[va0], 0)
if a[0] != b[0] or not np.array_equal(a[1], b[1]):
    raise SystemExit("ABORT: the FNN run is not bit-identical when repeated with the same seed (non-deterministic).")
print(f"determinism self-check passed (epochs {a[0]}, identical probabilities); {time.time() - t0:.1f} s", flush=True)

# --- cross-validation on the 15 frozen folds (training data only) -----------------------------------------
cv_rows = []
for view in VIEWS:
    vf = view_frame(frame, view)
    X, y, g = vf[FEATURES], vf["label"].to_numpy(), vf["profile_id"].to_numpy()
    for (r, k, tr, va) in make_splits(vf):
        for seed in SEEDS:
            epochs, prob = run_protocol(X.iloc[tr], y[tr], g[tr], X.iloc[va], seed)
            cv_rows.append({"view": view, "repeat": r, "fold": k, "seed": seed, "epochs": epochs, **metrics_for(y[va], prob)})
        print(f"[{view:11s}] repeat {r} fold {k} done; {time.time() - t0:.0f} s", flush=True)
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(REPORTS / "fnn_cv_results.csv", index=False)      # written before any test row is read

views_out = {}
for view in VIEWS:
    d = cv_df[cv_df["view"] == view]
    per_fold = d.groupby(["repeat", "fold"])[list(METRICS)].mean()          # average over the 5 seeds per fold
    agg = {m: summarize(per_fold[m].to_numpy()) for m in METRICS}
    views_out[view] = {"cv": {"metrics_mean": {m: agg[m][0] for m in METRICS}, "metrics_sd": {m: agg[m][1] for m in METRICS},
                              "epochs": {"min": int(d["epochs"].min()), "median": float(d["epochs"].median()),
                                         "max": int(d["epochs"].max()), "n": int(len(d))}}}

# --- final test evaluation (the only place the test rows are read) --------------------------------------------
tests = load_test(frame["row_id"])
for view in VIEWS:
    vf = view_frame(frame, view)
    X, y, g = vf[FEATURES], vf["label"].to_numpy(), vf["profile_id"].to_numpy()
    te = tests[view]
    Xte, yte = te[FEATURES], te["label"].to_numpy()
    W = draw_group_weights(te["profile_id"].to_numpy(), BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    probs, epochs_per_seed, per_seed = [], [], []
    for seed in SEEDS:
        epochs, prob = run_protocol(X, y, g, Xte, seed)
        probs.append(prob)
        epochs_per_seed.append(epochs)
        per_seed.append(metrics_for(yte, prob))
    mean_prob = np.mean(probs, axis=0)
    pred = (mean_prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_counts(yte, pred)
    sa_metrics = metrics_for(yte, mean_prob)
    boot = bootstrap_summary(W, yte, pred, mean_prob)
    fpr, tpr = roc_points(yte, mean_prob)
    across = {m: summarize([p[m] for p in per_seed]) for m in METRICS}
    views_out[view]["test"] = {
        "epochs_per_seed": epochs_per_seed,
        "metrics_mean_across_seeds": {m: across[m][0] for m in METRICS},
        "metrics_sd_across_seeds": {m: across[m][1] for m in METRICS},
        "seed_averaged": {"metrics": sa_metrics, "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                          "ci95": {m: [boot[m]["lo"], boot[m]["hi"]] for m in METRICS},
                          "n_undefined_bootstrap": {m: boot[m]["n_undefined"] for m in METRICS},
                          "roc": {"fpr": fpr, "tpr": tpr}},
    }
    print(f"[{view:11s}] test, seed-averaged probabilities: " + "  ".join(f"{m} {sa_metrics[m]:.3f}" for m in METRICS)
          + f"  | epochs per seed {epochs_per_seed}", flush=True)

meta = {
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "torch": torch.__version__, "python": platform.python_version(), "numpy": np.__version__,
    "seeds": SEEDS,
    "settings": {"architecture": "Linear(16,16)-ReLU-Dropout(0.2)-Linear(16,8)-ReLU-Linear(8,1)", "loss": "BCEWithLogitsLoss",
                 "optimiser": f"Adam lr {LR}, weight decay {WEIGHT_DECAY}", "batch_size": BATCH, "max_epochs": MAX_EPOCHS,
                 "patience": PATIENCE, "threshold": 0.5},
    "protocol": ("Two phases per fold and seed. Phase 1 chooses the epoch count E by early stopping on a group-separated inner "
                 "validation split (first split of StratifiedGroupKFold(5, shuffle=True, random_state=seed); a strictly lower inner-validation "
                 "BCE loss counts as an improvement, ties keep the earliest epoch, patience 20, at most 200 epochs). Phase 2 trains a fresh "
                 "network and a fresh age scaler on the whole outer training fold for exactly E epochs, and only that network is evaluated, "
                 "on the untouched outer validation fold (CV) or the untouched test set (final). The final network is never trained on the "
                 "inner-training subset alone. CV: each metric is averaged over the 5 seeds per fold, then mean ± SD over the 15 folds."),
    "reproducibility": ("For every run the Python, NumPy and PyTorch seeds are set to the current seed (0 to 4) and a separately seeded "
                        "torch.Generator drives the shuffled DataLoader. CPU only, torch.use_deterministic_algorithms(True), one thread; "
                        "no operation raised a determinism error, and a repeated run was bit-identical."),
    "determinism": {"use_deterministic_algorithms": True, "num_threads": 1, "self_check_bit_identical": True},
}
(REPORTS / "fnn_results.json").write_text(json.dumps(jsonable({"meta": meta, "views": views_out}), indent=2, sort_keys=True) + "\n")
print(f"wrote reports/fnn_cv_results.csv ({len(cv_df)} rows) and reports/fnn_results.json; elapsed {time.time() - t0:.0f} s")
