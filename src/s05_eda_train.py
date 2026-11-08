"""Day 1, step 8: exploratory data analysis on training rows only.

Reads ONLY data/processed/train_rows.csv. It never opens the full cleaned file,
the split assignment or any test row. Plots are descriptive; nothing is fitted.

History: on 7 Nov 2026 only two report sentences about the duplicate policy were updated to the approved
wording. No plotting or analysis logic changed and the script was not re-run. Pre-edit copy:
archive/day1_final_pass_pre_edit_2026-11-07/src/s05_eda_train.py

Run with:  MPLCONFIGDIR=<scratch> <env python> -I -B src/s05_eda_train.py
Writes:    reports/figures/fig01..fig04 (PNG), reports/05_eda_training.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prep import FIGURES_DIR, REPORTS_DIR, SYMPTOM_COLUMNS, TRAIN_ROWS, md_table  # noqa: E402

# Palette: categorical slots 1-2 of the reference palette, validated (light surface).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"Positive": "#2a78d6", "Negative": "#eb6834"}
CLASSES = ["Positive", "Negative"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

train = pd.read_csv(TRAIN_ROWS)
train["class"] = np.where(train["label"] == 1, "Positive", "Negative")
VIEWS = {"All training rows": train, "Deduplicated training rows": train[train["is_first_copy"] == 1]}
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
figs: list[tuple[str, str]] = []


def save(fig, name: str, caption: str) -> None:
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figs.append((name, caption))


# --- Fig 1: class balance --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.6))
width, gap = 0.36, 0.02
x = np.arange(len(VIEWS))
for i, cls in enumerate(CLASSES):
    vals = [int((v["class"] == cls).sum()) for v in VIEWS.values()]
    tot = [len(v) for v in VIEWS.values()]
    pos = x + (i - 0.5) * (width + gap)
    ax.bar(pos, vals, width=width, color=COLOR[cls], label=cls, zorder=3)
    for p, v, t in zip(pos, vals, tot):
        ax.text(p, v + 4, f"{v} ({100 * v / t:.1f}%)", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(x, list(VIEWS))
ax.set_ylabel("Rows")
ax.grid(axis="x", visible=False)
ax.set_title("Class balance in the training partition")
ax.legend(loc="upper right")
save(fig, "fig01_class_balance_train.png", "Class counts in the training partition, all rows versus deduplicated.")

# --- Fig 2: age distribution by class (share of class per 5-year bin) -----
bins = np.arange(15, 95, 5)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (vname, v) in zip(axes, VIEWS.items()):
    for cls in CLASSES:
        ages = v.loc[v["class"] == cls, "age"]
        counts, _ = np.histogram(ages, bins=bins)
        share = 100 * counts / max(len(ages), 1)
        ax.stairs(share, bins, baseline=None, color=COLOR[cls], linewidth=2, label=f"{cls} (n={len(ages)})")
    ax.set_title(vname)
    ax.set_xlabel("Age (5-year bins)")
    ax.legend(loc="upper right")
axes[0].set_ylabel("Share of class (%)")
fig.suptitle("Age distribution by class (training rows only)", fontweight="bold", y=1.02)
save(fig, "fig02_age_by_class_train.png", "Age distribution within each class, shown as the share of that class in each 5-year bin.")

# --- Fig 3: symptom prevalence by class (dot plot) -------------------------
prev = {vn: v.groupby("class")[SYMPTOM_COLUMNS].mean().T * 100 for vn, v in VIEWS.items()}
order = (prev["All training rows"]["Positive"] - prev["All training rows"]["Negative"]).sort_values().index
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
for ax, (vname, p) in zip(axes, prev.items()):
    p = p.loc[order]
    y = np.arange(len(order))
    ax.hlines(y, p.min(axis=1), p.max(axis=1), color=GRID, linewidth=2, zorder=2)
    for cls in CLASSES:
        ax.scatter(p[cls], y, s=46, color=COLOR[cls], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=cls)
    ax.set_yticks(y, [s.replace("_", " ") for s in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% answering Yes")
    ax.set_title(vname)
    ax.grid(axis="y", visible=False)
axes[1].legend(loc="lower right")
fig.suptitle("Symptom prevalence by class (training rows only)", fontweight="bold", y=1.0)
save(fig, "fig03_symptom_prevalence_train.png",
     "Percentage answering Yes to each symptom, by class. Symptoms are sorted by the Positive − Negative gap in the all-rows view.")

# --- Fig 4: gender by class --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (vname, v) in zip(axes, VIEWS.items()):
    genders = ["Female", "Male"]
    gx = np.arange(len(genders))
    for i, cls in enumerate(CLASSES):
        vals = [int(((v["gender_male"] == (g == "Male")) & (v["class"] == cls)).sum()) for g in genders]
        pos = gx + (i - 0.5) * (width + gap)
        ax.bar(pos, vals, width=width, color=COLOR[cls], label=cls, zorder=3)
        for pp, val in zip(pos, vals):
            ax.text(pp, val + 2, str(val), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(gx, genders)
    ax.grid(axis="x", visible=False)
    ax.set_title(vname)
axes[0].set_ylabel("Rows")
axes[1].legend(loc="upper left")
fig.suptitle("Gender by class (training rows only)", fontweight="bold", y=1.02)
save(fig, "fig04_gender_by_class_train.png", "Row counts by gender and class.")

# --- Report --------------------------------------------------------------------
gap_tbl = pd.DataFrame({
    f"{vn}: Positive − Negative (pp)": (p["Positive"] - p["Negative"]).round(1) for vn, p in prev.items()
}).loc[order[::-1]]
age_tbl = pd.concat({vn: v.groupby("class")["age"].describe()[["count", "mean", "50%", "min", "max"]].round(1)
                     for vn, v in VIEWS.items()}).reset_index().rename(columns={"level_0": "view", "50%": "median"})
gender_tbl = pd.concat({vn: v.groupby(np.where(v["gender_male"] == 1, "Male", "Female"))["label"]
                        .agg(rows="size", positive="sum").assign(pct_positive=lambda d: (100 * d.positive / d.rows).round(1))
                        for vn, v in VIEWS.items()}).reset_index().rename(columns={"level_0": "view", "level_1": "gender"})

L = [
    "# 05 · Exploratory data analysis (training rows only)",
    "",
    f"Input: `data/processed/train_rows.csv` only ({len(train)} rows; "
    f"{int(train['is_first_copy'].sum())} after deduplication). **No test rows were read.**",
    "Every figure shows two views: 'All training rows' is the primary analysis (all records) and 'Deduplicated training rows' is the sensitivity analysis (one record per unique questionnaire profile).",
    "All results are descriptive. Nothing was fitted, selected or engineered.",
    "",
]
for name, cap in figs:
    L += [f"![{cap}](figures/{name})", "", f"*{cap}*", ""]
L += [
    "## Symptom gap: Positive minus Negative, percentage points",
    "",
    md_table(gap_tbl, index=True),
    "",
    "## Age by class",
    "",
    md_table(age_tbl),
    "",
    "## Gender by class",
    "",
    md_table(gender_tbl),
    "",
    "## Reading notes",
    "",
    "- These are associations within one hospital's questionnaire sample. They are not causal and",
    "  are not population prevalence.",
    "- Where the two views differ, the difference comes from the identical questionnaire records, which the primary",
    "  analysis keeps and the sensitivity analysis counts once. The approved duplicate policy is in docs/PROJECT_PLAN.md §3.5.",
    "- No feature-selection or preprocessing decision has been made from these plots. Age bands for",
    "  Categorical NB were fixed in the plan (§3.10) before the EDA.",
]
(REPORTS_DIR / "05_eda_training.md").write_text("\n".join(L) + "\n")
print("figures:", [f for f, _ in figs])
print(gap_tbl.to_string())
print(gender_tbl.to_string(index=False))
