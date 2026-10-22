#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:25:35 2026

@author: herttaleinonen
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math

# ------------------------------------------------------------------
# Journal style
# ------------------------------------------------------------------
class R:
    BASE_SIZE    = 18
    FONT_FAMILY  = "Helvetica"
    LW_MEAN      = 2.6 #1.4
    LW_SPAG      = 2.6 #0.5
    LW_ERR       = 1.8
    PT_SIZE      = 6
    MARKER       = "o"
    CAPSIZE      = 4
    ALPHA_SPAG   = 0.10
    GRID_COLOR   = "#e5e5e5"
    GRID_LW      = 0.8
    
    
TITLE_FS = R.BASE_SIZE * 1.2
LABEL_FS = R.BASE_SIZE * 1.2
TICK_FS  = R.BASE_SIZE * 1.2
LEGEND_FS = R.BASE_SIZE * 1.15

def apply_r_style():
    mpl.rcParams.update({
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           R.BASE_SIZE,
        "axes.titlesize":      R.BASE_SIZE,
        "axes.labelsize":      R.BASE_SIZE,
        "xtick.labelsize":     R.BASE_SIZE * 0.8,
        "ytick.labelsize":     R.BASE_SIZE * 0.8,
        "font.weight":         "normal",
        "axes.titleweight":    "normal",
        "axes.labelweight":    "normal",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.spines.left":    True,
        "axes.spines.bottom":  True,
        "axes.linewidth":      0.8,
        "axes.edgecolor":      "#333333",
        "axes.facecolor":      "white",
        "figure.facecolor":    "white",
        "axes.grid":           True,
        "axes.grid.axis":      "y",
        "grid.color":          R.GRID_COLOR,
        "grid.linewidth":      R.GRID_LW,
        "grid.linestyle":      "-",
        "xtick.major.size":    3,
        "ytick.major.size":    3,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "legend.frameon":      False,
        "legend.fontsize":     R.BASE_SIZE * 0.85,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
    })

def mosaic_style():
    mpl.rcParams.update({
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           10,
        "axes.titlesize":      10,
        "axes.labelsize":      10,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.fontsize":     9,
        "font.weight":         "normal",
        "axes.grid":           True,
        "axes.grid.axis":      "y",
        "grid.color":          R.GRID_COLOR,
        "grid.linewidth":      R.GRID_LW,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
    })

apply_r_style()

# -----------------------
# Settings
# -----------------------
CSV_PATH = "replay_model_results_kfold_pooled.csv"
OUTDIR = "figures"
USE_CORRECT_ONLY_FOR_HUMAN_RT = False
MODEL_RT_FALLBACK_COL = "model_rt_mean_s"
DPRIME_YLIM = (1.15, 5.0)   # fixed y-range for all d' plots

COLOR_TP    = "#1f77b4"
COLOR_TA    = "#d62728"
COLOR_HUMAN = "#17becf"
COLOR_MODEL = "#2ca02c"

SPEED_ORDER = [0, 100, 200, 300, 400]
SPEED_DEG_EXACT = {0: 0.000, 100: 2.703, 200: 5.406, 300: 8.109, 400: 10.812}
SPEED_DEG_LABELS = {0: "0", 100: "3", 200: "5.5", 300: "8", 400: "11"}
SPEED_LABEL_TEXT = "Velocity (deg/s)"

def speed_display(x_px):
    x_px = np.asarray(x_px, dtype=int)
    return np.array([SPEED_DEG_EXACT[v] for v in x_px])

def speed_ticks_and_labels():
    ticks  = [SPEED_DEG_EXACT[v] for v in SPEED_ORDER]
    labels = [SPEED_DEG_LABELS[v] for v in SPEED_ORDER]
    return ticks, labels

def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def accuracy_by_pp(df_in):
    return (df_in.groupby(["participant","speed_px_s_used"], as_index=False)
                 .agg(human_acc=("human_acc","mean"), model_acc=("model_acc","mean")))

def rates_to_dprime(hit, n_pos, fa, n_neg):
    """Log-linear corrected d'. hit/fa are rates, n_pos/n_neg the number of
    target-present / target-absent trials in that participant x speed cell.
    For the model the rate is a mean probability, so rate * n is its expected count."""
    hit, fa = np.asarray(hit, float), np.asarray(fa, float)
    n_pos, n_neg = np.asarray(n_pos, float), np.asarray(n_neg, float)
    H = (hit * n_pos + 0.5) / (n_pos + 1)
    F = (fa  * n_neg + 0.5) / (n_neg + 1)
    return norm.ppf(H) - norm.ppf(F)

def pp_label(pp):
    return "pp" + str(int(pp.replace("kh", "")))

def add_panel_label(ax, label, fontsize=24):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="normal",
            va="top", ha="left")

def plot_mean_with_err(ax, x, y, err, color, label, marker="o", linestyle="-"):
    ax.plot(x, y, marker=marker, linestyle=linestyle, color=color,
            label=label, linewidth=R.LW_MEAN, markersize=R.PT_SIZE)
    ax.errorbar(x, y, yerr=err, fmt="none", color=color,
                elinewidth=R.LW_ERR, capsize=R.CAPSIZE, capthick=R.LW_ERR)

def set_speed_axis(ax):
    ticks, labels = speed_ticks_and_labels()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel(SPEED_LABEL_TEXT)
    ax.set_xlabel(SPEED_LABEL_TEXT, fontsize=LABEL_FS) 

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(CSV_PATH)
df["human_acc"] = df["human_correct"].astype(float)
df["model_acc"] = ((df["model_p_present"] >= 0.5).astype(int) == df["human_target_present"]).astype(float)

needed = ["participant","speed_px_s_used","human_target_present",
          "human_response","human_correct","human_rt_s","model_p_present"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df["speed_px_s_used"] = df["speed_px_s_used"].astype(int)
df = df[df["speed_px_s_used"].isin(SPEED_ORDER)].copy()
df["model_p_absent"] = 1.0 - df["model_p_present"]

MODEL_RT_PRESENT_COL = pick_first_existing(df, ["model_rt_present_mean_s","model_rt_present_median_s"])
MODEL_RT_ABSENT_COL  = pick_first_existing(df, ["model_rt_absent_mean_s", "model_rt_absent_median_s"])
HAS_CONDITIONAL_MODEL_RT = (MODEL_RT_PRESENT_COL is not None) and (MODEL_RT_ABSENT_COL is not None)

if not HAS_CONDITIONAL_MODEL_RT:
    if MODEL_RT_FALLBACK_COL not in df.columns:
        raise ValueError(f"No conditional model RT columns and fallback '{MODEL_RT_FALLBACK_COL}' missing.")
    print("[WARN] Using unconditional RT:", MODEL_RT_FALLBACK_COL)

def response_rates_by_pp(df_in):
    tp = df_in[df_in["human_target_present"] == 1]
    ta = df_in[df_in["human_target_present"] == 0]
    hit_pp = (tp.groupby(["participant","speed_px_s_used"], as_index=False)
                .agg(human_hit=("human_response","mean"),
                     model_hit=("model_p_present","mean"),
                     n_pos=("human_response","size")))
    fa_pp  = (ta.groupby(["participant","speed_px_s_used"], as_index=False)
                .agg(human_fa=("human_response","mean"),
                     model_fa=("model_p_present","mean"),
                     n_neg=("human_response","size")))
    return hit_pp, fa_pp

def make_dprime_pp(hit_pp, fa_pp):
    d = hit_pp.merge(fa_pp, on=["participant","speed_px_s_used"], how="inner")
    d["human_dprime"] = rates_to_dprime(d["human_hit"], d["n_pos"], d["human_fa"], d["n_neg"])
    d["model_dprime"] = rates_to_dprime(d["model_hit"], d["n_pos"], d["model_fa"], d["n_neg"])
    return d

hit_pp, fa_pp = response_rates_by_pp(df)
acc_pp = accuracy_by_pp(df)
dprime_pp = make_dprime_pp(hit_pp, fa_pp)


mode_tag = "pooled"  

ensure_outdir(OUTDIR)
dprime_pp.to_csv(f"{OUTDIR}/dprime_pp_{mode_tag}.csv", index=False)


def rt_pp_for_subset(df_in, model_rt_col, *, condition_on="human_correct",
                     deadline_s=3.5, nondecision_s=0.25):
    tmp = df_in.copy()
    if condition_on == "human_correct":
        tmp = tmp[tmp["human_correct"] == 1].copy()
    tmp[model_rt_col] = pd.to_numeric(tmp[model_rt_col], errors="coerce")
    tmp[model_rt_col] = tmp[model_rt_col].fillna(deadline_s + nondecision_s)
    return (tmp.groupby(["participant","speed_px_s_used"], as_index=False)
               .agg(human_rt=("human_rt_s","mean"), model_rt=(model_rt_col,"mean")))

rt_tp = df[df["human_target_present"] == 1].copy()
rt_ta = df[df["human_target_present"] == 0].copy()

if HAS_CONDITIONAL_MODEL_RT:
    rt_pp_tp = rt_pp_for_subset(rt_tp, MODEL_RT_PRESENT_COL)
    rt_pp_ta = rt_pp_for_subset(rt_ta, MODEL_RT_ABSENT_COL)
    
    #rt_pp_tp.to_csv(f"{OUTDIR}/rt_pp_tp_center.csv", index=False)
    #rt_pp_ta.to_csv(f"{OUTDIR}/rt_pp_ta_center.csv", index=False)

    rt_pp_ta.to_csv(f"{OUTDIR}/rt_pp_ta_{mode_tag}.csv", index=False)
    rt_pp_tp.to_csv(f"{OUTDIR}/rt_pp_tp_{mode_tag}.csv", index=False)
    
    rt_tp_tag = f"TP_correct_model={MODEL_RT_PRESENT_COL}"
    rt_ta_tag = f"TA_correct_model={MODEL_RT_ABSENT_COL}"
else:
    rt_pp_tp = rt_pp_for_subset(rt_tp, MODEL_RT_FALLBACK_COL)
    rt_pp_ta = rt_pp_for_subset(rt_ta, MODEL_RT_FALLBACK_COL)
    rt_tp_tag = f"TP_correct_model={MODEL_RT_FALLBACK_COL}"
    rt_ta_tag = f"TA_correct_model={MODEL_RT_FALLBACK_COL}"
    


def groupify(pp_df, human_col, model_col, prefix):
    g = (pp_df.groupby("speed_px_s_used", as_index=False)
              .agg(human_mean=(human_col,"mean"), human_sem=(human_col,sem),
                   model_mean=(model_col,"mean"), model_sem=(model_col,sem),
                   n_pp=("participant","nunique")))
    g = g.rename(columns={
        "human_mean": f"human_{prefix}_mean", "human_sem": f"human_{prefix}_sem",
        "model_mean": f"model_{prefix}_mean", "model_sem": f"model_{prefix}_sem",
    })
    g["speed_px_s_used"] = pd.Categorical(g["speed_px_s_used"], SPEED_ORDER, ordered=True)
    return g.sort_values("speed_px_s_used")

def rt_groupify(rt_pp):
    g = (rt_pp.groupby("speed_px_s_used", as_index=False)
              .agg(human_rt_mean=("human_rt","mean"), human_rt_sem=("human_rt",sem),
                   model_rt_mean=("model_rt","mean"), model_rt_sem=("model_rt",sem),
                   n_pp=("participant","nunique")))
    g["speed_px_s_used"] = pd.Categorical(g["speed_px_s_used"], SPEED_ORDER, ordered=True)
    return g.sort_values("speed_px_s_used")

hit_group    = groupify(hit_pp,    "human_hit",    "model_hit",    "hit")
fa_group     = groupify(fa_pp,     "human_fa",     "model_fa",     "fa")
acc_group    = groupify(acc_pp,    "human_acc",    "model_acc",    "acc")
dprime_group = groupify(dprime_pp, "human_dprime", "model_dprime", "dprime")
rt_group_tp  = rt_groupify(rt_pp_tp)
rt_group_ta  = rt_groupify(rt_pp_ta)

ensure_outdir(OUTDIR)

# =============================================================
# SINGLE-PANEL PLOTS
# =============================================================
apply_r_style()

for rt_group, tag, panel_label in [
    (rt_group_tp, rt_tp_tag, "a."),
    (rt_group_ta, rt_ta_tag, "b."),
]:
    color = COLOR_TP if rt_group is rt_group_tp else COLOR_TA
    x = speed_display(rt_group["speed_px_s_used"].values)
    fig, ax = plt.subplots(figsize=(7, 5))
    add_panel_label(ax, panel_label)
    plot_mean_with_err(ax, x, rt_group["human_rt_mean"], rt_group["human_rt_sem"],
                       color=color, label="Human", marker="o")
    plot_mean_with_err(ax, x, rt_group["model_rt_mean"], rt_group["model_rt_sem"],
                       color=color, label="Model", marker="^", linestyle="--")
    set_speed_axis(ax)
    ax.set_ylabel("Reaction time (s)", fontsize=LABEL_FS)
    ax.set_ylim(1.0, 3.3)
    ax.legend()
    fig.tight_layout()
    safe_tag = tag.replace("/","_").replace(" ","_").replace("=","_")
    fig.savefig(f"{OUTDIR}/rt_vs_speed_{safe_tag}.png")
    fig.savefig(f"{OUTDIR}/rt_vs_speed_{safe_tag}.pdf", format="pdf")
    plt.close()

    x = speed_display(dprime_group["speed_px_s_used"].values)
    fig, ax = plt.subplots(figsize=(7, 5))
    add_panel_label(ax, "c.")
    plot_mean_with_err(ax, x, dprime_group["human_dprime_mean"], dprime_group["human_dprime_sem"],
                       color=COLOR_HUMAN, label="Human")
    plot_mean_with_err(ax, x, dprime_group["model_dprime_mean"], dprime_group["model_dprime_sem"],
                       color=COLOR_MODEL, label="Model", linestyle="--")
    set_speed_axis(ax)
    ax.set_ylabel("Sensitivity (d')",fontsize=LABEL_FS)   
    ax.set_ylim(*DPRIME_YLIM)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/dprime_vs_speed.png")
    fig.savefig(f"{OUTDIR}/dprime_vs_speed.pdf", format="pdf")
    plt.close()
    print(" - dprime_vs_speed.png")
    
# =============================================================
# HIT RATE and FALSE-ALARM PLOTS
# =============================================================
for grp, key, ylabel, fname in [
    (hit_group, "hit", "Hit rate", "hit_vs_speed"),
    (fa_group,  "fa",  "False-alarm rate", "fa_vs_speed"),
]:
    x = speed_display(grp["speed_px_s_used"].values)
    fig, ax = plt.subplots(figsize=(7, 5))
    add_panel_label(ax, "a.")
    plot_mean_with_err(ax, x, grp[f"human_{key}_mean"], grp[f"human_{key}_sem"],
                       color=COLOR_HUMAN, label="Human")
    plot_mean_with_err(ax, x, grp[f"model_{key}_mean"], grp[f"model_{key}_sem"],
                       color=COLOR_MODEL, label="Model", linestyle="--")
    set_speed_axis(ax)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    if key == "hit":
        ax.set_ylim(0.5, 1.02)
    else:
        top = max((grp[f"human_{key}_mean"] + grp[f"human_{key}_sem"]).max(),
                  (grp[f"model_{key}_mean"] + grp[f"model_{key}_sem"]).max())
        ax.set_ylim(0.0, top * 1.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png")
    fig.savefig(f"{OUTDIR}/{fname}.pdf", format="pdf")
    plt.close()
    print(f" - {fname}.png")
    
# =============================================================
# HITS and FALSE ALARMS IN ONE PLOT
# =============================================================
x = speed_display(hit_group["speed_px_s_used"].values)
fig, ax = plt.subplots(figsize=(7, 5))
add_panel_label(ax, "d.")

# hits (target-present colour)
plot_mean_with_err(ax, x, hit_group["human_hit_mean"], hit_group["human_hit_sem"],
                   color=COLOR_TP, label="Hits, human", marker="o")
plot_mean_with_err(ax, x, hit_group["model_hit_mean"], hit_group["model_hit_sem"],
                   color=COLOR_TP, label="Hits, model", marker="^", linestyle="--")

# false alarms (target-absent colour)
plot_mean_with_err(ax, x, fa_group["human_fa_mean"], fa_group["human_fa_sem"],
                   color=COLOR_TA, label="False alarms, human", marker="o")
plot_mean_with_err(ax, x, fa_group["model_fa_mean"], fa_group["model_fa_sem"],
                   color=COLOR_TA, label="False alarms, model", marker="^", linestyle="--")

set_speed_axis(ax)
ax.set_ylabel("Response rate", fontsize=LABEL_FS)
ax.set_ylim(0.0, 1.0)
ax.legend(ncol=2, fontsize=14, loc="center right")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/hit_fa_vs_speed.png")
fig.savefig(f"{OUTDIR}/hit_fa_vs_speed.pdf", format="pdf")
plt.close()
print(" - hit_fa_vs_speed.png")


# =============================================================
# MOSAIC PLOTS
# =============================================================
mosaic_style()

def pivot_wide(pp_df, value_col):
    wide = pp_df.pivot(index="participant", columns="speed_px_s_used", values=value_col)
    for s in SPEED_ORDER:
        if s not in wide.columns:
            wide[s] = np.nan
    return wide[SPEED_ORDER].sort_index()

participants = sorted(df["participant"].unique(), key=lambda s: int(s.replace("kh","")))
n_pp  = len(participants)
ncols = 4
nrows = int(np.ceil(n_pp / ncols))

rt_tp_h_w = pivot_wide(rt_pp_tp, "human_rt")
rt_tp_m_w = pivot_wide(rt_pp_tp, "model_rt")
rt_ta_h_w = pivot_wide(rt_pp_ta, "human_rt")
rt_ta_m_w = pivot_wide(rt_pp_ta, "model_rt")

fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows), sharex=True, sharey=False)
axes = np.atleast_1d(axes).ravel()
fig.text(0.01, 0.98, "a.", fontsize=54, fontweight="normal", va="top", ha="left")
x = speed_display(SPEED_ORDER)
for i, pp in enumerate(participants):
    ax = axes[i]
    ax.plot(x, rt_tp_h_w.loc[pp].values, marker="o", linestyle="-", color=COLOR_TP, label="Target-present Human")
    ax.plot(x, rt_tp_m_w.loc[pp].values, marker="^", linestyle="--", color=COLOR_TP, label="Target-present Model")
    ax.plot(x, rt_ta_h_w.loc[pp].values, marker="o", linestyle="-", color=COLOR_TA, label="Target-absent Human")
    ax.plot(x, rt_ta_m_w.loc[pp].values, marker="^", linestyle="--", color=COLOR_TA, label="Target-absent Model")
    ax.set_title(pp_label(pp))
    ticks, labels = speed_ticks_and_labels()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=9, pad=2)
    ax.set_ylim(-0.02, 4.52)
    ax.grid(True, alpha=0.2)
for j in range(n_pp, len(axes)):
    axes[j].axis("off")
seen = {}
for ax in axes[:n_pp]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen: seen[l] = h
fig.supxlabel(SPEED_LABEL_TEXT, fontsize=54, y=0.01)
fig.supylabel("Reaction time (s)", fontsize=54, x=0.01)
fig.subplots_adjust(left=0.1, bottom=0.1, top=0.85, wspace=0.25, hspace=0.35)
fig.legend(seen.values(), seen.keys(), loc="upper left", ncol=2, fontsize=26,
           frameon=False, bbox_to_anchor=(0.08, 0.96))
fig.savefig(f"{OUTDIR}/individual_fits_mosaic.png", dpi=300)
fig.savefig(f"{OUTDIR}/individual_fits_mosaic.pdf", format="pdf")
plt.close()

dprime_h_w = pivot_wide(dprime_pp, "human_dprime")
dprime_m_w = pivot_wide(dprime_pp, "model_dprime")
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows), sharex=True, sharey=False)
axes = np.atleast_1d(axes).ravel()
fig.text(0.01, 0.98, "b.", fontsize=54, fontweight="normal", va="top", ha="left")
x = speed_display(SPEED_ORDER)
for i, pp in enumerate(participants):
    ax = axes[i]
    ax.plot(x, dprime_h_w.loc[pp].values, marker="o", linestyle="-", color=COLOR_HUMAN, label="Human")
    ax.plot(x, dprime_m_w.loc[pp].values, marker="^", linestyle="--", color=COLOR_MODEL, label="Model")
    ax.set_ylim(*DPRIME_YLIM)
    ax.set_title(pp_label(pp))
    ticks, labels = speed_ticks_and_labels()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=9, pad=2)
    ax.grid(True, alpha=0.2)
for j in range(n_pp, len(axes)):
    axes[j].axis("off")
seen = {}
for ax in axes[:n_pp]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen: seen[l] = h
fig.supxlabel(SPEED_LABEL_TEXT, fontsize=54, y=0.01)
fig.supylabel("Sensitivity (d')", fontsize=54, x=0.01)
fig.subplots_adjust(left=0.1, bottom=0.1, top=0.85, wspace=0.25, hspace=0.35)
fig.legend(seen.values(), seen.keys(), loc="upper left", ncol=2, fontsize=26,
           frameon=False, bbox_to_anchor=(0.08, 0.96))
fig.savefig(f"{OUTDIR}/individual_dprime_mosaic.png", dpi=300)
fig.savefig(f"{OUTDIR}/individual_dprime_mosaic.pdf", format="pdf")
plt.close()
print(" - individual_dprime_mosaic.png")

# ---- Plot 5: Fitted parameters -------------
apply_r_style()
FIT_CSV = "fitted_params_kfold_zscore_no-alpha.csv"
if os.path.exists(FIT_CSV):
    fit = pd.read_csv(FIT_CSV)
    needed_fit = ["participant", "eta", "theta"]
    missing_fit = [c for c in needed_fit if c not in fit.columns]
    if missing_fit:
        print(f"[WARN] {FIT_CSV} missing columns: {missing_fit}. Skipping.")
    else:
        fit["eta"]   = pd.to_numeric(fit["eta"],   errors="coerce")
        fit["theta"] = pd.to_numeric(fit["theta"], errors="coerce")
        fit = fit.dropna(subset=["eta","theta"]).copy()
        fit.to_csv(f"{OUTDIR}/fitted_params_copy.csv", index=False)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        add_panel_label(ax1, "a.")
        ax1.hist(fit["eta"].values, bins=10)
        ax1.set_xlabel("η (temporal efficiency)")
        ax1.set_ylabel("Count")
        ax2.hist(fit["theta"].values, bins=10)
        ax2.set_xlabel("Θ (present bound)")
        ax2.set_ylabel("Count")
        ax3.scatter(fit["eta"].values, fit["theta"].values, s=R.PT_SIZE*8)
        ax3.set_xlabel("η")
        ax3.set_ylabel("Θ")
        for _, r in fit.iterrows():
            ax3.annotate(str(r["participant"]), (float(r["eta"]), float(r["theta"])),
                         textcoords="offset points", xytext=(3,3), fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{OUTDIR}/fitted_parameters.png")
        fig.savefig(f"{OUTDIR}/fitted_parameters.pdf", format="pdf")
        plt.close()
        print(" - fitted_parameters.png")
else:
    print(f"[WARN] {FIT_CSV} not found. Skipping.")

# Save summary tables
hit_group.to_csv(f"{OUTDIR}/summary_hit_rate_vs_speed.csv", index=False)
fa_group.to_csv(f"{OUTDIR}/summary_false_alarm_rate_vs_speed.csv", index=False)
safe_tp = f"summary_rt_vs_speed_{rt_tp_tag}.csv".replace("/","_").replace(" ","_")
safe_ta = f"summary_rt_vs_speed_{rt_ta_tag}.csv".replace("/","_").replace(" ","_")
rt_group_tp.to_csv(os.path.join(OUTDIR, safe_tp), index=False)
rt_group_ta.to_csv(os.path.join(OUTDIR, safe_ta), index=False)
acc_group.to_csv(f"{OUTDIR}/summary_accuracy_vs_speed.csv", index=False)
dprime_group.to_csv(f"{OUTDIR}/summary_dprime_vs_speed.csv", index=False)
print("Wrote figures to:", OUTDIR)

# =============================================================
# SACCADE TIMING ANALYSIS
# =============================================================

def add_panel_label(ax, label, fontsize=24, y=1.05):
    ax.text(-0.12, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="normal",
            va="top", ha="left")
    
apply_r_style()

df = pd.read_csv("saccade_prediction_table.csv")
print("Loaded rows:", len(df))
df = df.copy()
df = df[np.isfinite(df["margin"])]
df = df[np.isfinite(df["abs_dv"])]
df = df[np.isfinite(df["fix_age_s"])]
df = df[np.isfinite(df["t_sec"])]
print("After cleaning:", len(df))

USE_PRE_RESPONSE_ONLY = True
if USE_PRE_RESPONSE_ONLY:
    df = df[df["t_sec"] < df["human_rt_s"]].copy()
    print("After RT filter:", len(df))
df = df[df["t_sec"] > 0.1]

print("\n=== BINNING CHECK (margin) ===")
df["margin_q"] = pd.qcut(df["margin"], 5, labels=False, duplicates="drop")
print(df.groupby("margin_q")["fix_change_next_200"].mean())
print(df.groupby("margin_q")["fix_change_next_200"].count())

for col in ["margin", "abs_dv", "t_sec", "fix_age_s"]:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

model_absdv = smf.glm(
    formula="fix_change_next_200 ~ z_abs_dv + z_t_sec + z_fix_age_s + C(speed_px_s) + C(target_present)",
    data=df, family=sm.families.Binomial()
).fit(cov_type="cluster", cov_kwds={"groups": df["participant"]})
print("\n=== LOGISTIC REGRESSION: abs_dv ===")
print(model_absdv.summary())

df_tp = df[df["target_present"] == 1].copy()
df_tp = df_tp[np.isfinite(df_tp["target_loglr"])]
df_tp["z_target_loglr"] = ((df_tp["target_loglr"] - df_tp["target_loglr"].mean()) / df_tp["target_loglr"].std(ddof=0))
model_target = smf.glm(
    formula="fix_change_next_200 ~ z_target_loglr + z_t_sec + z_fix_age_s + C(speed_px_s)",
    data=df_tp, family=sm.families.Binomial()
).fit(cov_type="cluster", cov_kwds={"groups": df_tp["participant"]})
print("\n=== TARGET-PRESENT ONLY ===")
print(model_target.summary())

def print_effect(name, model, var):
    if var not in model.params.index:
        print(f"{name}: parameter '{var}' not in model")
        return
    print(f"{name}: beta={model.params[var]:.3f}, p={model.pvalues[var]:.3e}")

print("\n=== KEY EFFECTS ===")
print_effect("abs_dv", model_absdv, "z_abs_dv")
if "model_target" in locals() and "z_target_loglr" in model_target.params.index:
    print_effect("target_loglr", model_target, "z_target_loglr")

N_BINS = 100
df_plot = df.copy()
df_plot["dv_bin"] = pd.qcut(df_plot["abs_dv"], N_BINS, labels=False, duplicates="drop")
bin_means   = df_plot.groupby("dv_bin")["fix_change_next_200"].mean()
bin_centers = df_plot.groupby("dv_bin")["abs_dv"].mean()

fig, ax = plt.subplots(figsize=(7, 5.5))
add_panel_label(ax, "a.", y=1.08) 
ax.plot(bin_centers, bin_means, marker="o", linewidth=R.LW_MEAN, markersize=R.PT_SIZE)
ax.set_xlabel("Model evidence (| decision variable |)")
ax.set_ylabel("Saccade probability")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/plot_absdv_binned.png", dpi=300)
fig.savefig(f"{OUTDIR}/plot_absdv_binned.pdf", format="pdf")
plt.close()

mosaic_style()
N_BINS = 30
participants = sorted(df["participant"].dropna().unique(),
                      key=lambda s: int(str(s).replace("kh","")))
n_pp  = len(participants)
ncols = 4
nrows = math.ceil(n_pp / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
fig.text(0.01, 0.98, "b.", fontsize=54, fontweight="normal", va="top", ha="left")

for i, pp in enumerate(participants):
    ax = axes[i]
    df_pp = df[df["participant"] == pp].copy()
    if len(df_pp) < N_BINS:
        ax.axis("off")
        continue
    df_pp["dv_bin"] = pd.qcut(df_pp["abs_dv"], N_BINS, labels=False, duplicates="drop")
    bm = df_pp.groupby("dv_bin")["fix_change_next_200"].mean()
    bc = df_pp.groupby("dv_bin")["abs_dv"].mean()
    ax.plot(bc.values, bm.values, marker="o", linestyle="-")
    ax.set_title(pp_label(pp))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=9, pad=2)
for j in range(n_pp, len(axes)):
    axes[j].axis("off")
fig.supxlabel("Model evidence (| decision variable |)", fontsize=54, y=0.01)
fig.supylabel("Saccade probability", fontsize=54, x=0.01)
fig.subplots_adjust(left=0.1, bottom=0.1, top=0.92, wspace=0.25, hspace=0.35)
fig.savefig(f"{OUTDIR}/individual_absdv_saccade_mosaic.png", dpi=300)
fig.savefig(f"{OUTDIR}/individual_absdv_saccade_mosaic.pdf", format="pdf")
plt.close()
print(" - individual_absdv_saccade_mosaic.png")

# =============================================================
# GAZE-POLICY (NULL MODEL) COMPARISON
# =============================================================
apply_r_style()

GAZE_MODES = ["real", "random_direction", "shuffle_time", "random_from_real_hist", "random",
              "ideal_coverage", "center", "corner_tl", ]

GAZE_MODE_LABEL = {
    "real":                  "Recorded gaze",
    "random_direction":      "Random direction",
    "shuffle_time":          "Shuffled fixations",
    "random_from_real_hist": "Resampled history",
    "random":                "Fully random",
    "ideal_coverage":        "Ideal coverage",
    "center":                "Constant central",
    "corner_tl":             "Constant peripheral",
}

# One distinct color per mode
MODE_COLOR = {
    "real":                   "#F0190A",
    "random_direction":       "#FCF29B",
    "shuffle_time":           "#B39DDB",
    "random_from_real_hist":  "#93C5EC",
    "random":                 "#58C8D7",
    "ideal_coverage":         "#7FC482",
    "center":                 "#7984C6",
    "corner_tl":              "#AFBBC1",
}

def _rmse(human, model):
    human = np.asarray(human, dtype=float)
    model = np.asarray(model, dtype=float)
    return np.sqrt(np.nanmean((human - model) ** 2))

RAW_FILE_CANDIDATES = {
    "real":                   ["replay_model_results_kfold_zscore_real.csv", "replay_model_results_kfold_zscore_v2.csv"],
    "random_direction":       ["replay_model_results_kfold_zscore_matched_eccentricity_random_direction.csv"],
    "shuffle_time":           ["replay_model_results_kfold_zscore_shuffle_time.csv"],
    "random_from_real_hist":  ["replay_model_results_kfold_zscore_random_from_real_hist.csv"],
    "random":                 ["replay_model_results_kfold_zscore_random.csv"],
    "ideal_coverage":         ["replay_model_results_kfold_zscore_ideal_coverage.csv"],
    "center":                 ["replay_model_results_kfold_zscore_center.csv"],
    "corner_tl":              ["replay_model_results_kfold_zscore_corner_tl.csv"],
}

def load_gaze_mode_rmse_from_raw(raw_dir=".", modes=GAZE_MODES):
    rows = []
    for mode in modes:
        path = None
        for candidate in RAW_FILE_CANDIDATES[mode]:
            p = os.path.join(raw_dir, candidate)
            if os.path.exists(p):
                path = p
                break
        if path is None:
            print(f"[WARN] no raw file found for mode '{mode}' "
                  f"(tried {RAW_FILE_CANDIDATES[mode]})")
            continue

        df_mode = pd.read_csv(path)
        df_mode["human_acc"] = df_mode["human_correct"].astype(float)
        df_mode["speed_px_s_used"] = df_mode["speed_px_s_used"].astype(int)
        df_mode = df_mode[df_mode["speed_px_s_used"].isin(SPEED_ORDER)].copy()

        rt_present_col = pick_first_existing(
            df_mode, ["model_rt_present_mean_s", "model_rt_present_median_s"])
        rt_absent_col = pick_first_existing(
            df_mode, ["model_rt_absent_mean_s", "model_rt_absent_median_s"])

        # -- reuses the existing aggregation functions, same logic as the rest of the script --

        rt_tp_m = df_mode[df_mode["human_target_present"] == 1].copy()
        rt_ta_m = df_mode[df_mode["human_target_present"] == 0].copy()
        rt_pp_tp_m = rt_pp_for_subset(rt_tp_m, rt_present_col)
        rt_pp_ta_m = rt_pp_for_subset(rt_ta_m, rt_absent_col)
        rt_group_tp_m = rt_groupify(rt_pp_tp_m)
        rt_group_ta_m = rt_groupify(rt_pp_ta_m)
        
        hit_pp_m, fa_pp_m = response_rates_by_pp(df_mode)
        dprime_pp_m = make_dprime_pp(hit_pp_m, fa_pp_m)
        dprime_group_m = groupify(dprime_pp_m, "human_dprime", "model_dprime", "dprime")

        rows.append({
            "mode": mode,
            "label": GAZE_MODE_LABEL[mode],
            "rt_tp_rmse": _rmse(rt_group_tp_m["human_rt_mean"], rt_group_tp_m["model_rt_mean"]),
            "rt_ta_rmse": _rmse(rt_group_ta_m["human_rt_mean"], rt_group_ta_m["model_rt_mean"]),
            "dprime_rmse": _rmse(dprime_group_m["human_dprime_mean"], dprime_group_m["model_dprime_mean"]),
        })
    return pd.DataFrame(rows)

gaze_rmse = load_gaze_mode_rmse_from_raw(raw_dir=".")
if gaze_rmse.empty:
    print("[WARN] No gaze-mode files found; skipping gaze_mode_comparison.")
    raise SystemExit(0)
gaze_rmse.to_csv(f"{OUTDIR}/gaze_mode_rmse_table.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
metric_cols = [("rt_tp_rmse",  "RMSE", ""),
               ("rt_ta_rmse",  "RMSE", ""),
               ("dprime_rmse", "RMSE", ""),]

for ax, (col, ylabel, panel_label) in zip(axes, metric_cols):
    add_panel_label(ax, panel_label, fontsize=TITLE_FS * 1.1)
    bar_colors = [MODE_COLOR[m] for m in gaze_rmse["mode"]]
    ax.bar(range(len(gaze_rmse)), gaze_rmse[col], color=bar_colors,
           edgecolor="#333333", linewidth=0.8)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_xticks([])  # labels moved to shared legend instead of x-axis
    ax.tick_params(axis="y", labelsize=TICK_FS)

handles = [mpl.patches.Patch(facecolor=MODE_COLOR[m], edgecolor="#333333",
                              label=GAZE_MODE_LABEL[m]) for m in gaze_rmse["mode"]]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=LEGEND_FS, bbox_to_anchor=(0.5, -0.12))

fig.tight_layout()
fig.savefig(f"{OUTDIR}/gaze_mode_comparison.png", bbox_inches="tight")
fig.savefig(f"{OUTDIR}/gaze_mode_comparison.pdf", format="pdf", bbox_inches="tight")
plt.close()
print(" - gaze_mode_comparison.png")
