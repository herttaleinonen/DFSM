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
# R-style theme 
# ------------------------------------------------------------------
class R:
    BASE_SIZE    = 18
    FONT_FAMILY  = "sans-serif"
    LW_MEAN      = 1.4
    LW_SPAG      = 0.5
    LW_ERR       = 0.8
    PT_SIZE      = 6
    MARKER       = "o"
    CAPSIZE      = 4
    ALPHA_SPAG   = 0.10
    GRID_COLOR   = "#e5e5e5"
    GRID_LW      = 0.8

def apply_r_style():
    mpl.rcParams.update({
        "font.family":         R.FONT_FAMILY,
        "font.size":           R.BASE_SIZE,
        "axes.titlesize":      R.BASE_SIZE,
        "axes.labelsize":      R.BASE_SIZE,
        "xtick.labelsize":     R.BASE_SIZE * 0.85,
        "ytick.labelsize":     R.BASE_SIZE * 0.85,
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
        "xtick.major.size":    4,
        "ytick.major.size":    4,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "legend.frameon":      False,
        "legend.fontsize":     R.BASE_SIZE * 0.85,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
    })

def mosaic_style():
    """Style suitable for mosaic figures."""
    mpl.rcParams.update({
        "font.size":           10,
        "axes.titlesize":      10,
        "axes.labelsize":      10,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.fontsize":     9,
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
CSV_PATH = "replay_model_results_test.csv"
OUTDIR = "figures"

USE_CORRECT_ONLY_FOR_HUMAN_RT = False
MODEL_RT_FALLBACK_COL = "model_rt_mean_s"

COLOR_TP    = "#1f77b4"
COLOR_TA    = "#d62728"
COLOR_HUMAN = "#17becf"
COLOR_MODEL = "#2ca02c"

# -----------------------
# Speed display mapping (px/s → deg/s)
# -----------------------
SPEED_ORDER = [0, 100, 200, 300, 400]

SPEED_DEG_EXACT = {
    0:   0.000,
    100: 2.703,
    200: 5.406,
    300: 8.109,
    400: 10.812,
}

SPEED_DEG_LABELS = {
    0:   "0",
    100: "3",
    200: "5.5",
    300: "8",
    400: "11",
}

SPEED_LABEL_TEXT = "Object velocity (deg/s)"

def speed_display(x_px):
    x_px = np.asarray(x_px, dtype=int)
    return np.array([SPEED_DEG_EXACT[v] for v in x_px])

def speed_ticks_and_labels():
    ticks  = [SPEED_DEG_EXACT[v] for v in SPEED_ORDER]
    labels = [SPEED_DEG_LABELS[v] for v in SPEED_ORDER]
    return ticks, labels

# -----------------------
# Helpers
# -----------------------
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
    return (
        df_in.groupby(["participant", "speed_px_s_used"], as_index=False)
             .agg(human_acc=("human_acc","mean"), model_acc=("model_acc","mean"))
    )

def rates_to_dprime(hit, fa, eps=1e-4):
    H = np.clip(hit, eps, 1 - eps)
    F = np.clip(fa,  eps, 1 - eps)
    return norm.ppf(H) - norm.ppf(F)

def pp_label(pp):
    return "pp" + str(int(pp.replace("kh", "")))

def add_panel_label(ax, label, fontsize=24, fontweight="bold"):
    """Add an article-style panel label (a., b., ...) to the upper-left corner."""
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
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

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(CSV_PATH)
df["human_acc"] = df["human_correct"].astype(float)
df["model_acc"] = (
    (df["model_p_present"] >= 0.5).astype(int) == df["human_target_present"]
).astype(float)

needed = ["participant","speed_px_s_used","human_target_present",
          "human_response","human_correct","human_rt_s","model_p_present"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df["speed_px_s_used"] = df["speed_px_s_used"].astype(int)
df = df[df["speed_px_s_used"].isin(SPEED_ORDER)].copy()
df["model_p_absent"] = 1.0 - df["model_p_present"]

# -----------------------
# Model RT columns
# -----------------------
MODEL_RT_PRESENT_COL = pick_first_existing(df, ["model_rt_present_mean_s","model_rt_present_median_s"])
MODEL_RT_ABSENT_COL  = pick_first_existing(df, ["model_rt_absent_mean_s", "model_rt_absent_median_s"])
HAS_CONDITIONAL_MODEL_RT = (MODEL_RT_PRESENT_COL is not None) and (MODEL_RT_ABSENT_COL is not None)

if not HAS_CONDITIONAL_MODEL_RT:
    if MODEL_RT_FALLBACK_COL not in df.columns:
        raise ValueError(f"No conditional model RT columns and fallback '{MODEL_RT_FALLBACK_COL}' missing.")
    print("[WARN] Using unconditional RT:", MODEL_RT_FALLBACK_COL)

# -----------------------
# d' per participant
# -----------------------
def response_rates_by_pp(df_in):
    tp = df_in[df_in["human_target_present"] == 1]
    ta = df_in[df_in["human_target_present"] == 0]
    hit_pp = (tp.groupby(["participant","speed_px_s_used"], as_index=False)
                .agg(human_hit=("human_response","mean"), model_hit=("model_p_present","mean")))
    fa_pp  = (ta.groupby(["participant","speed_px_s_used"], as_index=False)
                .agg(human_fa=("human_response","mean"),  model_fa=("model_p_present","mean")))
    return hit_pp, fa_pp

hit_pp, fa_pp = response_rates_by_pp(df)
acc_pp = accuracy_by_pp(df)

dprime_pp = hit_pp.merge(fa_pp, on=["participant","speed_px_s_used"], how="inner")
dprime_pp["human_dprime"] = rates_to_dprime(dprime_pp["human_hit"], dprime_pp["human_fa"])
dprime_pp["model_dprime"] = rates_to_dprime(dprime_pp["model_hit"], dprime_pp["model_fa"])

# -----------------------
# RT per participant per speed (TP vs TA)
# -----------------------
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
    rt_tp_tag = f"TP_correct_model={MODEL_RT_PRESENT_COL}"
    rt_ta_tag = f"TA_correct_model={MODEL_RT_ABSENT_COL}"
else:
    rt_pp_tp = rt_pp_for_subset(rt_tp, MODEL_RT_FALLBACK_COL)
    rt_pp_ta = rt_pp_for_subset(rt_ta, MODEL_RT_FALLBACK_COL)
    rt_tp_tag = f"TP_correct_model={MODEL_RT_FALLBACK_COL}"
    rt_ta_tag = f"TA_correct_model={MODEL_RT_FALLBACK_COL}"

def frac_imputed(df0, col):
    x = pd.to_numeric(df0[col], errors="coerce")
    return float(np.mean(np.isnan(x)))

print("TP model RT missing fraction:", frac_imputed(rt_tp, MODEL_RT_PRESENT_COL or MODEL_RT_FALLBACK_COL))
print("TA model RT missing fraction:", frac_imputed(rt_ta, MODEL_RT_ABSENT_COL  or MODEL_RT_FALLBACK_COL))

# -----------------------
# Group-level mean ± SEM
# -----------------------
def groupify(pp_df, human_col, model_col, prefix):
    g = (pp_df.groupby("speed_px_s_used", as_index=False)
              .agg(
                  human_mean=(human_col,"mean"), human_sem=(human_col,sem),
                  model_mean=(model_col,"mean"), model_sem=(model_col,sem),
                  n_pp=("participant","nunique"),
              ))
    g = g.rename(columns={
        "human_mean": f"human_{prefix}_mean", "human_sem": f"human_{prefix}_sem",
        "model_mean": f"model_{prefix}_mean", "model_sem": f"model_{prefix}_sem",
    })
    g["speed_px_s_used"] = pd.Categorical(g["speed_px_s_used"], SPEED_ORDER, ordered=True)
    return g.sort_values("speed_px_s_used")

def rt_groupify(rt_pp):
    g = (rt_pp.groupby("speed_px_s_used", as_index=False)
              .agg(
                  human_rt_mean=("human_rt","mean"), human_rt_sem=("human_rt",sem),
                  model_rt_mean=("model_rt","mean"), model_rt_sem=("model_rt",sem),
                  n_pp=("participant","nunique"),
              ))
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
# SINGLE-PANEL PLOTS (R style)
# =============================================================
apply_r_style()

# ------------- Plot 1: RT vs speed (TP and TA, separate figures) -------------
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
    ax.set_ylabel("Reaction time (s)")
    ax.set_ylim(1.0, 3.3)
    ax.legend()
    fig.tight_layout()

    safe_tag = tag.replace("/","_").replace(" ","_").replace("=","_")
    fig.savefig(f"{OUTDIR}/rt_vs_speed_{safe_tag}.png")
    plt.close()

# ------------- Plot 2: d' vs speed -------------
x = speed_display(dprime_group["speed_px_s_used"].values)

fig, ax = plt.subplots(figsize=(7, 5))
add_panel_label(ax, "c.")

plot_mean_with_err(ax, x, dprime_group["human_dprime_mean"], dprime_group["human_dprime_sem"],
                   color=COLOR_HUMAN, label="Human d'")
plot_mean_with_err(ax, x, dprime_group["model_dprime_mean"], dprime_group["model_dprime_sem"],
                   color=COLOR_MODEL, label="Model d'", linestyle="--")

set_speed_axis(ax)
ax.set_ylabel("Sensitivity (d')")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTDIR}/dprime_vs_speed.png")
plt.close()
print(" - dprime_vs_speed.png")

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

# ------------- Plot 3: Individual RT fits mosaic -------------
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows), sharex=True)
axes = np.atleast_1d(axes).ravel()

fig.text(0.01, 0.98, "a.", fontsize=54, fontweight="bold", va="top", ha="left")


x = speed_display(SPEED_ORDER)

for i, pp in enumerate(participants):
    ax = axes[i]
    ax.plot(x, rt_tp_h_w.loc[pp].values, marker="o", linestyle="-",
            color=COLOR_TP, label="TP Human")
    ax.plot(x, rt_tp_m_w.loc[pp].values, marker="^", linestyle="--",
            color=COLOR_TP, label="TP Model")
    ax.plot(x, rt_ta_h_w.loc[pp].values, marker="o", linestyle="-",
            color=COLOR_TA, label="TA Human")
    ax.plot(x, rt_ta_m_w.loc[pp].values, marker="^", linestyle="--",
            color=COLOR_TA, label="TA Model")
    ax.set_title(pp_label(pp))
    ticks, labels = speed_ticks_and_labels()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", labelsize=9, pad=2)
    ax.set_ylim(-0.02, 4.52)
    ax.grid(True, alpha=0.2)

for j in range(n_pp, len(axes)):
    axes[j].axis("off")

seen = {}
for ax in axes[:n_pp]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
            seen[l] = h

fig.supxlabel(SPEED_LABEL_TEXT, fontsize=30, y=0.03)
fig.supylabel("Reaction time (s)", fontsize=30, x=0.03)
fig.subplots_adjust(left=0.08, bottom=0.08, top=0.92, wspace=0.25, hspace=0.25)
fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=4, fontsize=20, frameon=False)
plt.savefig(f"{OUTDIR}/individual_fits_mosaic.png", dpi=300)
plt.close()

# ----------- Plot 4: Individual d' mosaic -------------
dprime_h_w = pivot_wide(dprime_pp, "human_dprime")
dprime_m_w = pivot_wide(dprime_pp, "model_dprime")

fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows), sharex=True)
axes = np.atleast_1d(axes).ravel()

fig.text(0.01, 0.98, "b.", fontsize=54, fontweight="bold", va="top", ha="left")

x = speed_display(SPEED_ORDER)

for i, pp in enumerate(participants):
    ax = axes[i]
    ax.plot(x, dprime_h_w.loc[pp].values, marker="o", linestyle="-",
            color=COLOR_HUMAN, label="Human d'")
    ax.plot(x, dprime_m_w.loc[pp].values, marker="^", linestyle="--",
            color=COLOR_MODEL, label="Model d'")
    ax.set_title(pp_label(pp))
    ticks, labels = speed_ticks_and_labels()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", labelsize=9, pad=2)
    ax.grid(True, alpha=0.2)

for j in range(n_pp, len(axes)):
    axes[j].axis("off")

seen = {}
for ax in axes[:n_pp]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
            seen[l] = h

fig.supxlabel(SPEED_LABEL_TEXT, fontsize=30, y=0.03)
fig.supylabel("Sensitivity (d′)", fontsize=30, x=0.03)

fig.subplots_adjust(left=0.08, bottom=0.08, top=0.85, wspace=0.25, hspace=0.25)
fig.legend(seen.values(), seen.keys(), 
           loc="upper left",
           ncol=2,
           fontsize=28,
           frameon=False,
           bbox_to_anchor=(0.08, 0.98))

plt.savefig(f"{OUTDIR}/individual_dprime_mosaic.png", dpi=300)
plt.close()
print(" - individual_dprime_mosaic.png")

# ---- Plot 5: Fitted parameters (eta, theta) -------------
apply_r_style()

FIT_CSV = "fitted_params_test.csv"
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
        add_panel_label(ax1, "f.")

        ax1.hist(fit["eta"].values, bins=10)
        ax1.set_xlabel("η (temporal efficiency)")
        ax1.set_ylabel("Count")
        ax1.set_title("Fitted η")

        ax2.hist(fit["theta"].values, bins=10)
        ax2.set_xlabel("Θ (present bound)")
        ax2.set_ylabel("Count")
        ax2.set_title("Fitted Θ")

        ax3.scatter(fit["eta"].values, fit["theta"].values, s=R.PT_SIZE*8)
        ax3.set_xlabel("η")
        ax3.set_ylabel("Θ")
        ax3.set_title("η vs Θ")
        for _, r in fit.iterrows():
            ax3.annotate(str(r["participant"]),
                         (float(r["eta"]), float(r["theta"])),
                         textcoords="offset points", xytext=(3,3), fontsize=9)

        fig.tight_layout()
        fig.savefig(f"{OUTDIR}/fitted_parameters.png")
        plt.close()
        print(" - fitted_parameters.png")
else:
    print(f"[WARN] {FIT_CSV} not found. Skipping.")

# -----------------------
# Save summary tables
# -----------------------
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
apply_r_style()

df = pd.read_csv("saccade_prediction_table_test.csv")
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
df_tp["z_target_loglr"] = (
    (df_tp["target_loglr"] - df_tp["target_loglr"].mean()) / df_tp["target_loglr"].std(ddof=0)
)
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

# Saccade Plot 1: group binned
N_BINS = 100
df_plot = df.copy()
df_plot["dv_bin"] = pd.qcut(df_plot["abs_dv"], N_BINS, labels=False, duplicates="drop")
bin_means   = df_plot.groupby("dv_bin")["fix_change_next_200"].mean()
bin_centers = df_plot.groupby("dv_bin")["abs_dv"].mean()

fig, ax = plt.subplots(figsize=(7, 5.5))
add_panel_label(ax, "a.")
ax.plot(bin_centers, bin_means, marker="o", linewidth=R.LW_MEAN, markersize=R.PT_SIZE)
ax.set_xlabel("Model evidence (|decision variable|)")
ax.set_ylabel("Saccade probability")
fig.tight_layout()
fig.savefig("plot_absdv_binned.png", dpi=150)
plt.close()

# Saccade Plot 2: individual mosaic
mosaic_style()

N_BINS = 30
participants = sorted(
    df["participant"].dropna().unique(),
    key=lambda s: int(str(s).replace("kh",""))
)
n_pp  = len(participants)
ncols = 4
nrows = math.ceil(n_pp / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows),
                         sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
fig.text(0.01, 0.98, "b.", fontsize=64, fontweight="bold", va="top", ha="left")

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
    ax.tick_params(axis="both", labelsize=9, pad=2)

for j in range(n_pp, len(axes)):
    axes[j].axis("off")

#fig.supxlabel("Model evidence (|decision variable|)", fontsize=30, y=0.03)
#fig.supylabel("Saccade probability", fontsize=30, x=0.03)
fig.subplots_adjust(left=0.08, bottom=0.08, top=0.92, wspace=0.25, hspace=0.25)
fig.savefig(f"{OUTDIR}/individual_absdv_saccade_mosaic.png", dpi=300)
plt.close()
print(" - individual_absdv_saccade_mosaic.png")
