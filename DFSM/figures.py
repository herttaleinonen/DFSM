#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:25:35 2026

@author: herttaleinonen
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math

# -----------------------
# Settings
# -----------------------
CSV_PATH = "replay_model_results_test.csv"   # change if needed
OUTDIR = "figures"

# RT choices:
# - human: only correct trials
USE_CORRECT_ONLY_FOR_HUMAN_RT = False

# If conditional model RT columns exist, prefer them:
#  - present RT for TP (hits)
#  - absent RT for TA (correct rejections)
MODEL_RT_FALLBACK_COL = "model_rt_mean_s"  # used if conditional RT columns are missing

# Colors per metric
COLOR_TP = "#1f77b4"   # blue
COLOR_TA = "#d62728"   # red
COLOR_HUMAN = "#17becf"   # turquoise 
COLOR_MODEL = "#2ca02c"   # green

# -----------------------
# Speed display mapping (px/s → deg/s)
# -----------------------
SPEED_ORDER = [0, 100, 200, 300, 400]

SPEED_LABEL_TEXT = "Object velocity (deg/s)"

SPEED_LABELS = {
    0: 0,
    100: 3,
    200: 6,
    300: 8,
    400: 11,
}

def speed_display(x_px):
    """Convert array-like pixel speeds to degree speeds."""
    x_px = np.asarray(x_px, dtype=int)
    return np.array([SPEED_LABELS[v] for v in x_px])

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
    """Return the first column in candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def accuracy_by_pp(df_in):
    return (
        df_in.groupby(["participant", "speed_px_s_used"], as_index=False)
             .agg(
                 human_acc=("human_acc", "mean"),
                 model_acc=("model_acc", "mean"),
             )
    )


def rates_to_dprime(hit, fa, eps=1e-4):
    """Convert hit / false alarm rates to d' with clipping."""
    H = np.clip(hit, eps, 1 - eps)
    F = np.clip(fa, eps, 1 - eps)
    return norm.ppf(H) - norm.ppf(F)


def pp_label(pp):
    """Convert kh12 → pp12"""
    return "pp" + str(int(pp.replace("kh", "")))

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(CSV_PATH)
# --- create accuracy variables ---
df["human_acc"] = df["human_correct"].astype(float)

df["model_acc"] = (
    (df["model_p_present"] >= 0.5).astype(int)
    == df["human_target_present"]
).astype(float)

# Basic sanity checks (human_response used for hit/FA plots)
needed = [
    "participant", "speed_px_s_used",
    "human_target_present", "human_response", "human_correct", "human_rt_s",
    "model_p_present"
]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Force speed type and restrict to expected speeds
df["speed_px_s_used"] = df["speed_px_s_used"].astype(int)
df = df[df["speed_px_s_used"].isin(SPEED_ORDER)].copy()

# Convenience: model probability of "absent"
df["model_p_absent"] = 1.0 - df["model_p_present"]

# -----------------------
# Pick model RT columns (conditional if available)
# -----------------------
MODEL_RT_PRESENT_COL = pick_first_existing(df, [
    "model_rt_present_mean_s", "model_rt_present_median_s"
])
MODEL_RT_ABSENT_COL = pick_first_existing(df, [
    "model_rt_absent_mean_s", "model_rt_absent_median_s"
])

HAS_CONDITIONAL_MODEL_RT = (MODEL_RT_PRESENT_COL is not None) and (MODEL_RT_ABSENT_COL is not None)

if not HAS_CONDITIONAL_MODEL_RT:
    # Backward compatible fallback
    if MODEL_RT_FALLBACK_COL not in df.columns:
        raise ValueError(
            "No conditional model RT columns found and fallback column "
            f"'{MODEL_RT_FALLBACK_COL}' is missing."
        )
    print("[WARN] Conditional model RT columns not found.")
    print("       Using unconditional RT:", MODEL_RT_FALLBACK_COL)
    print("       (To get proper TP-vs-TA model RTs, write model_rt_present_* and model_rt_absent_* to CSV.)")


# -----------------------
# d' per participant
# -----------------------
def response_rates_by_pp(df_in):
    tp = df_in[df_in["human_target_present"] == 1]
    ta = df_in[df_in["human_target_present"] == 0]

    hit_pp = (
        tp.groupby(["participant", "speed_px_s_used"], as_index=False)
          .agg(
              human_hit=("human_response", "mean"),
              model_hit=("model_p_present", "mean"),
          )
    )
    fa_pp = (
        ta.groupby(["participant", "speed_px_s_used"], as_index=False)
          .agg(
              human_fa=("human_response", "mean"),
              model_fa=("model_p_present", "mean"),
          )
    )
    return hit_pp, fa_pp

hit_pp, fa_pp = response_rates_by_pp(df)
acc_pp = accuracy_by_pp(df)

dprime_pp = hit_pp.merge(
    fa_pp,
    on=["participant", "speed_px_s_used"],
    how="inner"
)

dprime_pp["human_dprime"] = rates_to_dprime(
    dprime_pp["human_hit"], dprime_pp["human_fa"]
)

dprime_pp["model_dprime"] = rates_to_dprime(
    dprime_pp["model_hit"], dprime_pp["model_fa"]
)

# -----------------------
# RT per participant per speed (TP vs TA)
# -----------------------
def rt_pp_for_subset(
    df_in,
    model_rt_col,
    *,
    condition_on="human_correct",   # "none" | "human_correct"
    deadline_s=3.5,
    nondecision_s=0.25,
):

    tmp = df_in.copy()

    if condition_on == "human_correct":
        tmp = tmp[tmp["human_correct"] == 1].copy()
    elif condition_on != "none":
        raise ValueError("condition_on must be 'none' or 'human_correct'")

    # Impute missing model RTs so failures don't disappear
    tmp[model_rt_col] = pd.to_numeric(tmp[model_rt_col], errors="coerce")
    tmp[model_rt_col] = tmp[model_rt_col].fillna(deadline_s + nondecision_s)

    return (
        tmp.groupby(["participant", "speed_px_s_used"], as_index=False)
           .agg(
               human_rt=("human_rt_s", "mean"),
               model_rt=(model_rt_col, "mean"),
           )
    )

# TP correct RT
rt_tp = df[df["human_target_present"] == 1].copy()
if HAS_CONDITIONAL_MODEL_RT:
    rt_pp_tp = rt_pp_for_subset(rt_tp, MODEL_RT_PRESENT_COL)
    rt_tp_tag = f"TP_correct_model={MODEL_RT_PRESENT_COL}"
else:
    rt_pp_tp = rt_pp_for_subset(rt_tp, MODEL_RT_FALLBACK_COL)
    rt_tp_tag = f"TP_correct_model={MODEL_RT_FALLBACK_COL}"

# TA correct RT (“correct rejections”)
rt_ta = df[df["human_target_present"] == 0].copy()
if HAS_CONDITIONAL_MODEL_RT:
    rt_pp_ta = rt_pp_for_subset(rt_ta, MODEL_RT_ABSENT_COL)
    rt_ta_tag = f"TA_correct_model={MODEL_RT_ABSENT_COL}"
else:
    rt_pp_ta = rt_pp_for_subset(rt_ta, MODEL_RT_FALLBACK_COL)
    rt_ta_tag = f"TA_correct_model={MODEL_RT_FALLBACK_COL}"
    

def frac_imputed(df0, col, deadline=3.5+0.25):
    x = pd.to_numeric(df0[col], errors="coerce")
    return float(np.mean(np.isnan(x)))  # before fill

print("TP model RT missing fraction:", frac_imputed(rt_tp, MODEL_RT_PRESENT_COL or MODEL_RT_FALLBACK_COL))
print("TA model RT missing fraction:", frac_imputed(rt_ta, MODEL_RT_ABSENT_COL or MODEL_RT_FALLBACK_COL))

# -----------------------
# Group-level mean ± SEM across participants
# -----------------------
def groupify(pp_df, human_col, model_col, prefix):
    g = (
        pp_df.groupby("speed_px_s_used", as_index=False)
             .agg(
                 human_mean=(human_col, "mean"),
                 human_sem=(human_col, sem),
                 model_mean=(model_col, "mean"),
                 model_sem=(model_col, sem),
                 n_pp=("participant", "nunique"),
             )
    )
    g = g.rename(columns={
        "human_mean": f"human_{prefix}_mean",
        "human_sem": f"human_{prefix}_sem",
        "model_mean": f"model_{prefix}_mean",
        "model_sem": f"model_{prefix}_sem",
    })
    g["speed_px_s_used"] = pd.Categorical(g["speed_px_s_used"], SPEED_ORDER, ordered=True)
    return g.sort_values("speed_px_s_used")

hit_group = groupify(hit_pp, "human_hit", "model_hit", "hit")
fa_group  = groupify(fa_pp,  "human_fa",  "model_fa",  "fa")
acc_group = groupify(acc_pp, "human_acc", "model_acc", "acc")
dprime_group = groupify(dprime_pp, "human_dprime", "model_dprime", "dprime")

def rt_groupify(rt_pp):
    g = (
        rt_pp.groupby("speed_px_s_used", as_index=False)
             .agg(
                 human_rt_mean=("human_rt", "mean"),
                 human_rt_sem=("human_rt", sem),
                 model_rt_mean=("model_rt", "mean"),
                 model_rt_sem=("model_rt", sem),
                 n_pp=("participant", "nunique"),
             )
    )
    g["speed_px_s_used"] = pd.Categorical(g["speed_px_s_used"], SPEED_ORDER, ordered=True)
    return g.sort_values("speed_px_s_used")

rt_group_tp = rt_groupify(rt_pp_tp)
rt_group_ta = rt_groupify(rt_pp_ta)

# -----------------------
# Plotting
# -----------------------

ensure_outdir(OUTDIR)

# ------------- Plot 1: RT vs speed (TP correct and TA correct) -------------
for rt_group, tag in [(rt_group_tp, rt_tp_tag), (rt_group_ta, rt_ta_tag)]:

    color = COLOR_TP if rt_group is rt_group_tp else COLOR_TA

    x = speed_display(rt_group["speed_px_s_used"].values)

    plt.figure()

    # Human
    plt.plot(
        x,
        rt_group["human_rt_mean"],
        marker="o",
        linestyle="-",
        color=color,
        label="Human"
    )

    plt.fill_between(
        x,
        rt_group["human_rt_mean"] - rt_group["human_rt_sem"],
        rt_group["human_rt_mean"] + rt_group["human_rt_sem"],
        color=color,
        alpha=0.25
    )

    # Model
    plt.plot(
        x,
        rt_group["model_rt_mean"],
        marker="^",          
        linestyle="--",      
        color=color,
        label="Model"
    )

    plt.fill_between(
        x,
        rt_group["model_rt_mean"] - rt_group["model_rt_sem"],
        rt_group["model_rt_mean"] + rt_group["model_rt_sem"],
        color=color,
        alpha=0.12
    )

    plt.xlabel(SPEED_LABEL_TEXT, fontsize=16)
    plt.ylabel("Reaction time (s)", fontsize=16)
    plt.ylim(1.0, 3.3)
    plt.xticks(list(SPEED_LABELS.values()), fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.gcf().subplots_adjust(bottom=0.18)  

    safe_tag = tag.replace("/", "_").replace(" ", "_").replace("=", "_")
    plt.savefig(f"{OUTDIR}/rt_vs_speed_{safe_tag}.png", dpi=300)
    plt.close()

# ------------- Plot 2: d' vs speed -------------
x = speed_display(dprime_group["speed_px_s_used"].values)

plt.figure()

plt.plot(
    x,
    dprime_group["human_dprime_mean"],
    marker="o",
    linestyle="-",
    color=COLOR_HUMAN,
    label="Human d'"
)

plt.fill_between(
    x,
    dprime_group["human_dprime_mean"] - dprime_group["human_dprime_sem"],
    dprime_group["human_dprime_mean"] + dprime_group["human_dprime_sem"],
    color=COLOR_HUMAN,
    alpha=0.25
)

plt.plot(
    x,
    dprime_group["model_dprime_mean"],
    marker="o",
    linestyle="--",
    color=COLOR_MODEL,
    label="Model d'"
)

plt.fill_between(
    x,
    dprime_group["model_dprime_mean"] - dprime_group["model_dprime_sem"],
    dprime_group["model_dprime_mean"] + dprime_group["model_dprime_sem"],
    color=COLOR_MODEL,
    alpha=0.18
)

plt.xlabel(SPEED_LABEL_TEXT, fontsize=16)
plt.ylabel("Sensitivity (d')", fontsize=16)
plt.xticks(list(SPEED_LABELS.values()), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

plt.gcf().subplots_adjust(bottom=0.18)  
plt.savefig(f"{OUTDIR}/dprime_vs_speed.png", dpi=300)
plt.close()

print(" - dprime_vs_speed.png")


# ------------- Plot 3: Individual RT fits mosaic (one panel per participant) -------------
def pivot_wide(pp_df, value_col):
    """participant × speed wide table for easy plotting; returns DataFrame indexed by participant."""
    wide = pp_df.pivot(index="participant", columns="speed_px_s_used", values=value_col)
    # ensure all speeds exist as columns (may be missing for some participants)
    for s in SPEED_ORDER:
        if s not in wide.columns:
            wide[s] = np.nan
    return wide[SPEED_ORDER].sort_index()

participants = sorted(
    df["participant"].unique(),
    key=lambda s: int(s.replace("kh", ""))
)
n_pp = len(participants)

# Wide tables
rt_tp_h_w = pivot_wide(rt_pp_tp, "human_rt")
rt_tp_m_w = pivot_wide(rt_pp_tp, "model_rt")

rt_ta_h_w = pivot_wide(rt_pp_ta, "human_rt")
rt_ta_m_w = pivot_wide(rt_pp_ta, "model_rt")

# Choose grid layout automatically (~4 columns)
ncols = 4
nrows = int(np.ceil(n_pp / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), sharex=True)
axes = np.atleast_1d(axes).ravel()

x = speed_display(SPEED_ORDER)

for i, pp in enumerate(participants):
    ax = axes[i]
    
    # --- TP
    ax.plot(
        x,
        rt_tp_h_w.loc[pp].values,
        marker="o",              
        linestyle="-",
        color=COLOR_TP,
        label="TP Human",     
    )
    
    ax.plot(
        x,
        rt_tp_m_w.loc[pp].values,
        marker="^",              
        linestyle="--",
        color=COLOR_TP,
        label="TP Model",  
    )
    
    # --- TA
    ax.plot(
        x,
        rt_ta_h_w.loc[pp].values,
        marker="o",
        linestyle="-",
        color=COLOR_TA,
        label="TA Human", 
    )
    
    ax.plot(
        x,
        rt_ta_m_w.loc[pp].values,
        marker="^",
        linestyle="--",
        color=COLOR_TA,
        label="TA Model",
    )

    ax.set_title(pp_label(pp))
    ax.set_xticks(list(SPEED_LABELS.values()))
    ax.tick_params(axis='both', labelsize=9, pad=2)
    ax.set_ylim(-0.02, 4.52)  
    ax.grid(True, alpha=0.2)
  
# Turn off unused axes
for j in range(n_pp, len(axes)):
    axes[j].axis("off")

# Collect legend entries
seen = {}
for ax in axes[:n_pp]:
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h

# Superlabels 
fig.supxlabel(SPEED_LABEL_TEXT, fontsize=30, y=0.03)
fig.supylabel("Reaction time (s)", fontsize=30, x=0.03)

# Tighter layout for the entire figure
fig.subplots_adjust(
    left=0.08,
    bottom=0.08,
    top=0.92,
    wspace=0.25,
    hspace=0.25
)

# Legend centered above plots
fig.legend(
    seen.values(), seen.keys(),
    loc="upper center",
    ncol=4,
    fontsize=20,
    frameon=False
)

plt.savefig(f"{OUTDIR}/individual_fits_mosaic.png", dpi=300)
plt.close()


# ----------- Plot 4: Individual d' mosaic -------------
dprime_h_w = pivot_wide(dprime_pp, "human_dprime")
dprime_m_w = pivot_wide(dprime_pp, "model_dprime")

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4.2 * ncols, 3.6 * nrows),
    sharex=True
)

axes = np.atleast_1d(axes).ravel()
x = speed_display(SPEED_ORDER)

for i, pp in enumerate(participants):
    ax = axes[i]

    # --- d' Human 
    ax.plot(
        x,
        dprime_h_w.loc[pp].values,
        marker="o",
        linestyle="-",
        color=COLOR_HUMAN,        
        label="Human d'"
    )

    # --- d' Model 
    ax.plot(
        x,
        dprime_m_w.loc[pp].values,
        marker="^",
        linestyle="--",
        color=COLOR_MODEL,        
        label="Model d'"
    )

    ax.set_title(pp_label(pp))
    ax.set_xticks(list(SPEED_LABELS.values()))
    ax.tick_params(axis='both', labelsize=9, pad=2)
    ax.grid(True, alpha=0.2)

# Turn off unused axes
for j in range(n_pp, len(axes)):
    axes[j].axis("off")

# Collect unified legend entries 
seen = {}
for ax in axes[:n_pp]:
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h

# Labels 
fig.supxlabel(SPEED_LABEL_TEXT, fontsize=30, y=0.03)
fig.supylabel("Sensitivity (d′)", fontsize=30, x=0.03)

# Layout 
fig.subplots_adjust(
    left=0.08,
    bottom=0.08,
    top=0.92,
    wspace=0.25,
    hspace=0.25
)

# Legend centered above
fig.legend(
    seen.values(), seen.keys(),
    loc="upper center",
    ncol=4,
    fontsize=20,
    frameon=False
)

plt.savefig(f"{OUTDIR}/individual_dprime_mosaic.png", dpi=300)
plt.close()

print(" - individual_dprime_mosaic.png")


# ---- Plot 5: Fitted parameters (eta, theta) -------------
FIT_CSV = "fitted_params_test.csv"
if os.path.exists(FIT_CSV):
    fit = pd.read_csv(FIT_CSV)

    # Basic checks / cleanup
    needed_fit = ["participant", "eta", "theta"]
    missing_fit = [c for c in needed_fit if c not in fit.columns]
    if missing_fit:
        print(f"[WARN] {FIT_CSV} missing columns: {missing_fit}. Skipping fitted-parameter plot.")
    else:
        # Make sure numeric
        fit["eta"] = pd.to_numeric(fit["eta"], errors="coerce")
        fit["theta"] = pd.to_numeric(fit["theta"], errors="coerce")
        fit = fit.dropna(subset=["eta", "theta"]).copy()

        # Save a copy into OUTDIR 
        fit.to_csv(f"{OUTDIR}/fitted_params_copy.csv", index=False)

        plt.figure(figsize=(10, 3.5))

        # Panel 1: eta histogram
        ax1 = plt.subplot(1, 3, 1)
        ax1.hist(fit["eta"].values, bins=10)
        ax1.set_xlabel("η (temporal efficiency)")
        ax1.set_ylabel("Count")
        ax1.set_title("Fitted η")

        # Panel 2: theta histogram
        ax2 = plt.subplot(1, 3, 2)
        ax2.hist(fit["theta"].values, bins=10)
        ax2.set_xlabel("Θ (present bound)")
        ax2.set_ylabel("Count")
        ax2.set_title("Fitted Θ")

        # Panel 3: eta vs theta scatter + labels
        ax3 = plt.subplot(1, 3, 3)
        ax3.scatter(fit["eta"].values, fit["theta"].values)
        ax3.set_xlabel("η")
        ax3.set_ylabel("Θ")
        ax3.set_title("η vs Θ")

        # Label points 
        for _, r in fit.iterrows():
            ax3.annotate(
                str(r["participant"]),
                (float(r["eta"]), float(r["theta"])),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=7
            )

        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/fitted_parameters.png", dpi=300)
        plt.close()

        print(" - fitted_parameters.png")
else:
    print(f"[WARN] {FIT_CSV} not found. Skipping fitted-parameter plot.")

df_reg = df[df["speed_px_s_used"] > 0].copy()
tp = df_reg[df_reg["human_target_present"] == 1].copy()

# -----------------------
# Save summary tables
# -----------------------
hit_group.to_csv(f"{OUTDIR}/summary_hit_rate_vs_speed.csv", index=False)
fa_group.to_csv(f"{OUTDIR}/summary_false_alarm_rate_vs_speed.csv", index=False)
safe_tp = f"summary_rt_vs_speed_{rt_tp_tag}.csv".replace("/", "_").replace(" ", "_")
safe_ta = f"summary_rt_vs_speed_{rt_ta_tag}.csv".replace("/", "_").replace(" ", "_")
rt_group_tp.to_csv(os.path.join(OUTDIR, safe_tp), index=False)
rt_group_ta.to_csv(os.path.join(OUTDIR, safe_ta), index=False)
acc_group.to_csv(f"{OUTDIR}/summary_accuracy_vs_speed.csv", index=False)
dprime_group.to_csv(f"{OUTDIR}/summary_dprime_vs_speed.csv", index=False)

print("Wrote figures to:", OUTDIR)



# SACCADE TIMING ANALYSIS

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("saccade_prediction_table_test.csv")

print("Loaded rows:", len(df))


# -----------------------
# Basic cleaning
# -----------------------
df = df.copy()

# drop NaNs in key variables
df = df[np.isfinite(df["margin"])]
df = df[np.isfinite(df["abs_dv"])]
df = df[np.isfinite(df["fix_age_s"])]
df = df[np.isfinite(df["t_sec"])]

print("After cleaning:", len(df))


# -----------------------
# Restrict to pre-response
# -----------------------
USE_PRE_RESPONSE_ONLY = True

if USE_PRE_RESPONSE_ONLY:
    df = df[df["t_sec"] < df["human_rt_s"]].copy()
    print("After RT filter:", len(df))


# -----------------------
# Remove very early bins 
# -----------------------
df = df[df["t_sec"] > 0.1]

# -----------------------
# Sanity check: binning
# -----------------------
print("\n=== BINNING CHECK (margin) ===")

df["margin_q"] = pd.qcut(df["margin"], 5, labels=False, duplicates="drop")

summary = df.groupby("margin_q")["fix_change_next_200"].mean()
counts = df.groupby("margin_q")["fix_change_next_200"].count()

print("Fixation change probability by margin quintile:")
print(summary)
print("\nCounts per bin:")
print(counts)


# -----------------------
# Standardize predictors
# -----------------------
for col in ["margin", "abs_dv", "t_sec", "fix_age_s"]:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

model_margin = smf.glm(
    formula="fix_change_next_200 ~ z_abs_dv + z_t_sec + z_fix_age_s + C(speed_px_s) + C(target_present)",
    data=df,
    family=sm.families.Binomial()
).fit(cov_type="cluster", cov_kwds={"groups": df["participant"]})


# -----------------------
# MODEL: abs_dv predicts saccades
# -----------------------
print("\n=== LOGISTIC REGRESSION: abs_dv ===")

model_absdv = smf.glm(
    formula="fix_change_next_200 ~ z_abs_dv + z_t_sec + z_fix_age_s + C(speed_px_s) + C(target_present)",
    data=df,
    family=sm.families.Binomial()
).fit(cov_type="cluster", cov_kwds={"groups": df["participant"]})

print(model_absdv.summary())


# -----------------------
# Check target-present only
# -----------------------
print("\n=== TARGET-PRESENT ONLY (target_loglr) ===")

df_tp = df[df["target_present"] == 1].copy()

df_tp = df_tp[np.isfinite(df_tp["target_loglr"])]

df_tp["z_target_loglr"] = (
    df_tp["target_loglr"] - df_tp["target_loglr"].mean()
) / df_tp["target_loglr"].std(ddof=0)

model_target = smf.glm(
    formula="fix_change_next_200 ~ z_target_loglr + z_t_sec + z_fix_age_s + C(speed_px_s)",
    data=df_tp,
    family=sm.families.Binomial()
).fit(cov_type="cluster", cov_kwds={"groups": df_tp["participant"]})

print(model_target.summary())


# -----------------------
# Effect size summary
# -----------------------
def print_effect(name, model, var):
    if var not in model.params.index:
        print(f"{name}: parameter '{var}' not in model")
        return
    coef = model.params[var]
    pval = model.pvalues[var]
    print(f"{name}: beta={coef:.3f}, p={pval:.3e}")

print("\n=== KEY EFFECTS ===")
print_effect("abs_dv", model_absdv, "z_abs_dv")

if "model_target" in locals() and "z_target_loglr" in model_target.params.index:
    print_effect("target_loglr", model_target, "z_target_loglr")


# -----------------------
# Plot 1: Accumulated evidence vs saccade probability
# -----------------------
N_BINS = 100

df_plot = df.copy()
df_plot["dv_bin"] = pd.qcut(df_plot["abs_dv"], N_BINS, labels=False, duplicates="drop")

bin_means = df_plot.groupby("dv_bin")["fix_change_next_200"].mean()
bin_centers = df_plot.groupby("dv_bin")["abs_dv"].mean()

plt.figure()
plt.plot(bin_centers, bin_means, marker="o")

plt.xlabel("Model evidence (|decision variable|)", fontsize=14)
plt.ylabel("Saccade probability (next 200 ms)", fontsize=14)
plt.title("Saccade probability vs accumulated evidence", fontsize=16)

plt.grid(True)
plt.savefig("plot_absdv_binned.png", dpi=150, bbox_inches="tight")
plt.show()


# -----------------------
# Plot 2: Individual saccade-evidence mosaic
# -----------------------
N_BINS = 30
participants = sorted(
    df["participant"].dropna().unique(),
    key=lambda s: int(str(s).replace("kh", ""))
)
n_pp = len(participants)

ncols = 4
nrows = math.ceil(n_pp / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4.2 * ncols, 3.6 * nrows),
    sharex=True,
    sharey=True
)

axes = np.atleast_1d(axes).ravel()

for i, pp in enumerate(participants):
    ax = axes[i]

    df_pp = df[df["participant"] == pp].copy()

    if len(df_pp) < N_BINS:
        ax.axis("off")
        continue

    df_pp["dv_bin"] = pd.qcut(
        df_pp["abs_dv"],
        N_BINS,
        labels=False,
        duplicates="drop"
    )

    bin_means = df_pp.groupby("dv_bin")["fix_change_next_200"].mean()
    bin_centers = df_pp.groupby("dv_bin")["abs_dv"].mean()

    ax.plot(
        bin_centers.values,
        bin_means.values,
        marker="o",
        linestyle="-"
    )

    ax.set_title(pp_label(pp))
    ax.tick_params(axis="both", labelsize=9, pad=2)
    ax.grid(True, alpha=0.2)

# Turn off unused axes
for j in range(n_pp, len(axes)):
    axes[j].axis("off")

# Labels
fig.supxlabel("Model evidence (|decision variable|)", fontsize=30, y=0.03)
fig.supylabel("Saccade probability (next 200 ms)", fontsize=30, x=0.03)

# Layout
fig.subplots_adjust(
    left=0.08,
    bottom=0.08,
    top=0.92,
    wspace=0.25,
    hspace=0.25
)

plt.savefig(f"{OUTDIR}/individual_absdv_saccade_mosaic.png", dpi=300)
plt.close()

print(" - individual_absdv_saccade_mosaic.png")
