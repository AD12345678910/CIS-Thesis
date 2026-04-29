"""
sample_size.py
Tests how model performance varies with training set size, across a
logarithmic grid from n=20 to the full dataset. Compares Burr XII,
baseline ML, and hybrid ML at each sample size.

Usage: python sample_size.py  (run burr12_fit.py first)
"""

import os, glob, warnings
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Find features file ─────────────────────────────────────────────────────
candidates = glob.glob("*.csv") + glob.glob("**/*.csv", recursive=True)
feat_file  = next((f for f in candidates if "burr12_features" in f), None)
if feat_file is None:
    raise FileNotFoundError("Run burr12_fit.py first.")
print(f"Using: {feat_file}")

print("SAMPLE SIZE SENSITIVITY EXPERIMENT")

df     = pd.read_csv(feat_file)
T_full = df["tenure"].values.astype(float)
E_full = df["event"].values.astype(float)
n_full = len(T_full)

BASE_FEATURES = [
    "is_male","is_senior","partner","dependents","phone",
    "internet_fiber","internet_dsl","online_security","online_backup",
    "device_protection","tech_support","streaming_tv","streaming_movies",
    "multiple_lines","contract_1yr","contract_2yr",
    "pay_echeck","paperless","MonthlyCharges_z",
]
SURVIVAL_FEATURES = ["burr12_survival","burr12_hazard","burr12_churn_prob"]
HYBRID_FEATURES   = BASE_FEATURES + SURVIVAL_FEATURES

SAMPLE_SIZES = [20, 50, 100, 200, 500, 1000, 2000, 3500, 5000, n_full]
N_SEEDS      = 5
H            = 12

print(f"Dataset      : {n_full} customers ({E_full.mean():.1%} churn)")
print(f"Sample sizes : {SAMPLE_SIZES}")
print(f"Seeds/size   : {N_SEEDS}\n")


# ── Burr XII ───────────────────────────────────────────────────────────────

def burr12_sf(t, c, k, sigma):
    return (1.0 + (t / sigma) ** c) ** (-k)

def burr12_pdf(t, c, k, sigma):
    u = (t / sigma) ** c
    return (c * k / sigma) * (t / sigma)**(c-1) * (1.0 + u)**(-(k+1))

def burr12_hf(t, c, k, sigma):
    u = (t / sigma) ** c
    return (c * k / sigma) * (t / sigma)**(c-1) / (1.0 + u)

BOUNDS = [(0.05, 15.0), (0.001, 200.0), (0.5, 1000.0)]

def nll(params, T, E):
    c, k, sigma = params
    t   = np.clip(T, 1e-9, None)
    pdf = np.clip(burr12_pdf(t, c, k, sigma), 1e-300, None)
    sf  = np.clip(burr12_sf(t, c, k, sigma),  1e-300, None)
    val = -(E * np.log(pdf) + (1-E) * np.log(sf)).sum()
    return val if np.isfinite(val) else 1e12

def fit_burr12(T, E, n):
    # Faster DE settings for small samples
    popsize = 10 if n < 500 else 15
    maxiter = 500 if n < 500 else 1000
    res_de = differential_evolution(
        nll, BOUNDS, args=(T, E), seed=42,
        maxiter=maxiter, tol=1e-8,
        popsize=popsize, mutation=(0.5, 1.0),
        recombination=0.7, workers=1, polish=True,
    )
    res = minimize(nll, res_de.x, args=(T, E), method="L-BFGS-B",
                   bounds=BOUNDS,
                   options={"ftol":1e-10,"gtol":1e-8,"maxiter":2000})
    return res.x[0], res.x[1], res.x[2]

def churn_prob(T, c, k, sigma, h=H):
    sf_t  = np.clip(burr12_sf(T, c, k, sigma), 1e-9, None)
    sf_th = burr12_sf(T + h, c, k, sigma)
    return 1.0 - sf_th / sf_t


# ── ML ─────────────────────────────────────────────────────────────────────

def make_lr():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=500, C=1.0, random_state=42))])

def make_xgb(pos_weight):
    return xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=42, verbosity=0,
    )


# ── Main loop ──────────────────────────────────────────────────────────────

results = []

for n_sample in SAMPLE_SIZES:
    use_full = (n_sample >= n_full)
    cols = {k: [] for k in [
        "b12_auc","b12_pr","b12_c","b12_k","b12_sigma",
        "base_lr_auc","base_xgb_auc","hybrid_lr_auc","hybrid_xgb_auc",
        "base_lr_pr","base_xgb_pr","hybrid_lr_pr","hybrid_xgb_pr",
    ]}

    label = "full" if use_full else str(n_sample)
    print(f"  n={label:<6}", end="", flush=True)

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed * 100)

        # Stratified subsample
        if use_full:
            idx = np.arange(n_full)
        else:
            pos = np.where(E_full == 1)[0]
            neg = np.where(E_full == 0)[0]
            n_pos = max(5, int(n_sample * E_full.mean()))
            n_neg = n_sample - n_pos
            n_pos = min(n_pos, len(pos))
            n_neg = min(n_neg, len(neg))
            idx   = np.concatenate([
                rng.choice(pos, n_pos, replace=False),
                rng.choice(neg, n_neg, replace=False),
            ])

        sub    = df.iloc[idx].reset_index(drop=True)
        T_s    = sub["tenure"].values.astype(float)
        E_s    = sub["event"].values.astype(float)
        X_b    = sub[BASE_FEATURES].values
        X_h    = sub[HYBRID_FEATURES].values

        if E_s.sum() < 5 or (1-E_s).sum() < 5:
            print("!", end="", flush=True)
            continue

        # ── Burr XII: fit on subsample, predict on full ────────────────
        try:
            c, k, sigma = fit_burr12(T_s, E_s, len(T_s))
            pred_b12 = churn_prob(T_full, c, k, sigma)
            cols["b12_auc"].append(roc_auc_score(E_full, pred_b12))
            cols["b12_pr"].append(average_precision_score(E_full, pred_b12))
            cols["b12_c"].append(c)
            cols["b12_k"].append(k)
            cols["b12_sigma"].append(sigma)
        except Exception:
            pass

        # ── ML: train on sub, evaluate on held-out (or CV if full) ────
        pos_w = (1-E_s.mean()) / max(E_s.mean(), 1e-6)

        if use_full:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            try:
                p = cross_val_predict(make_lr(), X_b, E_s, cv=cv,
                                      method="predict_proba", n_jobs=1)[:,1]
                cols["base_lr_auc"].append(roc_auc_score(E_s, p))
                cols["base_lr_pr"].append(average_precision_score(E_s, p))

                p = cross_val_predict(make_xgb(pos_w), X_b, E_s, cv=cv,
                                      method="predict_proba", n_jobs=1)[:,1]
                cols["base_xgb_auc"].append(roc_auc_score(E_s, p))
                cols["base_xgb_pr"].append(average_precision_score(E_s, p))

                p = cross_val_predict(make_lr(), X_h, E_s, cv=cv,
                                      method="predict_proba", n_jobs=1)[:,1]
                cols["hybrid_lr_auc"].append(roc_auc_score(E_s, p))
                cols["hybrid_lr_pr"].append(average_precision_score(E_s, p))

                p = cross_val_predict(make_xgb(pos_w), X_h, E_s, cv=cv,
                                      method="predict_proba", n_jobs=1)[:,1]
                cols["hybrid_xgb_auc"].append(roc_auc_score(E_s, p))
                cols["hybrid_xgb_pr"].append(average_precision_score(E_s, p))
            except Exception:
                pass
        else:
            test_idx = np.setdiff1d(np.arange(n_full), idx)
            X_tb = df.iloc[test_idx][BASE_FEATURES].values
            X_th = df.iloc[test_idx][HYBRID_FEATURES].values
            E_t  = E_full[test_idx]
            if E_t.sum() < 5:
                continue
            try:
                m = make_lr(); m.fit(X_b, E_s)
                p = m.predict_proba(X_tb)[:,1]
                cols["base_lr_auc"].append(roc_auc_score(E_t, p))
                cols["base_lr_pr"].append(average_precision_score(E_t, p))

                m = make_xgb(pos_w); m.fit(X_b, E_s)
                p = m.predict_proba(X_tb)[:,1]
                cols["base_xgb_auc"].append(roc_auc_score(E_t, p))
                cols["base_xgb_pr"].append(average_precision_score(E_t, p))

                m = make_lr(); m.fit(X_h, E_s)
                p = m.predict_proba(X_th)[:,1]
                cols["hybrid_lr_auc"].append(roc_auc_score(E_t, p))
                cols["hybrid_lr_pr"].append(average_precision_score(E_t, p))

                m = make_xgb(pos_w); m.fit(X_h, E_s)
                p = m.predict_proba(X_th)[:,1]
                cols["hybrid_xgb_auc"].append(roc_auc_score(E_t, p))
                cols["hybrid_xgb_pr"].append(average_precision_score(E_t, p))
            except Exception:
                pass

        print(".", end="", flush=True)

    row = {"n": n_full if use_full else n_sample}
    for key, vals in cols.items():
        if vals:
            row[f"{key}_mean"] = round(np.mean(vals), 4)
            row[f"{key}_std"]  = round(np.std(vals),  4)
        else:
            row[f"{key}_mean"] = np.nan
            row[f"{key}_std"]  = np.nan

    results.append(row)
    print(f"  b12={row['b12_auc_mean']:.3f}  "
          f"base={row['base_xgb_auc_mean']:.3f}  "
          f"hybrid={row['hybrid_xgb_auc_mean']:.3f}")

res_df = pd.DataFrame(results)
res_df.to_csv("sample_size_results.csv", index=False)
print("\nSaved: sample_size_results.csv")


# PLOTS
# FINAL THESIS FIGURE: ROC-AUC + PR-AUC

# CLEAN: ONLY "ALL MODELS" PLOTS (NO SHADING)

ns = res_df["n"].values

def line(ax, key, color, label, ls="-", marker=None):
    m = res_df[f"{key}_mean"].values.astype(float)

    ax.plot(
        ns, m,
        color=color,
        lw=2.3,
        ls=ls,
        marker=marker,
        label=label
    )

def format_axis(ax, title, ylabel, baseline):
    ax.axhline(baseline, color="gray", ls=":", lw=1, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(x)) for x in ns], rotation=35)

    ax.set_xlabel("Training sample size (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")

    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")

# Create figure (2 rows: ROC + PR)
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# 1. ROC-AUC (ALL MODELS)

ax = axes[0]

line(ax, "b12_auc",        "#e63946", "Burr XII", "-")
line(ax, "base_xgb_auc",   "#457b9d", "Baseline XGBoost", "-.")
line(ax, "base_lr_auc",    "#2a9d8f", "Baseline LR", ":")
line(ax, "hybrid_xgb_auc", "#c77dff", "Hybrid XGBoost", "-")
line(ax, "hybrid_lr_auc",  "#9b72cf", "Hybrid LR", "--")

format_axis(ax, "ROC-AUC: All Models", "AUC", baseline=0.5)

# 2. PR-AUC (ALL MODELS)

ax = axes[1]

baseline_pr = E_full.mean()

line(ax, "b12_pr",        "#e63946", "Burr XII", "-")
line(ax, "base_xgb_pr",   "#457b9d", "Baseline XGBoost", "-.")
line(ax, "base_lr_pr",    "#2a9d8f", "Baseline LR", ":")
line(ax, "hybrid_xgb_pr", "#c77dff", "Hybrid XGBoost", "-")
line(ax, "hybrid_lr_pr",  "#9b72cf", "Hybrid LR", "--")

format_axis(ax, "PR-AUC: All Models", "PR-AUC", baseline=baseline_pr)

# Save
plt.tight_layout()
plt.savefig("thesis_all_models_clean.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: thesis_all_models_clean.png")

# SUMMARY

print("\n" + "=" * 78)
print("SUMMARY TABLE")
print(f"\n  {'n':>6}  {'Burr XII':>9}  {'Base XGB':>9}  "
      f"{'Hybrid XGB':>11}  {'ΔAUC':>8}  "
      f"{'c±sd':>10}  {'k±sd':>10}  {'σ±sd':>10}")
print("  " + "-"*78)

for _, r in res_df.iterrows():
    d = r["hybrid_xgb_auc_mean"] - r["base_xgb_auc_mean"]
    s = " ✓" if d > 0.002 else "  "
    print(f"  {int(r['n']):>6}  "
          f"{r['b12_auc_mean']:>9.4f}  "
          f"{r['base_xgb_auc_mean']:>9.4f}  "
          f"{r['hybrid_xgb_auc_mean']:>11.4f}  "
          f"{d:>+8.4f}{s}  "
          f"{r['b12_c_mean']:.2f}±{r['b12_c_std']:.2f}  "
          f"{r['b12_k_mean']:.3f}±{r['b12_k_std']:.3f}  "
          f"{r['b12_sigma_mean']:.1f}±{r['b12_sigma_std']:.1f}")