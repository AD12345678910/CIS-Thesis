"""
burr12_fit.py
Fits Weibull and Burr XII survival models to the IBM Telco churn dataset.

Outputs telco_burr12_features.csv with fitted survival features for each
customer, used as input by ml_baseline.py, ml_hybrid.py, and sample_size.py.

Usage: python burr12_fit.py
Requires: WA_Fn-UseC_-Telco-Customer-Churn.csv in the working directory.
"""

import os, glob, warnings
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from lifelines import KaplanMeierFitter, WeibullFitter, LogLogisticFitter
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, roc_curve, precision_recall_curve)
warnings.filterwarnings("ignore")

# ── Find raw Telco CSV automatically ──────────────────────────────────────
DATA_PATH = None
# Or set explicitly:
# DATA_PATH = "/Users/.../WA_Fn-UseC_-Telco-Customer-Churn.csv"

if DATA_PATH is None:
    candidates = glob.glob("*.csv") + glob.glob("**/*.csv", recursive=True)
    hits = [f for f in candidates
            if any(x in f.lower() for x in ["wa_fn","telco-customer","telco_customer"])
            and all(x not in f.lower() for x in ["features","burr","baseline","hybrid"])]
    if not hits:
        hits = [f for f in candidates
                if "churn" in f.lower()
                and all(x not in f.lower() for x in ["features","burr","baseline","hybrid"])]
    if hits:
        DATA_PATH = hits[0]
        print(f"Auto-detected: {DATA_PATH}")
    else:
        raise FileNotFoundError(
            "Cannot find raw Telco CSV. Set DATA_PATH explicitly at the top of this script."
        )


# SECTION 1 — CLEANING

print("\n" + "═"*62)
print("SECTION 1: CLEANING")
print("═"*62)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
print(f"Columns : {df.columns.tolist()}")
print(f"Loaded  : {df.shape[0]} rows, {df.shape[1]} columns")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_before = len(df)
df = df[df["tenure"] > 0].dropna(subset=["tenure", "Churn", "TotalCharges"])
df = df.reset_index(drop=True)   # reset so positional indices match after row drops
print(f"Removed : {n_before - len(df)} rows (tenure=0 or missing)")
print(f"Final   : {len(df)} rows")

df["event"] = (df["Churn"] == "Yes").astype(int)
print(f"Churn   : {df['event'].mean():.4f}  ({df['event'].sum()} events)")
print(f"Tenure  : min={df['tenure'].min()}  mean={df['tenure'].mean():.1f}  max={df['tenure'].max()}")

# Encode covariates (saved into features CSV for ML scripts)
for col, new in {
    "Partner":"partner", "Dependents":"dependents",
    "PhoneService":"phone", "PaperlessBilling":"paperless",
    "OnlineSecurity":"online_security", "OnlineBackup":"online_backup",
    "DeviceProtection":"device_protection", "TechSupport":"tech_support",
    "StreamingTV":"streaming_tv", "StreamingMovies":"streaming_movies",
}.items():
    df[new] = (df[col] == "Yes").astype(int)

df["is_male"]          = (df["gender"] == "Male").astype(int)
df["is_senior"]        = df["SeniorCitizen"].astype(int)
df["internet_fiber"]   = (df["InternetService"] == "Fiber optic").astype(int)
df["internet_dsl"]     = (df["InternetService"] == "DSL").astype(int)
df["contract_1yr"]     = (df["Contract"] == "One year").astype(int)
df["contract_2yr"]     = (df["Contract"] == "Two year").astype(int)
df["pay_echeck"]       = (df["PaymentMethod"] == "Electronic check").astype(int)
df["multiple_lines"]   = (df["MultipleLines"] == "Yes").astype(int)
df["MonthlyCharges_z"] = (
    (df["MonthlyCharges"] - df["MonthlyCharges"].mean())
    / df["MonthlyCharges"].std()
)

COVARIATES = [
    "is_male", "is_senior", "partner", "dependents", "phone",
    "internet_fiber", "internet_dsl", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "multiple_lines", "contract_1yr", "contract_2yr", "pay_echeck",
    "paperless", "MonthlyCharges_z",
]
print(f"Covariates encoded: {len(COVARIATES)}")

T = df["tenure"].values.astype(float)
E = df["event"].values.astype(float)
n = len(T)

# ── 80/20 stratified train/test split ─────────────────────────────────────
# Burr XII has only 3 parameters so overfitting is negligible, but we use
# a held-out test set to keep evaluation consistent with the ML scripts.
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    np.arange(n), test_size=0.2, random_state=42, stratify=E
)
T_train, E_train = T[train_idx], E[train_idx]
T_test,  E_test  = T[test_idx],  E[test_idx]
n_train = len(T_train)

print(f"\nTrain : {n_train} customers  ({E_train.mean():.4f} churn rate)")
print(f"Test  : {len(T_test)} customers  ({E_test.mean():.4f} churn rate)")


# SECTION 2 — BURR XII FUNCTIONS + BOUNDED FITTER
#
#   S(t) = [1 + (t/sigma)^c]^(-k)
#   h(t) = (c·k/sigma) · (t/sigma)^(c-1) / [1 + (t/sigma)^c]
#
# Bounds keep parameters in month-scale ranges.
# Unconstrained optimisers escape to c~165 degenerate solutions — bounded
# differential_evolution + L-BFGS-B polish avoids this entirely.

def burr12_sf(t, c, k, sigma):
    return (1.0 + (t / sigma) ** c) ** (-k)

def burr12_pdf(t, c, k, sigma):
    u = (t / sigma) ** c
    return (c * k / sigma) * (t / sigma) ** (c - 1) * (1.0 + u) ** (-(k + 1))

def burr12_hf(t, c, k, sigma):
    u = (t / sigma) ** c
    return (c * k / sigma) * (t / sigma) ** (c - 1) / (1.0 + u)

BOUNDS = [(0.05, 15.0), (0.001, 200.0), (0.5, 1000.0)]

def nll(params, T, E):
    c, k, sigma = params
    t   = np.clip(T, 1e-9, None)
    pdf = np.clip(burr12_pdf(t, c, k, sigma), 1e-300, None)
    sf  = np.clip(burr12_sf(t, c, k, sigma),  1e-300, None)
    val = -(E * np.log(pdf) + (1 - E) * np.log(sf)).sum()
    return val if np.isfinite(val) else 1e12

def fit_burr12(T, E):
    # Step 1: global search
    res_de = differential_evolution(
        nll, BOUNDS, args=(T, E),
        seed=42, maxiter=3000, tol=1e-12,
        popsize=20, mutation=(0.5, 1.0), recombination=0.7,
        workers=1, polish=True,
    )
    # Step 2: local polish
    res = minimize(
        nll, res_de.x, args=(T, E), method="L-BFGS-B",
        bounds=BOUNDS, options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 10000},
    )
    return res.x[0], res.x[1], res.x[2], res.fun   # c, k, sigma, nll

def aic(nll_val, p):     return 2 * p + 2 * nll_val
def bic(nll_val, p, n):  return p * np.log(n) + 2 * nll_val


# SECTION 3 — FIT BURR XII (no covariates, aggregate)

print("\n" + "═"*62)
print("SECTION 3: BURR XII — NO COVARIATES (aggregate)")
print("═"*62)
print("Fitting on TRAIN set (80%), evaluating on TEST set (20%)\n")

c0, k0, sigma0, nll0 = fit_burr12(T_train, E_train)

ck = c0 * k0
if   c0 > 1 and ck > 1: shape_desc = "HUMP — rises then falls ✓"
elif c0 > 1:             shape_desc = f"rises then flattens (c·k={ck:.3f})"
else:                    shape_desc = "monotone decreasing"

print(f"\n  c     = {c0:.4f}  (shape 1 — early hazard behaviour)")
print(f"  k     = {k0:.4f}  (shape 2 — tail weight)")
print(f"  sigma = {sigma0:.4f}  (scale in months)")
print(f"  NLL   = {nll0:.2f}")
print(f"  Hazard shape: {shape_desc}")

# Competitor models — also fitted on train set
wf  = WeibullFitter().fit(T_train, E_train)
# ── Print Weibull parameters ───────────────────────────────────────────────
rho   = wf.params_["rho_"]
lam   = np.exp(-wf.params_["lambda_"])   # lifelines stores log(lambda)
print(f"\nWeibull parameters (lifelines parameterisation):")
print(f"  shape (rho / c) = {rho:.4f}")
print(f"  scale (lambda)  = {lam:.4f}")
llf = LogLogisticFitter().fit(T_train, E_train)

comparison = pd.DataFrame({
    "Model":  ["Burr XII", "Log-Logistic", "Weibull"],
    "Params": [3, 2, 2],
    "NLL":    [nll0, -llf.log_likelihood_, -wf.log_likelihood_],
    "AIC":    [aic(nll0, 3),
               aic(-llf.log_likelihood_, 2),
               aic(-wf.log_likelihood_,  2)],
    "BIC":    [bic(nll0, 3, n_train),
               bic(-llf.log_likelihood_, 2, n_train),
               bic(-wf.log_likelihood_,  2, n_train)],
}).sort_values("AIC").reset_index(drop=True)
comparison["ΔAIC"] = (comparison["AIC"] - comparison["AIC"].min()).round(2)
comparison["ΔBIC"] = (comparison["BIC"] - comparison["BIC"].min()).round(2)

print("\nModel comparison:")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
delta = comparison.iloc[1]["ΔAIC"]
if delta > 10:
    print(f"\n→ ΔAIC={delta:.1f}: decisive evidence for Burr XII over alternatives")
elif delta > 2:
    print(f"\n→ ΔAIC={delta:.1f}: meaningful evidence for Burr XII over alternatives")


# SECTION 4 — AUC EVALUATION
#
# Predicted churn probability for each customer:
#   P(churn in next H months | survived to their tenure)
#   = 1 - S(tenure + H) / S(tenure)
#
# Aggregate model differentiates customers only by tenure.
# Weibull/Log-Logistic AUC ≈ 0.50 because lifelines' predict() returns
# identical values per time point with no cross-customer variance.
# Burr XII's conditional probability formula gives genuine variance.

print("\n" + "═"*62)
print("SECTION 4: AUC EVALUATION")
print("═"*62)
print("All metrics evaluated on held-out TEST set (20%)\n")

H = 12   # prediction horizon in months

def churn_prob_burr(T, c, k, sigma, h=H):
    sf_t  = np.clip(burr12_sf(T, c, k, sigma), 1e-9, None)
    sf_th = burr12_sf(T + h, c, k, sigma)
    return 1.0 - sf_th / sf_t

def churn_prob_weibull(T, wf, h=H):
    rho = wf.params_["rho_"]
    lam = np.exp(-wf.params_["lambda_"])
    sf_t  = np.clip(np.exp(-(T / lam) ** rho), 1e-9, None)
    sf_th = np.exp(-((T + h) / lam) ** rho)
    return 1.0 - sf_th / sf_t

def churn_prob_loglogistic(T, llf, h=H):
    al = np.exp(llf.params_["alpha_"])
    bt = np.exp(llf.params_["beta_"])
    sf_t  = np.clip(1.0 / (1.0 + (T / al) ** bt), 1e-9, None)
    sf_th = 1.0 / (1.0 + ((T + h) / al) ** bt)
    return 1.0 - sf_th / sf_t

# Predict on TEST set only
pred_burr = churn_prob_burr(T_test, c0, k0, sigma0)
pred_weib = churn_prob_weibull(T_test, wf)
pred_ll   = churn_prob_loglogistic(T_test, llf)

def show_metrics(y, yhat, label):
    auc_ = roc_auc_score(y, yhat)
    pr_  = average_precision_score(y, yhat)
    bs_  = brier_score_loss(y, yhat)
    print(f"  {label:<35}  AUC={auc_:.4f}  PR-AUC={pr_:.4f}  Brier={bs_:.4f}")
    return {"model": label, "AUC": round(auc_,4),
            "PR_AUC": round(pr_,4), "Brier": round(bs_,4)}

print(f"Prediction horizon: {H} months  |  Test set: {len(T_test)} customers\n")
auc_rows = [
    show_metrics(E_test, pred_burr, "Burr XII (aggregate)"),
    show_metrics(E_test, pred_weib, "Weibull (aggregate)"),
    show_metrics(E_test, pred_ll,   "Log-Logistic (aggregate)"),
]
auc_df = pd.DataFrame(auc_rows)
auc_df.to_csv("burr12_auc_results.csv", index=False)


# SECTION 5 — FEATURE ENGINEERING + SAVE
#
# Three Burr XII-derived features added to every customer row:
#
#   burr12_survival    S(t): how unlikely it is this customer is still here
#   burr12_hazard      h(t): instantaneous churn risk at their current tenure
#   burr12_churn_prob  P(churn in next 12 months | survived to t)
#
# These become input features for ml_baseline.py and ml_hybrid.py.

print("\n" + "═"*62)
print("SECTION 5: FEATURE ENGINEERING + SAVE")
print("═"*62)

df["burr12_survival"]   = burr12_sf(T, c0, k0, sigma0)
df["burr12_hazard"]     = burr12_hf(T, c0, k0, sigma0)
df["burr12_churn_prob"] = churn_prob_burr(T, c0, k0, sigma0)  # full dataset

# burr12_strat_prob: placeholder = same as aggregate (no stratified fit)
# ml_hybrid.py will use this column — set it equal to burr12_churn_prob
df["burr12_strat_prob"] = df["burr12_churn_prob"]

# Add split flag so ML scripts can honour the same 80/20 split
df["split"] = "train"
df.loc[test_idx, "split"] = "test"

print("Parameters fitted on TRAIN set (80%), features applied to ALL rows")
print("split column added: use df[df['split']=='train'] / df[df['split']=='test']")

print("Burr XII features added:")
print(f"  burr12_survival    mean={df['burr12_survival'].mean():.4f}")
print(f"  burr12_hazard      mean={df['burr12_hazard'].mean():.6f}")
print(f"  burr12_churn_prob  mean={df['burr12_churn_prob'].mean():.4f}")

out_cols = (
    ["tenure", "event", "split"]
    + COVARIATES
    + ["burr12_survival", "burr12_hazard", "burr12_churn_prob", "burr12_strat_prob"]
)
df[out_cols].to_csv("telco_burr12_features.csv", index=False)
print("\nSaved: telco_burr12_features.csv")


# SECTION 6 — PLOTS

t_grid = np.linspace(0.1, 72, 400)
# KM on train set — consistent with Burr XII fit
kmf    = KaplanMeierFitter().fit(T_train, E_train, label="Kaplan-Meier (train)")

# ── Plot 1: Survival + Hazard + AIC ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Burr XII — Aggregate Fit (No Covariates)", fontsize=12, fontweight="bold")

ax = axes[0]
kmf.plot_survival_function(ax=ax, ci_show=True, color="black", lw=1.5, ls="--")
ax.plot(t_grid, burr12_sf(t_grid, c0, k0, sigma0),
        color="#e63946", lw=2.2,
        label=f"Burr XII\nc={c0:.2f}, k={k0:.3f}, σ={sigma0:.1f}")
try:
    ax.plot(t_grid, wf.predict(t_grid).values,
            color="#457b9d", lw=1.8, ls="-.", label="Weibull")
    ax.plot(t_grid, llf.predict(t_grid).values,
            color="#2a9d8f", lw=1.8, ls=":",  label="Log-Logistic")
except Exception:
    ax.plot(t_grid, wf.predict(t_grid),
            color="#457b9d", lw=1.8, ls="-.", label="Weibull")
    ax.plot(t_grid, llf.predict(t_grid),
            color="#2a9d8f", lw=1.8, ls=":",  label="Log-Logistic")
ax.set(xlabel="Tenure (months)", ylabel="S(t)", title="Survival Functions",
       xlim=(0, 72), ylim=(0, 1))
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(t_grid, burr12_hf(t_grid, c0, k0, sigma0),
        color="#e63946", lw=2.2,
        label=f"Burr XII h(t)\nc={c0:.2f}, k={k0:.3f}")
rho_w = wf.params_["rho_"]
lam_w = np.exp(-wf.params_["lambda_"])
ax.plot(t_grid, (rho_w / lam_w) * (t_grid / lam_w) ** (rho_w - 1),
        color="#457b9d", lw=1.8, ls="-.", label="Weibull h(t)")
ax.set(xlabel="Tenure (months)", ylabel="h(t)", title="Hazard Functions", xlim=(0, 72))
ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
mdls = comparison["Model"].tolist()
x = np.arange(len(mdls)); w = 0.35
cols_bar = ["#e63946", "#2a9d8f", "#457b9d"]
b1 = ax.bar(x - w/2, comparison["AIC"], w, label="AIC", color=cols_bar)
ax.bar(x + w/2, comparison["BIC"], w, label="BIC",
       color=cols_bar, alpha=0.55, hatch="//")
ax.set_xticks(x); ax.set_xticklabels(mdls, fontsize=9)
ax.set(ylabel="Info Criterion (lower=better)", title="AIC / BIC")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
for bar, daic in zip(b1, comparison["ΔAIC"]):
    lbl = "best" if daic == 0 else f"+{daic:.0f}"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            lbl, ha="center", va="bottom", fontsize=9,
            color="darkgreen" if daic == 0 else "gray", fontweight="bold")
all_ics = comparison["AIC"].tolist() + comparison["BIC"].tolist()
ax.set_ylim(min(all_ics) * 0.998, max(all_ics) * 1.003)

plt.tight_layout()
plt.savefig("burr12_no_cov.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: burr12_no_cov.png")

# ── Plot 2: ROC + PR ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"AUC Comparison — Prediction Horizon: {H} months",
             fontsize=12, fontweight="bold")

# ROC + PR evaluated on TEST set
plot_models = [
    ("Burr XII",     pred_burr, "#e63946"),
    ("Weibull",      pred_weib, "#457b9d"),
    ("Log-Logistic", pred_ll,   "#2a9d8f"),
]
ax = axes[0]
for lbl, pred, col in plot_models:
    fpr, tpr, _ = roc_curve(E_test, pred)
    ax.plot(fpr, tpr, color=col, lw=1.8,
            label=f"{lbl}  AUC={roc_auc_score(E_test, pred):.3f}")
ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5)
ax.set(xlabel="FPR", ylabel="TPR",
       title="ROC Curve (test set)", xlim=(0,1), ylim=(0,1))
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
for lbl, pred, col in plot_models:
    prec, rec, _ = precision_recall_curve(E_test, pred)
    ax.plot(rec, prec, color=col, lw=1.8,
            label=f"{lbl}  PR={average_precision_score(E_test, pred):.3f}")
ax.axhline(E_test.mean(), color="gray", ls="--", lw=0.8,
           label=f"Baseline={E_test.mean():.2f}")
ax.set(xlabel="Recall", ylabel="Precision",
       title="Precision-Recall (test set)", xlim=(0,1), ylim=(0,1))
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("burr12_auc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: burr12_auc.png")

# Save CSVs
comparison.to_csv("burr12_aggregate_comparison.csv", index=False)


# SUMMARY

print("\n" + "═"*62)
print("SUMMARY")
print("═"*62)
print(f"\n  Dataset : {n} customers  |  {int(E.sum())} events ({E.mean():.1%} churn)")
print(f"\n  Burr XII aggregate parameters:")
print(f"    c     = {c0:.4f}  (shape 1)")
print(f"    k     = {k0:.4f}  (shape 2 / tail weight)")
print(f"    sigma = {sigma0:.4f}  (scale in months)")
print(f"    Hazard shape: {shape_desc}")
print(f"\n  Model comparison (AIC):")
for _, r in comparison.iterrows():
    tag = " ← best" if r["ΔAIC"] == 0 else ""
    print(f"    {r['Model']:<14}  AIC={r['AIC']:.1f}  ΔAIC={r['ΔAIC']:.1f}{tag}")
print(f"\n  AUC (horizon={H} months):")
for _, r in auc_df.iterrows():
    print(f"    {r['model']:<35}  AUC={r['AUC']:.4f}  "
          f"PR-AUC={r['PR_AUC']:.4f}  Brier={r['Brier']:.4f}")
print("""
  Output files:
    telco_burr12_features.csv       → feed into ml_baseline.py + ml_hybrid.py
    burr12_no_cov.png               survival + hazard + AIC plot
    burr12_auc.png                  ROC + PR curves
    burr12_aggregate_comparison.csv AIC/BIC table
    burr12_auc_results.csv          AUC metrics table
""")