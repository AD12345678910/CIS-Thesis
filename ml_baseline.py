"""
ml_baseline.py
Trains and evaluates baseline ML classifiers (Logistic Regression, Random
Forest, XGBoost) on static customer covariates, with no tenure or survival
features. Establishes the ML baseline performance.

Usage: python ml_baseline.py  (run burr12_fit.py first)
"""

import os, glob, warnings
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, log_loss,
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Find features file ─────────────────────────────────────────────────────
candidates = glob.glob("*.csv") + glob.glob("**/*.csv", recursive=True)
feat_file  = next((f for f in candidates if "burr12_features" in f), None)
if feat_file is None:
    raise FileNotFoundError("Cannot find telco_burr12_features.csv — run burr12_fit.py first.")
print(f"Using: {feat_file}")

# 1. LOAD

print("PLAIN ML BASELINE — IBM Telco Churn")

df = pd.read_csv(feat_file)

# ── Use the same 80/20 split as burr12_fit.py ─────────────────────────────
# The 'split' column was added by burr12_fit.py so all three scripts
# evaluate on exactly the same held-out 20% test set.
if "split" not in df.columns:
    raise ValueError("'split' column missing — re-run burr12_fit.py to regenerate telco_burr12_features.csv")

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df  = df[df["split"] == "test"].reset_index(drop=True)

print(f"Train : {len(train_df)} customers  ({train_df['event'].mean():.4f} churn rate)")
print(f"Test  : {len(test_df)} customers  ({test_df['event'].mean():.4f} churn rate)")

# NO tenure — Burr XII features carry the time-to-event information.
# Removing tenure forces the model to rely on static covariates only,
# so any lift from the hybrid is cleanly attributable to Burr XII.
ML_FEATURES = [
    "is_male", "is_senior", "partner", "dependents", "phone",
    "internet_fiber", "internet_dsl",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
    "multiple_lines", "contract_1yr", "contract_2yr",
    "pay_echeck", "paperless", "MonthlyCharges_z",
]

X_train = train_df[ML_FEATURES].values
y_train = train_df["event"].values
X_test  = test_df[ML_FEATURES].values
y_test  = test_df["event"].values

print(f"\nDataset : {len(df)} total  |  train={len(train_df)}  test={len(test_df)}")
print(f"Features: {len(ML_FEATURES)}  (no tenure, no survival features)")


# 2. TRAIN ON TRAIN SET, EVALUATE ON TEST SET
# Uses the same 80/20 split as burr12_fit.py so all models are evaluated
# on identical held-out customers — results are directly comparable.

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=1,
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(1-y_train.mean())/y_train.mean(),
        eval_metric="logloss", random_state=42, verbosity=0,
    ),
}

print("\n" + "=" * 62)
print("TRAIN → TEST METRICS  (80/20 split)")
print(f"\n  {'Model':<22} {'AUC':>7} {'PR-AUC':>8} {'LogLoss':>9} {'Brier':>7}")
print("  " + "-"*56)

results   = {}
test_preds = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    m = {
        "AUC":     round(roc_auc_score(y_test, prob), 4),
        "PR_AUC":  round(average_precision_score(y_test, prob), 4),
        "LogLoss": round(log_loss(y_test, prob), 4),
        "Brier":   round(brier_score_loss(y_test, prob), 4),
    }
    results[name]    = m
    test_preds[name] = prob
    print(f"  {name:<22} {m['AUC']:>7.4f} {m['PR_AUC']:>8.4f} "
          f"{m['LogLoss']:>9.4f} {m['Brier']:>7.4f}")

best_name = max(results, key=lambda k: results[k]["AUC"])
print(f"\nBest model by AUC: {best_name}  (AUC={results[best_name]['AUC']:.4f})")


# 3. FEATURE IMPORTANCE (fit on full train set)

print("\n" + "=" * 62)
print("FEATURE IMPORTANCE")

rf_full = RandomForestClassifier(n_estimators=300, max_depth=8,
    min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=1)
rf_full.fit(X_train, y_train)
rf_imp = pd.DataFrame({"feature": ML_FEATURES,
                        "importance": rf_full.feature_importances_}
                      ).sort_values("importance", ascending=False)
print("\nRandom Forest — top 10:")
print(rf_imp.head(10).to_string(index=False))

xgb_full = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(1-y_train.mean())/y_train.mean(),
    eval_metric="logloss", random_state=42, verbosity=0)
xgb_full.fit(X_train, y_train)
xgb_imp = pd.DataFrame({"feature": ML_FEATURES,
                         "importance": xgb_full.feature_importances_}
                       ).sort_values("importance", ascending=False)
print("\nXGBoost — top 10:")
print(xgb_imp.head(10).to_string(index=False))


# 4. PLOTS

colors = {"Logistic Regression": "#457b9d",
          "Random Forest":       "#2a9d8f",
          "XGBoost":             "#e63946"}

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Plain ML Baseline (no tenure) — IBM Telco Churn  [80/20 split]",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
for name, prob in test_preds.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax1.plot(fpr, tpr, color=colors[name], lw=2,
             label=f"{name}  AUC={results[name]['AUC']:.3f}")
ax1.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5)
ax1.set(xlabel="FPR", ylabel="TPR", title="ROC Curves (test set)", xlim=(0,1), ylim=(0,1))
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
for name, prob in test_preds.items():
    p, r, _ = precision_recall_curve(y_test, prob)
    ax2.plot(r, p, color=colors[name], lw=2,
             label=f"{name}  PR={results[name]['PR_AUC']:.3f}")
ax2.axhline(y_test.mean(), color="gray", ls="--", lw=1, label=f"Baseline={y_test.mean():.2f}")
ax2.set(xlabel="Recall", ylabel="Precision", title="PR Curves (test set)", xlim=(0,1), ylim=(0,1))
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
for name, prob in test_preds.items():
    fp, mp = calibration_curve(y_test, prob, n_bins=10)
    ax3.plot(mp, fp, color=colors[name], lw=2, marker="o", ms=4, label=name)
ax3.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5, label="Perfect")
ax3.set(xlabel="Mean Predicted", ylabel="Fraction Positive",
        title="Calibration (test set)", xlim=(0,1), ylim=(0,1))
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1, 0])
metric_names = ["AUC", "PR_AUC"]
x = np.arange(len(metric_names)); w = 0.25
for i, (name, res) in enumerate(results.items()):
    ax4.bar(x + i*w - w, [res[m] for m in metric_names], w,
            label=name, color=colors[name], alpha=0.85)
ax4.set_xticks(x); ax4.set_xticklabels(metric_names)
ax4.set(ylabel="Score", title="AUC & PR-AUC (test set)", ylim=(0,1))
ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
top10 = rf_imp.head(10)
ax5.barh(top10["feature"][::-1], top10["importance"][::-1],
         color="#2a9d8f", alpha=0.85)
ax5.set(xlabel="Importance", title="Random Forest — Top 10")
ax5.grid(axis="x", alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
top10x = xgb_imp.head(10)
ax6.barh(top10x["feature"][::-1], top10x["importance"][::-1],
         color="#e63946", alpha=0.85)
ax6.set(xlabel="Importance", title="XGBoost — Top 10")
ax6.grid(axis="x", alpha=0.3)

plt.savefig("ml_baseline_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: ml_baseline_plots.png")


# 5. SAVE

pd.DataFrame(results).T.reset_index().rename(
    columns={"index": "Model"}).to_csv("ml_baseline_metrics.csv", index=False)

preds_df = pd.DataFrame({"y_true": y_test})
for name, prob in test_preds.items():
    preds_df[f"prob_{name.replace(' ','_')}"] = prob
preds_df.to_csv("ml_baseline_predictions.csv", index=False)

print("\n" + "=" * 62)
print("SUMMARY")
print(f"\n  Train: {len(y_train)}  Test: {len(y_test)}  (80/20 split, same as burr12_fit.py)")
print(f"\n  {'Model':<22} {'AUC':>7} {'PR-AUC':>8} {'LogLoss':>9} {'Brier':>7}")
print("  " + "-"*55)
for name, res in results.items():
    marker = " ← best" if name == best_name else ""
    print(f"  {name:<22} {res['AUC']:>7.4f} {res['PR_AUC']:>8.4f} "
          f"{res['LogLoss']:>9.4f} {res['Brier']:>7.4f}{marker}")
print("""
Files saved:
  ml_baseline_metrics.csv      — metrics table
  ml_baseline_predictions.csv  — test set probabilities
  ml_baseline_plots.png        — ROC, PR, calibration, importance

Next: run ml_hybrid.py
""")
# ── Feature importance plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Feature Importance — Baseline ML Models",
             fontsize=12, fontweight="bold")

DISPLAY_NAMES = {
    "is_male": "Gender (male)", "is_senior": "Senior citizen",
    "partner": "Partner", "dependents": "Dependents",
    "phone": "Phone service", "internet_fiber": "Internet: fiber optic",
    "internet_dsl": "Internet: DSL", "online_security": "Online security",
    "online_backup": "Online backup", "device_protection": "Device protection",
    "tech_support": "Tech support", "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming movies", "multiple_lines": "Multiple lines",
    "contract_1yr": "Contract: one year", "contract_2yr": "Contract: two year",
    "pay_echeck": "Payment: e-check", "paperless": "Paperless billing",
    "MonthlyCharges_z": "Monthly charges (z)",
}

for ax, model, title, col in zip(
    axes,
    [rf_full, xgb_full],
    ["Random Forest", "XGBoost"],
    ["#2a9d8f", "#e63946"],
):
    imp = pd.DataFrame({
        "display": [DISPLAY_NAMES.get(f, f) for f in ML_FEATURES],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(12)
    ax.barh(imp["display"][::-1], imp["importance"][::-1],
            color=col, alpha=0.88, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Feature importance")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("fig_feature_importance_baseline.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: fig_feature_importance_baseline.png")