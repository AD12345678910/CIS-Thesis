"""
ml_hybrid.py
Trains and evaluates hybrid models that combine static customer covariates
with Burr XII survival features (S(t), h(t), next-period churn probability).
Compares performance against the baseline from ml_baseline.py.

Usage: python ml_hybrid.py  (run burr12_fit.py and ml_baseline.py first)
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


# 1. LOAD + DEFINE FEATURE SETS

print("HYBRID ML — Burr XII + ML — IBM Telco Churn")

df = pd.read_csv(feat_file)

# ── Use the same 80/20 split as burr12_fit.py ─────────────────────────────
if "split" not in df.columns:
    raise ValueError("'split' column missing — re-run burr12_fit.py first.")

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df  = df[df["split"] == "test"].reset_index(drop=True)

print(f"Train : {len(train_df)} customers  ({train_df['event'].mean():.4f} churn rate)")
print(f"Test  : {len(test_df)} customers  ({test_df['event'].mean():.4f} churn rate)")

BASE_FEATURES = [
    "is_male", "is_senior", "partner", "dependents", "phone",
    "internet_fiber", "internet_dsl",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
    "multiple_lines", "contract_1yr", "contract_2yr",
    "pay_echeck", "paperless", "MonthlyCharges_z",
]

SURVIVAL_FEATURES = [
    "burr12_survival",    # S(t)
    "burr12_hazard",      # h(t)
    "burr12_churn_prob",  # P(churn next 12m | survived to t)
]

HYBRID_FEATURES = BASE_FEATURES + SURVIVAL_FEATURES

y_train     = train_df["event"].values
y_test      = test_df["event"].values
X_train_b   = train_df[BASE_FEATURES].values
X_test_b    = test_df[BASE_FEATURES].values
X_train_h   = train_df[HYBRID_FEATURES].values
X_test_h    = test_df[HYBRID_FEATURES].values

print(f"\nDataset        : {len(df)} total  |  train={len(train_df)}  test={len(test_df)}")
print(f"Base features  : {len(BASE_FEATURES)}  (covariates only, no tenure)")
print(f"Hybrid features: {len(HYBRID_FEATURES)}  (+{len(SURVIVAL_FEATURES)} Burr XII features)")
print(f"Survival features: {SURVIVAL_FEATURES}")

# 2. MODEL FACTORY + TRAIN/TEST EVALUATION

def make_models(y_tr):
    return {
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
            scale_pos_weight=(1-y_tr.mean())/y_tr.mean(),
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
    }

def run_split(X_tr, y_tr, X_te, y_te, label):
    """Train on train set, evaluate on test set."""
    models  = make_models(y_tr)
    results = {}
    preds   = {}
    print(f"\n  [{label}]")
    print(f"  {'Model':<22} {'AUC':>7} {'PR-AUC':>8} {'LogLoss':>9} {'Brier':>7}")
    print("  " + "-"*56)
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        m = {
            "AUC":     round(roc_auc_score(y_te, prob), 4),
            "PR_AUC":  round(average_precision_score(y_te, prob), 4),
            "LogLoss": round(log_loss(y_te, prob), 4),
            "Brier":   round(brier_score_loss(y_te, prob), 4),
        }
        results[name] = m
        preds[name]   = prob
        print(f"  {name:<22} {m['AUC']:>7.4f} {m['PR_AUC']:>8.4f} "
              f"{m['LogLoss']:>9.4f} {m['Brier']:>7.4f}")
    return results, preds


# 3. RUN BOTH FEATURE SETS

print("\n" + "=" * 64)
print("TRAIN → TEST METRICS  (80/20 split, same as burr12_fit.py)")

res_base,   preds_base   = run_split(X_train_b, y_train, X_test_b, y_test,
                                      "BASELINE  — covariates only, no tenure")
res_hybrid, preds_hybrid = run_split(X_train_h, y_train, X_test_h, y_test,
                                      "HYBRID    — covariates + Burr XII features")


# 4. LIFT TABLE

print("\n" + "=" * 64)
print("LIFT: HYBRID vs BASELINE")
print(f"\n  {'Model':<22} {'ΔAUC':>8} {'ΔPR-AUC':>9} {'ΔLogLoss':>10} {'ΔBrier':>8}")
print("  " + "-"*60)

lift_rows = []
for name in res_base:
    b = res_base[name]
    h = res_hybrid[name]
    dauc = h["AUC"]     - b["AUC"]
    dpr  = h["PR_AUC"]  - b["PR_AUC"]
    dll  = h["LogLoss"] - b["LogLoss"]
    dbr  = h["Brier"]   - b["Brier"]
    sig  = " ✓" if dauc > 0.002 else ""
    print(f"  {name:<22} {dauc:>+8.4f}{sig} {dpr:>+9.4f} "
          f"{dll:>+10.4f} {dbr:>+8.4f}")
    lift_rows.append({
        "Model":        name,
        "Base_AUC":     b["AUC"],    "Hybrid_AUC":   h["AUC"],    "ΔAUC":   round(dauc,4),
        "Base_PR":      b["PR_AUC"], "Hybrid_PR":    h["PR_AUC"], "ΔPR":    round(dpr,4),
        "Base_LL":      b["LogLoss"],"Hybrid_LL":    h["LogLoss"],"ΔLL":    round(dll,4),
        "Base_Brier":   b["Brier"],  "Hybrid_Brier": h["Brier"],  "ΔBrier": round(dbr,4),
    })
print("\n  ✓ = ΔAUC > 0.002 (meaningful lift threshold)")


# 5. FEATURE IMPORTANCE

print("\n" + "=" * 64)
print("FEATURE IMPORTANCE — Hybrid XGBoost (full-data fit)")

xgb_full = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(1-y_train.mean())/y_train.mean(),
    eval_metric="logloss", random_state=42, verbosity=0,
)

rf_full = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    class_weight="balanced", random_state=42, n_jobs=1,
)
rf_full.fit(X_train_h, y_train)

xgb_full.fit(X_train_h, y_train)
imp_df = pd.DataFrame({
    "feature":    HYBRID_FEATURES,
    "importance": xgb_full.feature_importances_,
}).sort_values("importance", ascending=False)
imp_df["type"] = imp_df["feature"].apply(
    lambda f: "Survival" if f in SURVIVAL_FEATURES else "Base ML")

print(f"\n  {'Feature':<25} {'Importance':>10}  Type")
print("  " + "-"*50)
for _, row in imp_df.iterrows():
    marker = " ◀ survival" if row["type"] == "Survival" else ""
    print(f"  {row['feature']:<25} {row['importance']:>10.4f}{marker}")

surv_total = imp_df[imp_df["type"]=="Survival"]["importance"].sum()
base_total = imp_df[imp_df["type"]=="Base ML"]["importance"].sum()
print(f"\n  Base ML features total   : {base_total:.1%}")
print(f"  Survival features total  : {surv_total:.1%}")


# 6. PLOTS

cols_base   = {"Logistic Regression": "#a8c5da",
               "Random Forest":       "#a8ddd4",
               "XGBoost":             "#f4b8bc"}
cols_hybrid = {"Logistic Regression": "#457b9d",
               "Random Forest":       "#2a9d8f",
               "XGBoost":             "#e63946"}

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Hybrid Burr XII + ML vs Baseline — IBM Telco Churn\n"
             "(No tenure in either set  |  Light dashed = Baseline  |  Dark solid = Hybrid)",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(3, 3, hspace=0.42, wspace=0.32)

# ROC
ax1 = fig.add_subplot(gs[0, 0])
for name in res_base:
    fpr, tpr, _ = roc_curve(y_test, preds_base[name])
    ax1.plot(fpr, tpr, color=cols_base[name], lw=1.5, ls="--",
             label=f"{name} base   {res_base[name]['AUC']:.3f}")
for name in res_hybrid:
    fpr, tpr, _ = roc_curve(y_test, preds_hybrid[name])
    ax1.plot(fpr, tpr, color=cols_hybrid[name], lw=2.0,
             label=f"{name} hybrid {res_hybrid[name]['AUC']:.3f}")
ax1.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.4)
ax1.set(xlabel="FPR", ylabel="TPR", title="ROC Curves", xlim=(0,1), ylim=(0,1))
ax1.legend(fontsize=6.5, loc="lower right"); ax1.grid(alpha=0.3)

# PR
ax2 = fig.add_subplot(gs[0, 1])
for name in res_base:
    p, r, _ = precision_recall_curve(y_test, preds_base[name])
    ax2.plot(r, p, color=cols_base[name], lw=1.5, ls="--",
             label=f"{name} base   {res_base[name]['PR_AUC']:.3f}")
for name in res_hybrid:
    p, r, _ = precision_recall_curve(y_test, preds_hybrid[name])
    ax2.plot(r, p, color=cols_hybrid[name], lw=2.0,
             label=f"{name} hybrid {res_hybrid[name]['PR_AUC']:.3f}")
ax2.axhline(y_test.mean(), color="gray", ls=":", lw=1, label=f"baseline={y_test.mean():.2f}")
ax2.set(xlabel="Recall", ylabel="Precision", title="PR Curves", xlim=(0,1), ylim=(0,1))
ax2.legend(fontsize=6.5); ax2.grid(alpha=0.3)

# Calibration
ax3 = fig.add_subplot(gs[0, 2])
for name in res_base:
    fp, mp = calibration_curve(y_test, preds_base[name], n_bins=10)
    ax3.plot(mp, fp, color=cols_base[name], lw=1.5, ls="--", marker="o", ms=3)
for name in res_hybrid:
    fp, mp = calibration_curve(y_test, preds_hybrid[name], n_bins=10)
    ax3.plot(mp, fp, color=cols_hybrid[name], lw=2.0, marker="o", ms=4, label=name)
ax3.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5, label="Perfect")
ax3.set(xlabel="Mean Predicted", ylabel="Fraction Positive",
        title="Calibration", xlim=(0,1), ylim=(0,1))
ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

# AUC bar
model_names = list(res_base.keys())
x = np.arange(len(model_names)); w = 0.35

ax4 = fig.add_subplot(gs[1, 0])
base_aucs   = [res_base[m]["AUC"]   for m in model_names]
hybrid_aucs = [res_hybrid[m]["AUC"] for m in model_names]
ax4.bar(x-w/2, base_aucs,   w, label="Baseline",
        color=list(cols_base.values()),   alpha=0.6)
b2 = ax4.bar(x+w/2, hybrid_aucs, w, label="Hybrid",
             color=list(cols_hybrid.values()), alpha=0.9)
for bar, val in zip(b2, hybrid_aucs):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=8)
ax4.set(ylabel="AUC", title="AUC: Baseline vs Hybrid")
ax4.set_ylim(min(base_aucs+hybrid_aucs)*0.99, min(1.0, max(hybrid_aucs)*1.015))
ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=0.3)

# PR-AUC bar
ax5 = fig.add_subplot(gs[1, 1])
base_prs   = [res_base[m]["PR_AUC"]   for m in model_names]
hybrid_prs = [res_hybrid[m]["PR_AUC"] for m in model_names]
ax5.bar(x-w/2, base_prs,   w, label="Baseline",
        color=list(cols_base.values()),   alpha=0.6)
b3 = ax5.bar(x+w/2, hybrid_prs, w, label="Hybrid",
             color=list(cols_hybrid.values()), alpha=0.9)
for bar, val in zip(b3, hybrid_prs):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax5.set_xticks(x)
ax5.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=8)
ax5.set(ylabel="PR-AUC", title="PR-AUC: Baseline vs Hybrid")
ax5.set_ylim(min(base_prs+hybrid_prs)*0.97, min(1.0, max(hybrid_prs)*1.03))
ax5.legend(fontsize=8); ax5.grid(axis="y", alpha=0.3)

# Feature importance
ax6 = fig.add_subplot(gs[1, 2])
top15 = imp_df.head(15)
bar_colors = ["#e63946" if t == "Survival" else "#adb5bd" for t in top15["type"]]
ax6.barh(top15["feature"][::-1], top15["importance"][::-1],
         color=bar_colors[::-1], alpha=0.9)
ax6.set(xlabel="Importance", title="XGBoost Importance (Hybrid)\nRed = Burr XII features")
ax6.grid(axis="x", alpha=0.3)

# Lift table
ax7 = fig.add_subplot(gs[2, :])
ax7.axis("off")
col_labels = ["Model",
              "Base AUC", "Hybrid AUC", "ΔAUC",
              "Base PR",  "Hybrid PR",  "ΔPR",
              "Base Brier","Hybrid Brier","ΔBrier"]
table_data = []
for r in lift_rows:
    table_data.append([
        r["Model"],
        f"{r['Base_AUC']:.4f}",   f"{r['Hybrid_AUC']:.4f}",  f"{r['ΔAUC']:+.4f}",
        f"{r['Base_PR']:.4f}",    f"{r['Hybrid_PR']:.4f}",    f"{r['ΔPR']:+.4f}",
        f"{r['Base_Brier']:.4f}", f"{r['Hybrid_Brier']:.4f}", f"{r['ΔBrier']:+.4f}",
    ])
tbl = ax7.table(cellText=table_data, colLabels=col_labels,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.scale(1, 1.8)
delta_cols  = [3, 6, 9]
delta_signs = [1, 1, -1]
for ri in range(len(table_data)):
    for ci, sign in zip(delta_cols, delta_signs):
        val = float(table_data[ri][ci])
        if val * sign > 0.0005:
            tbl[ri+1, ci].set_facecolor("#c8f7c5")
        elif val * sign < -0.0005:
            tbl[ri+1, ci].set_facecolor("#f7c8c8")
ax7.set_title("Lift Table: Hybrid vs Baseline  (green = improvement)",
              fontsize=10, fontweight="bold", pad=12)

plt.savefig("ml_comparison_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: ml_comparison_plots.png")


# 7. SAVE

pd.DataFrame(lift_rows).to_csv("ml_hybrid_metrics.csv", index=False)
imp_df.to_csv("ml_hybrid_importance.csv", index=False)

preds_df = pd.DataFrame({"y_true": y_test})
for name in preds_base:
    preds_df[f"base_{name.replace(' ','_')}"]   = preds_base[name]
    preds_df[f"hybrid_{name.replace(' ','_')}"] = preds_hybrid[name]
preds_df.to_csv("ml_hybrid_predictions.csv", index=False)

print("\n" + "=" * 64)
print("SUMMARY")
print(f"\n  Survival features share of XGBoost importance: {surv_total:.1%}")
print(f"\n  {'Model':<22} {'Base AUC':>10} {'Hybrid AUC':>12} {'ΔAUC':>8}")
print("  " + "-"*55)
for r in lift_rows:
    sig = " ✓" if r["ΔAUC"] > 0.002 else ""
    print(f"  {r['Model']:<22} {r['Base_AUC']:>10.4f} "
          f"{r['Hybrid_AUC']:>12.4f} {r['ΔAUC']:>+8.4f}{sig}")
print("""
Files saved:
  ml_hybrid_metrics.csv       full lift table
  ml_hybrid_predictions.csv   OOF probabilities (base + hybrid)
  ml_hybrid_importance.csv    feature importance breakdown
  ml_comparison_plots.png     ROC, PR, calibration, importance, lift table
""")

# ── Feature importance plot ────────────────────────────────────────────────
SURVIVAL_FEATURES = ["burr12_survival", "burr12_hazard", "burr12_churn_prob"]

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
    "burr12_sf": "Burr XII: S(t)",
    "burr12_hf": "Burr XII: h(t)",
    "burr12_churn_prob_12m": "Burr XII: next-period P(churn)",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Feature Importance — Hybrid Models (Burr XII features in purple)",
             fontsize=12, fontweight="bold")

for ax, model, features, title, col in zip(
    axes,
    [rf_full, xgb_full],
    [HYBRID_FEATURES, HYBRID_FEATURES],
    ["Hybrid Random Forest", "Hybrid XGBoost"],
    ["#2a9d8f", "#e63946"],
):
    imp = pd.DataFrame({
        "feature": features,
        "display": [DISPLAY_NAMES.get(f, f) for f in features],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(12)
    colors = ["#7b2d8b" if f in SURVIVAL_FEATURES else col
              for f in imp["feature"]]
    ax.barh(imp["display"][::-1], imp["importance"][::-1],
            color=colors[::-1], alpha=0.88, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Feature importance")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

from matplotlib.patches import Patch
fig.legend(handles=[
    Patch(facecolor="#7b2d8b", alpha=0.88, label="Burr XII survival features"),
    Patch(facecolor="#2a9d8f", alpha=0.88, label="Customer covariates"),
], loc="lower center", ncol=2, fontsize=9, frameon=False,
   bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig("fig_feature_importance_hybrid.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: fig_feature_importance_hybrid.png")