"""
tenure_ablation.py
Compares three feature conditions across all classifiers:
  1. ML with static covariates only (no tenure)
  2. ML with static covariates + raw tenure
  3. Hybrid with static covariates + Burr XII survival features (no raw tenure)

Tests whether hybrid gains are driven by structured survival features
or simply by reintroducing tenure in a different form.

Usage: python tenure_ablation.py  (run burr12_fit.py first)
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

OUTDIR = Path("tenure_vs_hybrid_outputs")
OUTDIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Load feature file
# ------------------------------------------------------------

files = glob.glob("*.csv") + glob.glob("**/*.csv", recursive=True)
feat_file = next((f for f in files if "telco_burr12_features" in f), None)

if feat_file is None:
    raise FileNotFoundError("Could not find telco_burr12_features.csv")

df = pd.read_csv(feat_file)
print(f"Using: {feat_file}")

# ------------------------------------------------------------
# Columns
# ------------------------------------------------------------

TARGET = "event"
TENURE = "tenure"

BASE_FEATURES = [
    "is_male", "is_senior", "partner", "dependents", "phone",
    "internet_fiber", "internet_dsl", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "multiple_lines", "contract_1yr", "contract_2yr",
    "pay_echeck", "paperless", "MonthlyCharges_z",
]

SURVIVAL_FEATURES = [
    "burr12_survival",
    "burr12_hazard",
    "burr12_churn_prob",
]

BASE_NO_TENURE = BASE_FEATURES
BASE_WITH_TENURE = BASE_FEATURES + [TENURE]
HYBRID_NO_TENURE = BASE_FEATURES + SURVIVAL_FEATURES

needed = [TARGET, TENURE] + BASE_FEATURES + SURVIVAL_FEATURES
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[needed].dropna().copy()

X_base_no_tenure = df[BASE_NO_TENURE]
X_base_with_tenure = df[BASE_WITH_TENURE]
X_hybrid_no_tenure = df[HYBRID_NO_TENURE]
y = df[TARGET].astype(int)

# ------------------------------------------------------------
# Same 80/20 split for all models
# ------------------------------------------------------------

idx_train, idx_test = train_test_split(
    df.index,
    test_size=0.20,
    stratify=y,
    random_state=42
)

y_train = y.loc[idx_train]
y_test = y.loc[idx_test]

print("=" * 70)
print("TENURE VS HYBRID EXPERIMENT")
print("=" * 70)
print(f"Train: {len(idx_train)} customers ({y_train.mean():.4f} churn rate)")
print(f"Test : {len(idx_test)} customers ({y_test.mean():.4f} churn rate)")
print()
print("Comparison:")
print("1. ML without tenure")
print("2. ML with raw tenure")
print("3. Hybrid without raw tenure, using Burr XII features")
print()

# ------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------

def make_models(pos_weight):
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
    }

def evaluate(y_true, p):
    return {
        "AUC": roc_auc_score(y_true, p),
        "PR-AUC": average_precision_score(y_true, p),
        "LogLoss": log_loss(y_true, p),
        "Brier": brier_score_loss(y_true, p),
    }

def run_model_set(feature_name, X):
    X_train = X.loc[idx_train]
    X_test = X.loc[idx_test]

    pos_weight = (1 - y_train.mean()) / max(y_train.mean(), 1e-9)
    models = make_models(pos_weight)

    rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, p)

        rows.append({
            "feature_set": feature_name,
            "model": model_name,
            **metrics
        })

    return rows

# ------------------------------------------------------------
# Run experiments
# ------------------------------------------------------------

rows = []
rows += run_model_set("ML no tenure", X_base_no_tenure)
rows += run_model_set("ML with raw tenure", X_base_with_tenure)
rows += run_model_set("Hybrid no raw tenure", X_hybrid_no_tenure)

res = pd.DataFrame(rows)
res.to_csv(OUTDIR / "tenure_vs_hybrid_results.csv", index=False)

print("=" * 70)
print("RESULTS")
print("=" * 70)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ------------------------------------------------------------
# Lift tables
# ------------------------------------------------------------

pivot_auc = res.pivot(index="model", columns="feature_set", values="AUC")
pivot_pr = res.pivot(index="model", columns="feature_set", values="PR-AUC")

lift = pd.DataFrame({
    "AUC: tenure - no tenure": pivot_auc["ML with raw tenure"] - pivot_auc["ML no tenure"],
    "AUC: hybrid - no tenure": pivot_auc["Hybrid no raw tenure"] - pivot_auc["ML no tenure"],
    "AUC: hybrid - tenure": pivot_auc["Hybrid no raw tenure"] - pivot_auc["ML with raw tenure"],
    "PR-AUC: tenure - no tenure": pivot_pr["ML with raw tenure"] - pivot_pr["ML no tenure"],
    "PR-AUC: hybrid - no tenure": pivot_pr["Hybrid no raw tenure"] - pivot_pr["ML no tenure"],
    "PR-AUC: hybrid - tenure": pivot_pr["Hybrid no raw tenure"] - pivot_pr["ML with raw tenure"],
})

lift.to_csv(OUTDIR / "tenure_vs_hybrid_lift.csv")

print()
print("=" * 70)
print("LIFT TABLE")
print("=" * 70)
print(lift.to_string(float_format=lambda x: f"{x:+.4f}"))

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

plot_order = ["ML no tenure", "ML with raw tenure", "Hybrid no raw tenure"]
colors = {
    "ML no tenure": "#9ecae1",
    "ML with raw tenure": "#4c78a8",
    "Hybrid no raw tenure": "#8e63b7",
}

for metric, filename, ylabel, title, ylim in [
    ("AUC", "tenure_vs_hybrid_auc.png", "ROC-AUC",
     "Raw Tenure vs Burr XII Survival Features: ROC-AUC", (0.72, 0.88)),
    ("PR-AUC", "tenure_vs_hybrid_pr_auc.png", "PR-AUC",
     "Raw Tenure vs Burr XII Survival Features: PR-AUC", (0.45, 0.72)),
]:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = range(len(res["model"].unique()))
    width = 0.25
    models = ["Logistic Regression", "Random Forest", "XGBoost"]

    for i, feature_set in enumerate(plot_order):
        vals = [
            res[(res["feature_set"] == feature_set) & (res["model"] == m)][metric].iloc[0]
            for m in models
        ]

        positions = [j + (i - 1) * width for j in x]
        bars = ax.bar(
            positions,
            vals,
            width=width,
            label=feature_set,
            color=colors[feature_set]
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.004,
                f"{val:.3f}",
                ha="center",
                fontsize=8
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(["LogReg", "RF", "XGBoost"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

print()
print("Saved outputs to:", OUTDIR)
print("Saved:")
print(" - tenure_vs_hybrid_results.csv")
print(" - tenure_vs_hybrid_lift.csv")
print(" - tenure_vs_hybrid_auc.png")
print(" - tenure_vs_hybrid_pr_auc.png")

# ------------------------------------------------------------
# Clean combined plot: AUC + PR-AUC
# ------------------------------------------------------------

plot_order = ["ML no tenure", "ML with raw tenure", "Hybrid no raw tenure"]
models = ["Logistic Regression", "Random Forest", "XGBoost"]

colors = {
    "ML no tenure": "#9ecae1",
    "ML with raw tenure": "#4c78a8",
    "Hybrid no raw tenure": "#8e63b7",
}

short_labels = {
    "ML no tenure": "ML, no tenure",
    "ML with raw tenure": "ML + raw tenure",
    "Hybrid no raw tenure": "Hybrid, no raw tenure",
}

fig, axes = plt.subplots(2, 1, figsize=(10.5, 8))

for ax, metric, ylabel, title, ylim in [
    (axes[0], "AUC", "ROC-AUC", "Effect of Raw Tenure vs Burr XII Features: ROC-AUC", (0.78, 0.86)),
    (axes[1], "PR-AUC", "PR-AUC", "Effect of Raw Tenure vs Burr XII Features: PR-AUC", (0.54, 0.68)),
]:
    x = range(len(models))
    width = 0.24

    for i, feature_set in enumerate(plot_order):
        vals = [
            res[
                (res["feature_set"] == feature_set) &
                (res["model"] == model)
            ][metric].iloc[0]
            for model in models
        ]

        positions = [j + (i - 1) * width for j in x]

        bars = ax.bar(
            positions,
            vals,
            width=width,
            color=colors[feature_set],
            label=short_labels[feature_set]
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.0025,
                f"{val:.3f}",
                ha="center",
                fontsize=8
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(["LogReg", "Random Forest", "XGBoost"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(OUTDIR / "tenure_vs_hybrid_combined.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", OUTDIR / "tenure_vs_hybrid_combined.png")