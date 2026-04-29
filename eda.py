import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = Path("data_setup_outputs")
OUTDIR.mkdir(exist_ok=True)

# Find dataset
candidates = glob.glob("*.csv") + glob.glob("**/*.csv", recursive=True)

preferred = [
    f for f in candidates
    if "burr12_features" in f or "telco" in f.lower()
]

if not preferred:
    raise FileNotFoundError("Could not find Telco CSV. Put this script in project root or update FILE path.")

FILE = preferred[0]
print(f"Using dataset: {FILE}")

df = pd.read_csv(FILE)

# Basic assumptions
event_col = "event"
tenure_col = "tenure"

if event_col not in df.columns:
    raise ValueError("Expected column 'event' for churn label.")

if tenure_col not in df.columns:
    raise ValueError("Expected column 'tenure' for customer tenure.")

# Clean
df = df.copy()
df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
df[tenure_col] = pd.to_numeric(df[tenure_col], errors="coerce")
df = df.dropna(subset=[event_col, tenure_col])

n = len(df)
churn_rate = df[event_col].mean()
non_churn_rate = 1 - churn_rate

# ------------------------------------------------------------
# 1. Churn class balance
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))

counts = df[event_col].value_counts().sort_index()
labels = ["Retained", "Churned"]
values = [counts.get(0, 0), counts.get(1, 0)]

bars = ax.bar(labels, values)
ax.set_title("Class Balance: Retained vs Churned", fontweight="bold")
ax.set_ylabel("Number of customers")
ax.grid(axis="y", alpha=0.25)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val,
        f"{val:,}\n({val / n:.1%})",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.savefig(OUTDIR / "01_class_balance.png", dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# 2. Tenure distribution by churn status
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

df[df[event_col] == 0][tenure_col].plot(
    kind="hist", bins=30, alpha=0.55, ax=ax, label="Retained"
)
df[df[event_col] == 1][tenure_col].plot(
    kind="hist", bins=30, alpha=0.55, ax=ax, label="Churned"
)

ax.set_title("Tenure Distribution by Churn Status", fontweight="bold")
ax.set_xlabel("Tenure")
ax.set_ylabel("Number of customers")
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(OUTDIR / "02_tenure_distribution_by_churn.png", dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# 3. Churn rate by tenure bucket
# ------------------------------------------------------------
df["tenure_bucket"] = pd.qcut(
    df[tenure_col],
    q=6,
    duplicates="drop"
)

bucket = (
    df.groupby("tenure_bucket", observed=True)
      .agg(
          customers=(event_col, "size"),
          churn_rate=(event_col, "mean"),
          avg_tenure=(tenure_col, "mean")
      )
      .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 5))

x_labels = [str(x) for x in bucket["tenure_bucket"]]
bars = ax.bar(x_labels, bucket["churn_rate"])

ax.set_title("Observed Churn Rate by Tenure Bucket", fontweight="bold")
ax.set_xlabel("Tenure bucket")
ax.set_ylabel("Churn rate")
ax.set_ylim(0, max(bucket["churn_rate"]) + 0.08)
ax.tick_params(axis="x", rotation=35)
ax.grid(axis="y", alpha=0.25)

for bar, val in zip(bars, bucket["churn_rate"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01,
        f"{val:.1%}",
        ha="center",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig(OUTDIR / "03_churn_rate_by_tenure_bucket.png", dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# 4. Key covariates by churn status
# ------------------------------------------------------------
possible_features = [
    "MonthlyCharges_z",
    "paperless",
    "pay_echeck",
    "contract_1yr",
    "contract_2yr",
    "internet_fiber",
    "online_security",
    "tech_support",
]

features = [c for c in possible_features if c in df.columns]

if features:
    means = df.groupby(event_col)[features].mean().T
    means.columns = ["Retained", "Churned"]

    ax = means.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Average Feature Values by Churn Status", fontweight="bold")
    ax.set_ylabel("Mean value")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTDIR / "04_feature_means_by_churn.png", dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------
# 5. Dataset summary CSV
# ------------------------------------------------------------
summary = {
    "dataset_file": FILE,
    "num_customers": n,
    "num_churned": int(df[event_col].sum()),
    "num_retained": int((1 - df[event_col]).sum()),
    "churn_rate": churn_rate,
    "retention_rate": non_churn_rate,
    "tenure_min": df[tenure_col].min(),
    "tenure_median": df[tenure_col].median(),
    "tenure_mean": df[tenure_col].mean(),
    "tenure_max": df[tenure_col].max(),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUTDIR / "dataset_summary.csv", index=False)

bucket.to_csv(OUTDIR / "tenure_bucket_summary.csv", index=False)

# ------------------------------------------------------------
# 6. Thesis-ready text summary
# ------------------------------------------------------------
text = f"""
DATA SETUP SUMMARY

Dataset used:
- {FILE}

Sample construction:
- Final usable sample: {n:,} customers.
- Outcome variable: event = 1 if the customer churned, 0 otherwise.
- Churned customers: {int(df[event_col].sum()):,} ({churn_rate:.1%}).
- Retained customers: {int((1 - df[event_col]).sum()):,} ({non_churn_rate:.1%}).

Key modeling variables:
- Tenure is used as the time-to-event variable for the probabilistic survival model.
- The churn indicator is used as the event/censoring variable.
- Static customer covariates are used for the machine learning models.
- Hybrid models combine the original ML covariates with survival-derived features.

Important data characteristics:
- The dataset is moderately imbalanced, with churners representing {churn_rate:.1%} of customers.
- This motivates reporting both ROC-AUC and PR-AUC, since PR-AUC is more sensitive to performance on the churn class.
- Tenure varies substantially across customers, making the dataset appropriate for survival-style modeling.
- Differences in observed churn rates across tenure buckets provide evidence of duration dependence.
- Static covariates such as contract type, payment method, internet service, and monthly charges provide predictive information for ML models.

Figures generated:
1. 01_class_balance.png
2. 02_tenure_distribution_by_churn.png
3. 03_churn_rate_by_tenure_bucket.png
4. 04_feature_means_by_churn.png
"""
# ------------------------------------------------------------
# 5. Contract type distribution
# ------------------------------------------------------------
contract_cols = [c for c in ["contract_1yr", "contract_2yr"] if c in df.columns]

if all(c in df.columns for c in ["contract_1yr", "contract_2yr"]):
    contract_counts = {
        "Month-to-month": ((df["contract_1yr"] == 0) & (df["contract_2yr"] == 0)).sum(),
        "One year": (df["contract_1yr"] == 1).sum(),
        "Two year": (df["contract_2yr"] == 1).sum(),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(contract_counts.keys(), contract_counts.values(), color="#4c78a8")
    ax.set_title("Contract Type Distribution", fontweight="bold")
    ax.set_ylabel("Number of customers")
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, contract_counts.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(contract_counts.values()) * 0.02,
            f"{val:,}\n({val / n:.1%})",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(OUTDIR / "05_contract_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# 6. Internet service distribution
# ------------------------------------------------------------
if all(c in df.columns for c in ["internet_fiber", "internet_dsl"]):
    internet_counts = {
        "Fiber optic": (df["internet_fiber"] == 1).sum(),
        "DSL": (df["internet_dsl"] == 1).sum(),
        "No internet": ((df["internet_fiber"] == 0) & (df["internet_dsl"] == 0)).sum(),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(internet_counts.keys(), internet_counts.values(), color="#72b7b2")
    ax.set_title("Internet Service Distribution", fontweight="bold")
    ax.set_ylabel("Number of customers")
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, internet_counts.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(internet_counts.values()) * 0.02,
            f"{val:,}\n({val / n:.1%})",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(OUTDIR / "06_internet_service_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# 7. Monthly charges distribution
# ------------------------------------------------------------
if "MonthlyCharges" in df.columns:
    charge_col = "MonthlyCharges"
elif "MonthlyCharges_z" in df.columns:
    charge_col = "MonthlyCharges_z"
else:
    charge_col = None

if charge_col is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[charge_col].dropna(), bins=30, color="#4c78a8", alpha=0.85)
    ax.set_title("Distribution of Monthly Charges", fontweight="bold")
    ax.set_xlabel(charge_col)
    ax.set_ylabel("Number of customers")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTDIR / "07_monthly_charges_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
(OUTDIR / "data_setup_summary.txt").write_text(text)

print("\nSaved outputs to:", OUTDIR)
print(text)