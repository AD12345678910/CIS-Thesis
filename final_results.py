from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Setup

OUTDIR = Path("final_results")
OUTDIR.mkdir(exist_ok=True)

# FINAL RESULTS (from your 80/20 holdout runs)

results = [
    # Probabilistic
    {"group": "Probabilistic", "model": "Weibull", "auc": 0.5000, "pr_auc": 0.2658},
    {"group": "Probabilistic", "model": "Burr XII", "auc": 0.7475, "pr_auc": 0.4902},

    # ML
    {"group": "ML", "model": "Logistic Regression", "auc": 0.8129, "pr_auc": 0.5651},
    {"group": "ML", "model": "Random Forest", "auc": 0.8124, "pr_auc": 0.5729},
    {"group": "ML", "model": "XGBoost", "auc": 0.8056, "pr_auc": 0.5716},

    # Hybrid
    {"group": "Hybrid", "model": "Hybrid Logistic Regression", "auc": 0.8400, "pr_auc": 0.6577},
    {"group": "Hybrid", "model": "Hybrid Random Forest", "auc": 0.8426, "pr_auc": 0.6499},
    {"group": "Hybrid", "model": "Hybrid XGBoost", "auc": 0.8341, "pr_auc": 0.6459},
]

df = pd.DataFrame(results)
df.to_csv(OUTDIR / "final_results_manual.csv", index=False)

print("\nFinal Results:")
print(df)

# Colors by group

colors = {
    "Probabilistic": "#d95f5f",  # red
    "ML": "#4c78a8",             # blue
    "Hybrid": "#8e63b7",         # purple
}

bar_colors = [colors[g] for g in df["group"]]
labels = df["model"].tolist()

# Plot

fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# ROC-AUC

ax = axes[0]
bars = ax.bar(labels, df["auc"], color=bar_colors)

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_ylim(0.4, 0.9)
ax.set_ylabel("ROC-AUC")
ax.set_title("Final Holdout ROC-AUC Comparison", fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.tick_params(axis="x", rotation=25)

for bar, val in zip(bars, df["auc"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.008,
        f"{val:.3f}",
        ha="center",
        fontsize=9
    )

# PR-AUC

ax = axes[1]
bars = ax.bar(labels, df["pr_auc"], color=bar_colors)

ax.set_ylim(0.2, 0.72)

ax.set_ylabel("PR-AUC")
ax.set_title("Final Holdout PR-AUC Comparison", fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.tick_params(axis="x", rotation=25)


for bar, val in zip(bars, df["pr_auc"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.008,
        f"{val:.3f}",
        ha="center",
        fontsize=9
    )

# Group separators (fixed)

for ax in axes:
    ax.axvline(1.5, color="gray", linewidth=0.8, alpha=0.4)  # after Probabilistic
    ax.axvline(4.5, color="gray", linewidth=0.8, alpha=0.4)  # after ML

# Save

plt.tight_layout()
plt.savefig(OUTDIR / "final_auc_pr_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(OUTDIR / "final_results_manual.csv")
print(OUTDIR / "final_auc_pr_comparison.png")