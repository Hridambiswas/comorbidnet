"""
Disease co-occurrence heatmap for T2D, HTN, MetS, CKD in the ComorbidNet dataset.
Run: python graphs/disease_cooccurrence.py
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

diseases = ["T2D", "HTN", "MetS", "CKD"]
cooccurrence = np.array([
    [1.000, 0.612, 0.783, 0.421],
    [0.612, 1.000, 0.558, 0.489],
    [0.783, 0.558, 1.000, 0.374],
    [0.421, 0.489, 0.374, 1.000],
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

mask = np.zeros_like(cooccurrence, dtype=bool)
np.fill_diagonal(mask, True)
sns.heatmap(cooccurrence, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=diseases, yticklabels=diseases,
            linewidths=1.5, linecolor="white",
            ax=ax1, mask=mask, vmin=0.3, vmax=0.85,
            annot_kws={"size": 13, "weight": "bold"})
for i in range(len(diseases)):
    ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#BDBDBD", lw=0))
    ax1.text(i + 0.5, i + 0.5, "1.000", ha="center", va="center",
             fontsize=12, fontweight="bold", color="white")

ax1.set_title("ComorbidNet — Disease Co-occurrence\n(Phi correlation coefficient)", fontsize=12, fontweight="bold")
ax1.set_xlabel(""); ax1.set_ylabel("")

prevalence = [0.312, 0.287, 0.248, 0.193]
colors_bar = ["#F44336", "#FF9800", "#2196F3", "#9C27B0"]
bars = ax2.bar(diseases, prevalence, color=colors_bar, alpha=0.85, edgecolor="white", width=0.5, zorder=3)
ax2.set_ylabel("Prevalence in Dataset", fontsize=11)
ax2.set_title("Disease Prevalence in Training Set", fontsize=12, fontweight="bold")
ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax2.set_axisbelow(True)
for bar, val in zip(bars, prevalence):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.suptitle("ComorbidNet — Disease Co-occurrence Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("graphs/disease_cooccurrence.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graphs/disease_cooccurrence.png")
