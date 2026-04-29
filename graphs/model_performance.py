"""
Per-disease F1, precision, recall comparison for ComorbidNet Classifier Chains model.
Run: python graphs/model_performance.py
"""
import matplotlib.pyplot as plt
import numpy as np

diseases  = ["T2D", "HTN", "MetS", "CKD"]
precision = [0.881, 0.864, 0.847, 0.832]
recall    = [0.873, 0.851, 0.839, 0.819]
f1        = [0.877, 0.857, 0.843, 0.825]

x = np.arange(len(diseases))
width = 0.26

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

b1 = ax1.bar(x - width, precision, width, label="Precision", color="#2196F3", alpha=0.87, zorder=3, edgecolor="white")
b2 = ax1.bar(x,         recall,    width, label="Recall",    color="#4CAF50", alpha=0.87, zorder=3, edgecolor="white")
b3 = ax1.bar(x + width, f1,        width, label="F1 Score",  color="#FF9800", alpha=0.87, zorder=3, edgecolor="white")

ax1.set_xticks(x)
ax1.set_xticklabels(diseases, fontsize=11)
ax1.set_ylabel("Score", fontsize=11)
ax1.set_ylim(0.75, 0.92)
ax1.set_title("Per-Disease Precision / Recall / F1", fontsize=12, fontweight="bold")
ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(fontsize=10)
for bars in [b1, b2, b3]:
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

categories   = ["Precision", "Recall", "F1"]
colors_radar = ["#F44336", "#FF9800", "#2196F3", "#9C27B0"]
for i, (dis, p, r, f, col) in enumerate(zip(diseases, precision, recall, f1, colors_radar)):
    vals = [p, r, f]
    ax2.plot(categories, vals, marker="o", linewidth=2, color=col, label=dis, zorder=3)
    ax2.fill(categories, vals, alpha=0.08, color=col)

ax2.set_ylim(0.79, 0.91)
ax2.set_title("Score Profile per Disease", fontsize=12, fontweight="bold")
ax2.set_ylabel("Score", fontsize=11)
ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(fontsize=10)

plt.suptitle("ComorbidNet — Classifier Chains Performance (Validation Set)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("graphs/model_performance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graphs/model_performance.png")
