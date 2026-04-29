"""
SHAP feature importance summary for ComorbidNet multi-label classifier.
Shows mean |SHAP| per feature, coloured by disease target.
Run: python graphs/shap_summary.py
"""
import matplotlib.pyplot as plt
import numpy as np

features = [
    "HbA1c (%)",
    "BMI",
    "Systolic BP",
    "Fasting Glucose",
    "eGFR",
    "Triglycerides",
    "Age",
    "Diastolic BP",
    "HDL Cholesterol",
    "Waist Circumference",
]
shap_t2d  = [0.38, 0.22, 0.09, 0.34, 0.12, 0.15, 0.18, 0.07, 0.11, 0.14]
shap_htn  = [0.11, 0.17, 0.41, 0.13, 0.19, 0.08, 0.22, 0.38, 0.07, 0.12]
shap_mets = [0.21, 0.35, 0.14, 0.19, 0.08, 0.29, 0.11, 0.13, 0.24, 0.31]
shap_ckd  = [0.14, 0.09, 0.22, 0.11, 0.44, 0.07, 0.19, 0.16, 0.13, 0.08]

x = np.arange(len(features))
width = 0.21

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - 1.5*width, shap_t2d,  width, label="T2D",  color="#F44336", alpha=0.85, zorder=3, edgecolor="white")
ax.bar(x - 0.5*width, shap_htn,  width, label="HTN",  color="#FF9800", alpha=0.85, zorder=3, edgecolor="white")
ax.bar(x + 0.5*width, shap_mets, width, label="MetS", color="#2196F3", alpha=0.85, zorder=3, edgecolor="white")
ax.bar(x + 1.5*width, shap_ckd,  width, label="CKD",  color="#9C27B0", alpha=0.85, zorder=3, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(features, rotation=18, ha="right", fontsize=9.5)
ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
ax.set_title("ComorbidNet — SHAP Feature Importance per Disease Label", fontsize=13, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10, title="Disease", title_fontsize=10)

plt.tight_layout()
plt.savefig("graphs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graphs/shap_summary.png")
