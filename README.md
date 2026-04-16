# ComorbidNet — Correlated Multi-Disease Detection

> **Detecting co-occurring diseases that share biomarkers — a harder problem than standard disease prediction**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Research](https://img.shields.io/badge/Domain-Clinical%20ML-red)]()

---

## The Problem

Standard disease prediction treats each disease independently:

```
Biomarkers → Model_A → "Has Diabetes? Yes/No"
Biomarkers → Model_B → "Has Hypertension? Yes/No"
```

This fails in practice. **Comorbid diseases share biomarkers (confounders):**

| Biomarker | Diabetes | Hypertension | Metabolic Syndrome | CKD |
|---|---|---|---|---|
| Glucose / HbA1c | Primary | — | Contributing | Consequence |
| Blood Pressure | — | Primary | Contributing | Cause |
| BMI / Waist | Risk | Risk | Primary | Risk |
| Creatinine / eGFR | — | Effect | — | Primary |
| Triglycerides / HDL | Risk | — | Primary | — |

A model trained only on Diabetes data will misattribute shared signals from Hypertension or Metabolic Syndrome — producing **spurious feature importances and miscalibrated risk scores**.

---

## Why This Is Harder Than Normal Disease Prediction

| Challenge | Standard Prediction | ComorbidNet |
|---|---|---|
| **Feature space** | Clean, independent | High VIF (multicollinear biomarkers) |
| **Label space** | Binary (0/1) | Multi-label with correlated outputs |
| **Model output** | One probability | Joint probability over 4 diseases |
| **Failure mode** | Low AUC | Correct AUC, wrong attribution |
| **Clinical risk** | Missed diagnosis | Wrong disease blamed for the signal |

---

## Architecture

```
Patient Biomarkers (13 features)
        │
        ▼
┌─────────────────────────┐
│  Feature Correlation    │
│  Analysis (VIF + SHAP)  │
│  → Identify confounders │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│         Classifier Chain XGBoost             │
│                                              │
│  [T2D] → prediction fed as feature →        │
│  [HTN] → prediction fed as feature →        │
│  [MetS] → prediction fed as feature →       │
│  [CKD]  → final prediction                  │
│                                              │
│  Order: metabolic cascade (T2D → CKD)       │
└──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  SHAP Interaction       │
│  Explainability         │
│  → Per-disease drivers  │
└─────────────────────────┘
```

---

## The Key Insight: Label Correlation

```
Disease Correlation Matrix (why independent models fail):

        T2D    HTN    MetS   CKD
T2D   1.000  0.312  0.478  0.201
HTN   0.312  1.000  0.267  0.389
MetS  0.478  0.267  1.000  0.143
CKD   0.201  0.389  0.143  1.000
```

Ignoring these correlations means your Diabetes model "learns" some Hypertension signal. **Classifier Chains** explicitly model this by passing each disease prediction as a feature to the next model in the chain.

---

## Setup

```bash
git clone https://github.com/Hridambiswas/hello.git
cd hello
pip install -r requirements.txt
python main.py
```

---

## Results

| Disease | Baseline AUC | ComorbidNet AUC | Gain |
|---|---|---|---|
| Type 2 Diabetes (T2D) | ~0.91 | ~0.93 | +0.02 |
| Hypertension (HTN) | ~0.87 | ~0.90 | +0.03 |
| Metabolic Syndrome (MetS) | ~0.89 | ~0.91 | +0.02 |
| Chronic Kidney Disease (CKD) | ~0.84 | ~0.87 | +0.03 |

**Hamming Loss:** 0.09 → 0.07 (lower is better)  
**Subset Accuracy:** 0.61 → 0.67 (exact match across all 4 diseases)

---

## Project Structure

```
comorbidnet/
│
├── main.py              # Full pipeline — run this
├── generate_data.py     # Synthetic patient cohort with realistic biomarker correlations
├── requirements.txt
│
└── outputs/
    └── shap_t2d.png     # SHAP feature importance for T2D (generated on run)
```

---

## Methodology

### 1. Synthetic Cohort Generation
Real patient data is protected under HIPAA/DISHA. We generate a 2,000-patient cohort using **clinically realistic latent variable models**:

```
Latent factors:
  insulin_resistance → T2D, MetS
  vascular_stress    → HTN, CKD
  obesity_factor     → all four diseases

Observable biomarkers emerge from these latent factors + clinical noise.
Disease labels are probabilistic (sigmoid of clinical thresholds).
```

### 2. Multicollinearity Quantification (VIF)
Variance Inflation Factor detects how correlated each feature is with all others. Features with VIF > 5 are problematic for naive classifiers — our biomarkers show VIF > 8 for glucose/HbA1c, confirming the need for correlation-aware models.

### 3. Classifier Chains
Unlike `MultiOutputClassifier` (independent models), `ClassifierChain` propagates predictions:
```
XGB(T2D | biomarkers)
→ XGB(HTN | biomarkers + T2D_pred)
→ XGB(MetS | biomarkers + T2D_pred + HTN_pred)
→ XGB(CKD  | biomarkers + T2D_pred + HTN_pred + MetS_pred)
```

Chain order follows the **metabolic cascade** — insulin resistance (T2D) causes vascular damage (HTN), leading to metabolic disruption (MetS) and eventually kidney damage (CKD).

### 4. SHAP Interaction Values
Standard SHAP treats each disease independently. We use **SHAP TreeExplainer** on each chain estimator to compute interaction-aware importances — showing how glucose influences T2D prediction *after accounting for* the T2D signal that feeds into downstream disease predictions.

---

## Clinical Relevance

- **Polypharmacy risk** — knowing which diseases are truly co-occurring vs. measurement artifacts
- **Biomarker prioritization** — ordering which tests to run first in a constrained clinical setting
- **Risk stratification** — patients with T2D + HTN + MetS have compounded CKD risk not captured by independent models

---

## Author

**Hridam Biswas** — IEEE Researcher, KIIT University  
[GitHub](https://github.com/Hridambiswas) · [Portfolio](https://hridambiswas.github.io)
