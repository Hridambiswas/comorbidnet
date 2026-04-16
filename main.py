# main.py — ComorbidNet: Correlated Multi-Disease Detection Pipeline
# Author: Hridam Biswas | IEEE Researcher | KIIT University

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    hamming_loss, roc_auc_score
)
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier
from generate_data import generate_cohort

DISEASES    = ["T2D", "HTN", "MetS", "CKD"]
FEATURES    = [
    "age","bmi","glucose_mg_dl","hba1c_pct","systolic_bp","diastolic_bp",
    "triglycerides","hdl_mg_dl","creatinine_mg_dl","egfr_ml_min",
    "waist_circumference_cm","smoking","family_history"
]
RANDOM_SEED = 42


def load_data():
    print("=" * 65)
    print("  ComorbidNet — Correlated Multi-Disease Detection")
    print("=" * 65)
    df = generate_cohort(n=2000)
    print(f"\n[DATA] {len(df)} patients | {len(FEATURES)} biomarkers | {len(DISEASES)} diseases")
    print(f"\n[PREVALENCE]")
    for d in DISEASES:
        print(f"  {d:6s}: {df[d].mean():.1%}")
    return df


def analyse_correlations(df: pd.DataFrame):
    print("\n[CORRELATION ANALYSIS]")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df[FEATURES]),
        columns=FEATURES
    )
    vif_data = pd.DataFrame({
        "Feature": FEATURES,
        "VIF": [variance_inflation_factor(X_scaled.values, i)
                for i in range(len(FEATURES))]
    }).sort_values("VIF", ascending=False)
    print("\n  VIF (>5 = correlated features, hard for naive classifiers):")
    print(vif_data.to_string(index=False))

    label_corr = df[DISEASES].corr()
    print(f"\n  Label correlation (why single-disease models fail):")
    print(label_corr.round(3).to_string())
    return scaler, vif_data, label_corr


def baseline_independent(X_train, X_test, y_train, y_test):
    print("\n[BASELINE] Independent per-disease XGBoost (ignores label correlation)")
    base_preds = np.zeros((len(X_test), len(DISEASES)))
    base_aucs  = []
    for i, disease in enumerate(DISEASES):
        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, use_label_encoder=False,
            eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0
        )
        clf.fit(X_train, y_train[:, i])
        proba = clf.predict_proba(X_test)[:, 1]
        base_preds[:, i] = (proba > 0.5).astype(int)
        auc = roc_auc_score(y_test[:, i], proba)
        base_aucs.append(auc)
        print(f"  {disease}: AUC={auc:.4f}")
    print(f"  Hamming Loss : {hamming_loss(y_test, base_preds):.4f}")
    print(f"  Subset Acc   : {(base_preds == y_test).all(axis=1).mean():.4f}")
    return base_preds, base_aucs


def comorbidnet_chain(X_train, X_test, y_train, y_test):
    print("\n[COMORBIDNET] Classifier Chain XGBoost (exploits label correlation)")
    base_xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, gamma=0.1,
        min_child_weight=3, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0
    )
    chain = ClassifierChain(base_xgb, order=[0, 1, 2, 3], random_state=RANDOM_SEED)
    chain.fit(X_train, y_train)

    chain_proba = chain.predict_proba(X_test)
    chain_preds = (chain_proba > 0.5).astype(int)
    chain_aucs  = []
    for i, disease in enumerate(DISEASES):
        auc = roc_auc_score(y_test[:, i], chain_proba[:, i])
        chain_aucs.append(auc)
        print(f"  {disease}: AUC={auc:.4f}")
    print(f"  Hamming Loss : {hamming_loss(y_test, chain_preds):.4f}")
    print(f"  Subset Acc   : {(chain_preds == y_test).all(axis=1).mean():.4f}")
    return chain, chain_preds, chain_proba, chain_aucs


def explain_with_shap(chain, X_test: pd.DataFrame):
    print("\n[SHAP] Computing interaction-aware feature importances...")
    explainer = shap.TreeExplainer(chain.estimators_[0])
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=FEATURES,
        plot_type="bar",
        show=False,
        title=f"SHAP Feature Importance — {DISEASES[0]} (T2D)"
    )
    plt.tight_layout()
    plt.savefig("outputs/shap_t2d.png", dpi=150)
    plt.close()
    print("  Saved → outputs/shap_t2d.png")


def print_comparison(base_aucs, chain_aucs):
    print("\n" + "=" * 65)
    print("  RESULTS: Baseline vs ComorbidNet")
    print("=" * 65)
    print(f"  {'Disease':<10} {'Baseline AUC':>14} {'ComorbidNet AUC':>16} {'Gain':>8}")
    print(f"  {'-'*52}")
    for i, d in enumerate(DISEASES):
        gain = chain_aucs[i] - base_aucs[i]
        arrow = "▲" if gain > 0 else "▼"
        print(f"  {d:<10} {base_aucs[i]:>14.4f} {chain_aucs[i]:>16.4f} {arrow}{abs(gain):>6.4f}")
    print(f"\n  Mean Baseline AUC   : {np.mean(base_aucs):.4f}")
    print(f"  Mean ComorbidNet AUC: {np.mean(chain_aucs):.4f}")
    print("=" * 65)


def main():
    import os
    os.makedirs("outputs", exist_ok=True)

    df      = load_data()
    scaler, _, _ = analyse_correlations(df)

    X = pd.DataFrame(scaler.transform(df[FEATURES]), columns=FEATURES)
    y = df[DISEASES].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y[:, 0]
    )

    _, base_aucs             = baseline_independent(X_train, X_test, y_train, y_test)
    chain, _, _, chain_aucs  = comorbidnet_chain(X_train, X_test, y_train, y_test)

    explain_with_shap(chain, X_test)
    print_comparison(base_aucs, chain_aucs)


if __name__ == "__main__":
    main()
