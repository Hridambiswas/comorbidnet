# generate_data.py — Synthetic patient cohort generator
# Simulates realistic biomarker correlations across 4 comorbid conditions

import numpy as np
import pandas as pd

np.random.seed(42)


def generate_cohort(n: int = 2000) -> pd.DataFrame:
    """
    Generates a synthetic patient cohort with realistic inter-disease correlations.
    Diseases: Type 2 Diabetes (T2D), Hypertension (HTN),
              Metabolic Syndrome (MetS), Chronic Kidney Disease (CKD)
    """
    insulin_resistance = np.random.normal(0, 1, n)
    vascular_stress    = np.random.normal(0, 1, n)
    obesity_factor     = np.random.normal(0, 1, n)

    age         = np.clip(np.random.normal(52, 14, n), 18, 90)
    bmi         = np.clip(28 + 3*obesity_factor + np.random.normal(0, 2, n), 15, 55)
    glucose     = np.clip(100 + 18*insulin_resistance + 5*obesity_factor + np.random.normal(0, 8, n), 60, 400)
    hba1c       = np.clip(5.5 + 0.6*insulin_resistance + 0.2*obesity_factor + np.random.normal(0, 0.4, n), 4.0, 14.0)
    systolic_bp = np.clip(120 + 12*vascular_stress + 4*obesity_factor + np.random.normal(0, 8, n), 80, 220)
    diastolic_bp= np.clip(80  + 8*vascular_stress  + 2*obesity_factor + np.random.normal(0, 5, n), 50, 130)
    triglycerides=np.clip(150 + 40*insulin_resistance + 20*obesity_factor + np.random.normal(0, 30, n), 50, 800)
    hdl         = np.clip(50  - 8*insulin_resistance  - 4*obesity_factor + np.random.normal(0, 8, n), 15, 100)
    creatinine  = np.clip(1.0 + 0.4*vascular_stress  + 0.1*obesity_factor + np.random.normal(0, 0.2, n), 0.4, 8.0)
    egfr        = np.clip(90  - 15*vascular_stress   - 5*obesity_factor  + np.random.normal(0, 10, n), 5, 120)
    waist_circ  = np.clip(90  + 10*obesity_factor    + 3*insulin_resistance + np.random.normal(0, 6, n), 55, 160)
    smoking     = (np.random.rand(n) < 0.22).astype(int)
    family_hist = (np.random.rand(n) < 0.35).astype(int)

    p_t2d  = 1 / (1 + np.exp(-(0.08*(glucose-126) + 1.5*(hba1c-6.5) + 0.3*obesity_factor + 0.02*(age-45))))
    p_htn  = 1 / (1 + np.exp(-(0.06*(systolic_bp-140) + 0.4*vascular_stress + 0.01*(age-50) + 0.3*smoking)))
    p_mets = 1 / (1 + np.exp(-(0.5*(bmi-30) + 0.4*insulin_resistance + 0.03*(triglycerides-150) - 0.05*(hdl-40))))
    p_ckd  = 1 / (1 + np.exp(-(2*(creatinine-1.3) - 0.05*(egfr-60) + 0.3*vascular_stress + 0.015*(age-55))))

    t2d  = (np.random.rand(n) < p_t2d).astype(int)
    htn  = (np.random.rand(n) < p_htn).astype(int)
    mets = (np.random.rand(n) < p_mets).astype(int)
    ckd  = (np.random.rand(n) < p_ckd).astype(int)

    df = pd.DataFrame({
        "age": age.round(1),
        "bmi": bmi.round(1),
        "glucose_mg_dl": glucose.round(1),
        "hba1c_pct": hba1c.round(2),
        "systolic_bp": systolic_bp.round(1),
        "diastolic_bp": diastolic_bp.round(1),
        "triglycerides": triglycerides.round(1),
        "hdl_mg_dl": hdl.round(1),
        "creatinine_mg_dl": creatinine.round(2),
        "egfr_ml_min": egfr.round(1),
        "waist_circumference_cm": waist_circ.round(1),
        "smoking": smoking,
        "family_history": family_hist,
        "T2D": t2d,
        "HTN": htn,
        "MetS": mets,
        "CKD": ckd,
    })
    return df


if __name__ == "__main__":
    df = generate_cohort(2000)
    df.to_csv("patient_cohort.csv", index=False)
    print(f"Generated {len(df)} patients")
    print(f"\nDisease prevalence:")
    for d in ["T2D","HTN","MetS","CKD"]:
        print(f"  {d}: {df[d].mean():.1%}")
    print(f"\nLabel correlation matrix:")
    print(df[["T2D","HTN","MetS","CKD"]].corr().round(3))
