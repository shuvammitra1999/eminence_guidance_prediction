"""
Generate 81 synthetic patients with REALISTIC noise levels.

Key principle: the synthetic data must match the real data's weak correlations
between radiographic features and AEI (r=0.1-0.4), NOT produce textbook-perfect
relationships (r=0.9+). The unexplained variance (~70%) is real clinical biology
(TMJ disc, functional loading, genetics) that radiographs cannot capture.

Method:
1. Generate inter-correlated features using the real data's correlation matrix
2. Compute Mean_AEI from a weak-signal formula + large Gaussian noise (σ=2.5)
3. Generate Right/Left AEI with realistic asymmetry
"""

import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh

SEED = 42

# ---------------------------------------------------------------------------
# Feature generation parameters — taken directly from the 19 real patients
# ---------------------------------------------------------------------------

# Variable order for multivariate normal (features only, NOT AEI):
# 0: SN_GoGn, 1: Occ_Plane, 2: Ramus_Ht, 3: Cond_Ht, 4: Bigonial, 5: ANB
FEAT_NAMES = ["SN_GoGn", "Occ_Plane", "Ramus_Ht", "Cond_Ht", "Bigonial", "ANB"]

# Real data means and SDs
FEAT_MEANS = {
    "F": np.array([25.5, 12.6, 61.5, 12.0, 183.0, 2.1]),
    "M": np.array([23.5, 11.0, 64.5, 13.5, 193.0, 2.1]),
}
FEAT_SDS = np.array([5.6, 3.1, 4.2, 1.9, 12.0, 1.25])

# Inter-feature correlation matrix from real data (6x6, excludes Sex since
# Sex is generated separately as binary, and we apply sex-specific means)
# Order: SN_GoGn, Occ_Plane, Ramus_Ht, Cond_Ht, Bigonial, ANB
# fmt: off
FEAT_CORR = np.array([
    [ 1.000,  0.244,  0.322,  0.275,  0.446, -0.140],  # SN_GoGn
    [ 0.244,  1.000,  0.012, -0.311, -0.014, -0.386],  # Occ_Plane
    [ 0.322,  0.012,  1.000,  0.256,  0.233, -0.129],  # Ramus_Ht
    [ 0.275, -0.311,  0.256,  1.000,  0.070, -0.235],  # Cond_Ht
    [ 0.446, -0.014,  0.233,  0.070,  1.000, -0.156],  # Bigonial
    [-0.140, -0.386, -0.129, -0.235, -0.156,  1.000],  # ANB
])
# fmt: on

# ---------------------------------------------------------------------------
# AEI generation: weak signal + noise (matching literature R²≈0.25-0.30)
# ---------------------------------------------------------------------------

# Coefficients for the linear signal component
# These produce individual r ≈ 0.15-0.35 when combined with σ_noise=2.5
AEI_INTERCEPT = 22.0
AEI_COEFS = {
    "Sex":      2.5,    # males ~2.5 deg steeper
    "SN_GoGn": -0.10,   # high angle → flatter eminence
    "Occ_Plane":-0.15,  # steeper OPA → flatter eminence
    "Ramus_Ht": 0.20,   # taller ramus → steeper eminence
    "Cond_Ht":  0.35,   # taller condyle → steeper eminence
    "Bigonial": -0.08,  # wider mandible → flatter eminence
    "ANB":      0.50,   # larger ANB → steeper eminence
}
AEI_NOISE_SD = 2.5  # the ~70% unexplained variance

# Clinical ranges for validation
CLINICAL_RANGES = {
    "SN_GoGn":   (14, 40),
    "Occ_Plane": (5, 22),
    "Ramus_Ht":  (48, 75),
    "Cond_Ht":   (7, 17),
    "Bigonial":  (150, 215),
    "ANB":       (-1.5, 5.5),
    "Mean_AEI":  (10, 35),
    "Age":       (14, 30),
}


def _ensure_pd(corr):
    """Nudge correlation matrix to be positive-definite if needed."""
    min_eig = eigvalsh(corr).min()
    if min_eig < 1e-8:
        corr = corr + np.eye(corr.shape[0]) * (abs(min_eig) + 1e-6)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return corr


def generate_synthetic(n=81, seed=SEED):
    rng = np.random.default_rng(seed)

    corr = _ensure_pd(FEAT_CORR.copy())
    cov = np.outer(FEAT_SDS, FEAT_SDS) * corr

    # Assign sex: ~30% male
    sex = rng.binomial(1, 0.30, size=n)

    # Assign age: uniform 14-30 with slight skew toward younger (matching real data)
    age_pool = list(range(14, 31))
    age_weights = [3, 3, 3, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1]
    age_weights = np.array(age_weights, dtype=float)
    age_weights /= age_weights.sum()

    rows = []
    attempts = 0

    while len(rows) < n and attempts < n * 30:
        attempts += 1
        idx = len(rows)
        s = sex[idx] if idx < n else rng.binomial(1, 0.30)

        # Sample features from multivariate normal with sex-specific means
        means = FEAT_MEANS["M"] if s == 1 else FEAT_MEANS["F"]
        sample = rng.multivariate_normal(means, cov)

        vals = {name: sample[i] for i, name in enumerate(FEAT_NAMES)}

        # Check clinical ranges
        valid = True
        for name, (lo, hi) in CLINICAL_RANGES.items():
            if name in vals and (vals[name] < lo or vals[name] > hi):
                valid = False
                break
        if not valid:
            continue

        # Generate AEI: weak signal + noise
        signal = AEI_INTERCEPT
        signal += AEI_COEFS["Sex"] * s
        signal += AEI_COEFS["SN_GoGn"] * (vals["SN_GoGn"] - 25.5)
        signal += AEI_COEFS["Occ_Plane"] * (vals["Occ_Plane"] - 12.6)
        signal += AEI_COEFS["Ramus_Ht"] * (vals["Ramus_Ht"] - 61.5)
        signal += AEI_COEFS["Cond_Ht"] * (vals["Cond_Ht"] - 12.0)
        signal += AEI_COEFS["Bigonial"] * (vals["Bigonial"] - 183.0)
        signal += AEI_COEFS["ANB"] * (vals["ANB"] - 2.1)

        noise = rng.normal(0, AEI_NOISE_SD)
        mean_aei = signal + noise

        if mean_aei < 10 or mean_aei > 35:
            continue

        # Right/Left AEI with realistic asymmetry (SD ~1.5 deg)
        asym = rng.normal(0, 1.5)
        asym = np.clip(asym, -5, 5)
        right_aei = mean_aei + asym / 2
        left_aei = mean_aei - asym / 2

        if right_aei < 8 or right_aei > 36 or left_aei < 8 or left_aei > 36:
            continue

        # Age
        age = rng.choice(age_pool, p=age_weights)

        rows.append({
            "Patient_ID": idx + 1,
            "Age": int(age),
            "Sex": s,
            "SN_GoGn": round(vals["SN_GoGn"], 1),
            "Occ_Plane": round(vals["Occ_Plane"], 1),
            "Ramus_Ht": round(vals["Ramus_Ht"], 1),
            "Cond_Ht": round(vals["Cond_Ht"], 1),
            "Bigonial": round(vals["Bigonial"], 1),
            "ANB": round(vals["ANB"], 1),
            "Right_AEI": round(right_aei, 1),
            "Left_AEI": round(left_aei, 1),
            "Mean_AEI": round(mean_aei, 1),
        })

    if len(rows) < n:
        raise RuntimeError(f"Only generated {len(rows)}/{n} after {attempts} attempts")

    return pd.DataFrame(rows)


def validate_synthetic(df):
    """Validate that synthetic data matches real-data characteristics."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    print(f"Generated {len(df)} synthetic patients")
    print(f"Sex distribution: {df['Sex'].value_counts().to_dict()} (0=F, 1=M)")

    # Correlations with AEI (should be r=0.1-0.4, NOT 0.9+)
    print(f"\nCorrelations with Mean_AEI (target: r=0.1-0.4):")
    for col in ["Sex", "SN_GoGn", "Occ_Plane", "Ramus_Ht", "Cond_Ht", "Bigonial", "ANB"]:
        r = df[col].corr(df["Mean_AEI"])
        status = "OK" if 0.05 < abs(r) < 0.50 else "TOO HIGH" if abs(r) >= 0.50 else "very weak"
        print(f"  {col:12s}  r = {r:+.3f}  ({status})")

    # Combined R² (should be 0.25-0.35)
    feats = ["Sex", "SN_GoGn", "Occ_Plane", "Ramus_Ht", "Cond_Ht", "Bigonial", "ANB"]
    X = df[feats].values
    y = df["Mean_AEI"].values
    lr = LinearRegression().fit(X, y)
    r2 = r2_score(y, lr.predict(X))
    print(f"\nCombined R² (OLS, 7 features): {r2:.3f}  (target: 0.25-0.35)")

    # Without ANB
    X6 = df[feats[:6]].values
    lr6 = LinearRegression().fit(X6, y)
    r2_6 = r2_score(y, lr6.predict(X6))
    print(f"Combined R² (OLS, 6 features, no ANB): {r2_6:.3f}")

    # Sex dimorphism
    m = df[df["Sex"] == 1]["Mean_AEI"]
    f = df[df["Sex"] == 0]["Mean_AEI"]
    print(f"\nAEI by Sex: M={m.mean():.1f}±{m.std():.1f}  F={f.mean():.1f}±{f.std():.1f}  diff={m.mean()-f.mean():+.1f}")

    print(f"\nDescriptive statistics:")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    df = generate_synthetic(81)
    validate_synthetic(df)
    df.to_csv("synthetic_81.csv", index=False)
    print(f"\nSaved to synthetic_81.csv")
