"""
Generate 81 clinically-informed synthetic patients for Class I skeletal pattern.

Uses a multivariate normal approach with:
- Sex-specific means (correct dimorphism: males > females for Bigonial, Ramus, Condylar Ht)
- Clinically-informed correlation matrix based on cephalometric literature
- Right/Left AEI generated separately with realistic asymmetry
- ANB always = SNA - SNB (mathematically consistent)
"""

import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh

SEED = 42

# ---------------------------------------------------------------------------
# Population parameters (anchored to the 19 real patients, adjusted for
# clinical correctness)
# ---------------------------------------------------------------------------

# Variable order for the multivariate normal:
# 0: Age, 1: SNA, 2: SNB, 3: SN_GoGn, 4: Occ_Plane,
# 5: Ramus_Ht, 6: Cond_Ht, 7: Bigonial, 8: Mean_AEI

VAR_NAMES = [
    "Age", "SNA", "SNB", "SN_GoGn", "Occ_Plane",
    "Ramus_Ht", "Cond_Ht", "Bigonial", "Mean_AEI",
]

# Female (baseline) means — derived from the 14 female real patients
FEMALE_MEANS = np.array([
    20.0,   # Age
    82.5,   # SNA
    80.0,   # SNB
    26.0,   # SN-GoGn
    13.0,   # Occlusal Plane Angle
    60.0,   # Ramus Height (mm)
    11.5,   # Condylar Height (mm)
    184.0,  # Bigonial Width (mm)
    21.5,   # Mean AEI (deg)
])

# Male adjustments (added to female means)
# Based on known sexual dimorphism in craniofacial morphology
MALE_OFFSETS = np.array([
    0.0,    # Age — no offset
    1.5,    # SNA — slightly more prognathic maxilla
    2.0,    # SNB — more prognathic mandible
    -2.0,   # SN-GoGn — males tend toward lower angle
    -1.5,   # Occ Plane — flatter occlusal plane
    4.0,    # Ramus Height — taller ramus
    2.0,    # Condylar Height — taller condyle
    10.0,   # Bigonial Width — KEY: males WIDER (correct dimorphism)
    3.0,    # Mean AEI — steeper eminence
])

# Standard deviations (common to both sexes, from real data)
SDS = np.array([
    4.5,    # Age
    4.0,    # SNA
    3.5,    # SNB
    5.5,    # SN-GoGn
    3.0,    # Occ Plane
    4.5,    # Ramus Height
    2.0,    # Condylar Height
    11.0,   # Bigonial Width
    3.8,    # Mean AEI
])

# ---------------------------------------------------------------------------
# Clinically-informed correlation matrix (9x9)
# Based on cephalometric literature for Class I patients
# ---------------------------------------------------------------------------

# fmt: off
CORR_MATRIX = np.array([
    # Age    SNA    SNB   GoGn   OccP   RamH  CondH  BigW   AEI
    [ 1.00,  0.00,  0.00, -0.05,  0.00,  0.10,  0.20,  0.05, -0.10],  # Age
    [ 0.00,  1.00,  0.95, -0.35, -0.50,  0.05,  0.05, -0.08,  0.30],  # SNA
    [ 0.00,  0.95,  1.00, -0.35, -0.50,  0.10,  0.15, -0.08,  0.35],  # SNB
    [-0.05, -0.35, -0.35,  1.00,  0.15, -0.30,  0.10,  0.25, -0.35],  # SN-GoGn
    [ 0.00, -0.50, -0.50,  0.15,  1.00, -0.10, -0.20,  0.05, -0.30],  # Occ Plane
    [ 0.10,  0.05,  0.10, -0.30, -0.10,  1.00,  0.25,  0.30, -0.05],  # Ramus Ht
    [ 0.20,  0.05,  0.15,  0.10, -0.20,  0.25,  1.00,  0.10,  0.20],  # Cond Ht
    [ 0.05, -0.08, -0.08,  0.25,  0.05,  0.30,  0.10,  1.00, -0.35],  # Bigonial
    [-0.10,  0.30,  0.35, -0.35, -0.30, -0.05,  0.20, -0.35,  1.00],  # Mean AEI
])
# fmt: on

# Clinical value ranges (hard bounds for validation)
CLINICAL_RANGES = {
    "Age":       (14, 30),
    "SNA":       (74, 95),
    "SNB":       (72, 92),
    "SN_GoGn":   (14, 40),
    "Occ_Plane": (5, 22),
    "Ramus_Ht":  (48, 75),
    "Cond_Ht":   (7, 17),
    "Bigonial":  (150, 210),
    "Mean_AEI":  (10, 32),
}


def _ensure_positive_definite(corr: np.ndarray) -> np.ndarray:
    """If the correlation matrix is not positive-definite, nudge it."""
    min_eig = eigvalsh(corr).min()
    if min_eig < 1e-8:
        corr += np.eye(corr.shape[0]) * (abs(min_eig) + 1e-6)
        # Re-normalize diagonal to 1
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return corr


def _corr_to_cov(corr: np.ndarray, sds: np.ndarray) -> np.ndarray:
    """Convert correlation matrix + standard deviations to covariance matrix."""
    return np.outer(sds, sds) * corr


def generate_synthetic(n: int = 81, seed: int = SEED) -> pd.DataFrame:
    """
    Generate n synthetic patients with clinically correct relationships.

    Returns a DataFrame with columns:
        Patient_ID, Age, Sex, SNA, SNB, ANB, SN_GoGn, Occ_Plane,
        Ramus_Ht, Cond_Ht, Bigonial, Right_AEI, Left_AEI, Mean_AEI
    """
    rng = np.random.default_rng(seed)

    corr = _ensure_positive_definite(CORR_MATRIX.copy())
    cov = _corr_to_cov(corr, SDS)

    # Assign sex: ~30% male (matching real data proportion)
    sex = rng.binomial(1, 0.30, size=n)

    rows = []
    attempts = 0
    max_attempts = n * 20  # safety valve

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        s = sex[len(rows)] if len(rows) < n else rng.binomial(1, 0.30)

        # Sex-specific means
        means = FEMALE_MEANS + s * MALE_OFFSETS

        # Sample from multivariate normal
        sample = rng.multivariate_normal(means, cov)

        # Round Age to integer
        sample[0] = round(sample[0])

        # Build a dict for validation
        vals = {name: sample[i] for i, name in enumerate(VAR_NAMES)}

        # Enforce clinical ranges
        valid = True
        for name, (lo, hi) in CLINICAL_RANGES.items():
            if vals[name] < lo or vals[name] > hi:
                valid = False
                break

        if not valid:
            continue

        # Compute ANB = SNA - SNB (must be Class I range: -1 to 5)
        anb = vals["SNA"] - vals["SNB"]
        if anb < -1.0 or anb > 5.0:
            continue

        # Generate Right/Left AEI with realistic asymmetry
        mean_aei = vals["Mean_AEI"]
        asymmetry = rng.normal(0, 1.5)  # typical R-L difference ~2-4 deg
        asymmetry = np.clip(asymmetry, -5.0, 5.0)
        right_aei = mean_aei + asymmetry / 2
        left_aei = mean_aei - asymmetry / 2

        # Ensure individual AEI values are in plausible range
        if right_aei < 8 or right_aei > 35 or left_aei < 8 or left_aei > 35:
            continue

        rows.append({
            "Patient_ID": len(rows) + 1,
            "Age": int(vals["Age"]),
            "Sex": s,
            "SNA": round(vals["SNA"], 1),
            "SNB": round(vals["SNB"], 1),
            "ANB": round(anb, 1),
            "SN_GoGn": round(vals["SN_GoGn"], 1),
            "Occ_Plane": round(vals["Occ_Plane"], 1),
            "Ramus_Ht": round(vals["Ramus_Ht"], 1),
            "Cond_Ht": round(vals["Cond_Ht"], 1),
            "Bigonial": round(vals["Bigonial"], 1),
            "Right_AEI": round(right_aei, 1),
            "Left_AEI": round(left_aei, 1),
            "Mean_AEI": round(mean_aei, 1),
        })

    if len(rows) < n:
        raise RuntimeError(
            f"Could only generate {len(rows)}/{n} valid patients "
            f"after {max_attempts} attempts. Loosen constraints or check parameters."
        )

    return pd.DataFrame(rows)


def validate_synthetic(df: pd.DataFrame) -> None:
    """Print validation summary for the generated synthetic data."""
    print(f"Generated {len(df)} synthetic patients")
    print(f"\nSex distribution: {df['Sex'].value_counts().to_dict()} (0=F, 1=M)")

    # ANB consistency
    anb_diff = (df["SNA"] - df["SNB"] - df["ANB"]).abs()
    print(f"ANB = SNA - SNB consistency: max deviation = {anb_diff.max():.2f}")

    # Mean AEI consistency
    mean_check = ((df["Right_AEI"] + df["Left_AEI"]) / 2 - df["Mean_AEI"]).abs()
    print(f"Mean AEI = (R+L)/2 consistency: max deviation = {mean_check.max():.2f}")

    # Sex dimorphism checks
    males = df[df["Sex"] == 1]
    females = df[df["Sex"] == 0]
    print(f"\nSex dimorphism (Male mean / Female mean):")
    for col in ["Bigonial", "Ramus_Ht", "Cond_Ht", "Mean_AEI"]:
        m = males[col].mean()
        f = females[col].mean()
        direction = "M > F" if m > f else "M < F (WRONG)" if m < f else "M = F"
        print(f"  {col:12s}: M={m:.1f}  F={f:.1f}  ({direction})")

    # Key correlation checks
    print(f"\nKey correlations:")
    print(f"  SNA-SNB:             {df['SNA'].corr(df['SNB']):.3f}  (expect ~0.95)")
    print(f"  SN_GoGn-Ramus_Ht:   {df['SN_GoGn'].corr(df['Ramus_Ht']):.3f}  (expect negative)")
    print(f"  Age-Cond_Ht:         {df['Age'].corr(df['Cond_Ht']):.3f}  (expect positive)")
    print(f"  SNB-Mean_AEI:        {df['SNB'].corr(df['Mean_AEI']):.3f}  (expect positive)")
    print(f"  SN_GoGn-Mean_AEI:    {df['SN_GoGn'].corr(df['Mean_AEI']):.3f}  (expect negative)")
    print(f"  Bigonial-Mean_AEI:   {df['Bigonial'].corr(df['Mean_AEI']):.3f}  (expect negative)")

    # R-L asymmetry
    rl_diff = (df["Right_AEI"] - df["Left_AEI"]).abs()
    print(f"\nR-L AEI asymmetry: mean={rl_diff.mean():.1f} deg, max={rl_diff.max():.1f} deg")

    # Descriptive stats
    print(f"\nDescriptive statistics:")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    from config import SYNTHETIC_DATA_PATH

    df = generate_synthetic(81)
    validate_synthetic(df)
    df.to_excel(SYNTHETIC_DATA_PATH, index=False)
    print(f"\nSaved to {SYNTHETIC_DATA_PATH}")
