import numpy as np
import pandas as pd
from pathlib import Path

# Single source of truth: same feature distributions as the classifier data
from sample_data import generate_raw_features

RNG = np.random.default_rng(seed=7)

# ── Calibration ───────────────────────────────────────────────────────────────
NOISE_SCALE = 0.08   # rater-disagreement noise; tuned for R^2 ~0.90-0.92

# Formula weights (domain-informed, sum to 1.0)
W = {
    "completeness_score":      0.30,
    "missing_required_count":  0.20,
    "out_of_range_count":      0.15,
    "critical_fields_missing": 0.20,
    "extraction_confidence":   0.10,
    "plausibility_issues":     0.05,
}
assert abs(sum(W.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

MAX_MISSING_REQ  = 10
MAX_OUT_OF_RANGE = 8
MAX_CRITICAL     = 7
MAX_PLAUSIBILITY = 6


# ── Quality score computation ─────────────────────────────────────────────────
def compute_quality_score(df: pd.DataFrame, noise_scale: float = NOISE_SCALE) -> np.ndarray:
    """
    Three-stage construction:

    1. Weighted base score (linear, domain-informed weights).
    2. Nonlinear critical penalty: ANY critical field missing costs an
       extra 0.08 -- a threshold effect the linear part cannot express.
       (Real reviewers treat 'one critical issue' categorically worse
       than 'slightly lower average quality'.)
    3. Percentile stretch to [0,1]: the raw formula floor is ~0.45 because
       realistic records are never worst-case on all dimensions at once.
       Stretching over the empirical 1st-99th percentile restores full
       scale usage (a record at the 1st percentile of this population IS
       a 0.0-quality record, by definition of the score).
    4. Heteroscedastic noise: rater disagreement is largest for mid-range
       records and smallest at the extremes -- matching how human quality
       ratings actually behave.
    """
    base = (
        W["completeness_score"]      * df["completeness_score"]
        + W["missing_required_count"]  * (1 - df["missing_required_count"]  / MAX_MISSING_REQ).clip(0, 1)
        + W["out_of_range_count"]      * (1 - df["out_of_range_count"]      / MAX_OUT_OF_RANGE).clip(0, 1)
        + W["critical_fields_missing"] * (1 - df["critical_fields_missing"] / MAX_CRITICAL).clip(0, 1)
        + W["extraction_confidence"]   * df["extraction_confidence"]
        + W["plausibility_issues"]     * (1 - df["plausibility_issues"]     / MAX_PLAUSIBILITY).clip(0, 1)
    ).values

    # Nonlinear threshold penalty
    crit_penalty = 0.08 * (df["critical_fields_missing"] > 0).values
    base = base - crit_penalty

    # Percentile stretch to full [0,1]
    p_lo, p_hi = np.percentile(base, [1, 99])
    base = (base - p_lo) / (p_hi - p_lo)

    # Heteroscedastic rater noise (max in the middle, min at extremes)
    mid_distance = 1 - np.abs(base - 0.5) * 2
    noise = RNG.normal(0, noise_scale, len(df)) * (0.4 + 0.6 * np.clip(mid_distance, 0, 1))

    return np.round(np.clip(base + noise, 0.0, 1.0), 4)


# ── Build ─────────────────────────────────────────────────────────────────────
def build_dataset(target_n: int = 2000) -> pd.DataFrame:
    df = generate_raw_features(target_n)
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    df["quality_score"] = compute_quality_score(df)
    return df


# ── Validate ──────────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame) -> None:
    assert df["quality_score"].between(0.0, 1.0).all(), \
        "quality_score out of [0,1]"
    assert (df["critical_fields_missing"] <= df["missing_required_count"]).all(), \
        "critical_fields_missing > missing_required_count"
    assert (df["high_severity_flags"] <= df["total_flags"]).all(), \
        "high_severity_flags > total_flags"
    assert df["total_flags"].equals(
        df["missing_required_count"] + df["missing_optional_count"]
        + df["out_of_range_count"] + df["plausibility_issues"]
    ), "total_flags mismatch"

    # All three bands must be populated
    bins = [0.0, 0.33, 0.66, 1.01]
    buckets = pd.cut(df["quality_score"], bins=bins, labels=["Low", "Mid", "High"], right=False)
    for band in ["Low", "Mid", "High"]:
        count = (buckets == band).sum()
        assert count >= len(df) * 0.05, \
            f"Score band '{band}' below 5% -- calibration drifted"

    print("All validation checks passed.")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print(f"  ClinOrigin AI -- Quality Score Data Summary (v3)")
    print(f"{'='*55}")
    print(f"  Total records : {len(df)}")
    print(f"  Noise scale   : {NOISE_SCALE} (heteroscedastic)")
    print(f"  quality_score : min={df['quality_score'].min():.3f}  "
          f"max={df['quality_score'].max():.3f}  "
          f"mean={df['quality_score'].mean():.3f}  "
          f"std={df['quality_score'].std():.3f}")

    bins = [0.0, 0.33, 0.66, 1.01]
    labels_bin = ["Low (0-0.33)", "Mid (0.34-0.66)", "High (0.67-1.0)"]
    buckets = pd.cut(df["quality_score"], bins=bins, labels=labels_bin, right=False)
    print(f"\n  Score distribution:")
    for bucket, count in buckets.value_counts().sort_index().items():
        print(f"    {bucket:<20} {count:>5}  ({count/len(df)*100:.1f}%)")

    print(f"\n  Feature correlations with quality_score:")
    corr = df.corr(numeric_only=True)["quality_score"].drop("quality_score").sort_values(ascending=False)
    for feat, val in corr.items():
        print(f"    {feat:<30} {val:+.3f}")
    print(f"{'='*55}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_dataset(target_n=2000)
    validate(df)
    print_summary(df)

    out_path = Path(__file__).parent / "training_data_quality.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {list(df.columns)}")