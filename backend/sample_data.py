import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax

RNG = np.random.default_rng(seed=42)

# ── Calibration parameters (tuned, see module docstring) ──────────────────────
NOISE_STD        = 0.02   # label noise on risk scale; higher = more ambiguity
TEMPERATURE      = 3.5    # logit sharpness; higher = more confident labels
CLEAN_PERCENTILE = 55     # risk percentile below which Clean dominates
BLOCK_PERCENTILE = 85     # risk percentile above which Block dominates


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ci(arr, lo, hi):
    """Clip to integer range."""
    return np.clip(np.round(arr).astype(int), lo, hi)


# ── Step 1: Generate raw features ─────────────────────────────────────────────
def generate_raw_features(n: int) -> pd.DataFrame:
    """
    Generate n records with realistic, overlapping feature distributions.
    Single continuous quality spectrum -- no sub-population engineering.
    """
    # completeness_score: bimodal -- most records decent, long left tail
    cs_mode = RNG.choice([0, 1], n, p=[0.25, 0.75])
    completeness_score = np.where(
        cs_mode == 1,
        np.clip(RNG.beta(7, 1.8, n), 0.55, 1.00),
        np.clip(RNG.beta(2, 5, n), 0.05, 0.70),
    )

    # missing_required_count: zero-inflated (55% of records have none)
    missing_required_count = _ci(
        np.where(RNG.random(n) < 0.55, 0, RNG.exponential(2.5, n)),
        0, 10,
    )

    # missing_optional_count: always some, right-skewed
    missing_optional_count = _ci(RNG.exponential(2.0, n), 0, 12)

    # out_of_range_count: mostly low, occasional spikes
    out_of_range_count = _ci(
        np.where(
            RNG.random(n) < 0.65,
            RNG.choice([0, 1], n, p=[0.6, 0.4]),
            RNG.normal(3.5, 1.5, n),
        ),
        0, 8,
    )

    # plausibility_issues: correlated with out_of_range, noisy
    plausibility_issues = _ci(
        out_of_range_count * RNG.beta(1.5, 3.0, n) + RNG.exponential(0.5, n),
        0, 6,
    )

    # total_flags: derived sum (pipeline invariant)
    total_flags = (
        missing_required_count
        + missing_optional_count
        + out_of_range_count
        + plausibility_issues
    )

    # critical_fields_missing: subset of missing_required_count
    critical_fraction = RNG.beta(1.2, 4.0, n)  # skewed low
    critical_fields_missing = _ci(
        missing_required_count * critical_fraction,
        0, 10,
    )
    critical_fields_missing = np.minimum(critical_fields_missing, missing_required_count)

    # high_severity_flags: subset of total_flags, with severity spikes
    hsf_base = total_flags * RNG.beta(1.0, 3.5, n)
    spike_mask = RNG.random(n) < 0.12
    hsf_base = np.where(spike_mask, hsf_base + RNG.uniform(2, 5, n), hsf_base)
    high_severity_flags = _ci(hsf_base, 0, 25)
    high_severity_flags = np.minimum(high_severity_flags, total_flags)

    # extraction_confidence: correlated with completeness, noisy
    conf_noise = RNG.normal(0, 0.08, n)
    extraction_confidence = np.clip(
        completeness_score * 0.85 + 0.10 + conf_noise, 0.04, 1.00
    )

    # field_count_total: log-normal around realistic CRF size
    field_count_total = _ci(RNG.lognormal(np.log(24), 0.35, n), 8, 50)

    return pd.DataFrame({
        "completeness_score":      np.round(completeness_score, 4),
        "missing_required_count":  missing_required_count,
        "missing_optional_count":  missing_optional_count,
        "out_of_range_count":      out_of_range_count,
        "plausibility_issues":     plausibility_issues,
        "total_flags":             total_flags,
        "critical_fields_missing": critical_fields_missing,
        "high_severity_flags":     high_severity_flags,
        "extraction_confidence":   np.round(extraction_confidence, 4),
        "field_count_total":       field_count_total,
    })


# ── Step 2: Latent risk score ──────────────────────────────────────────────────
def compute_risk_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Continuous risk score per record. Higher = more likely Block.
    Domain-informed weights; no hard thresholds.
    """
    max_flags = 20.0
    max_missing = 10.0

    score = (
        0.30 * (1.0 - df["completeness_score"])
        + 0.22 * (df["critical_fields_missing"] / max_missing).clip(0, 1)
        + 0.20 * (df["high_severity_flags"] / max_flags).clip(0, 1)
        + 0.12 * (df["missing_required_count"] / max_missing).clip(0, 1)
        + 0.08 * (df["out_of_range_count"] / max_flags).clip(0, 1)
        + 0.08 * (1.0 - df["extraction_confidence"])
    ).values

    return score


# ── Step 3: Probabilistic label assignment (percentile-calibrated) ─────────────
def assign_labels_probabilistic(
    df: pd.DataFrame,
    noise_std: float = NOISE_STD,
    temperature: float = TEMPERATURE,
) -> pd.Series:
    """
    Labels are drawn from a softmax over three logits whose thresholds
    are calibrated to PERCENTILES of the actual risk distribution.

    Why percentiles instead of fixed values:
      The risk score's real range depends on the feature distributions.
      Fixed thresholds (v2: 0.35 / 0.72) silently missed the actual range
      (~0.0-0.56), making Block unreachable except by noise.
      Percentile thresholds adapt automatically -- the 85th percentile is
      always inside the distribution, whatever its scale.

    Ambiguity model:
      Records far from the thresholds get confident labels (sharp logits).
      Records near a threshold can tip either way (softmax sampling).
      This mirrors real inter-reviewer disagreement, which is concentrated
      on genuine boundary cases, not spread uniformly.
    """
    n = len(df)
    risk = compute_risk_scores(df)

    # Small calibrated noise: boundary records can tip, clear cases cannot
    risk_noisy = risk + RNG.normal(0, noise_std, n)

    # Percentile-calibrated thresholds (adapt to actual distribution)
    t_clean  = np.percentile(risk_noisy, CLEAN_PERCENTILE)
    t_block  = np.percentile(risk_noisy, BLOCK_PERCENTILE)
    mid      = (t_clean + t_block) / 2
    halfspan = max((t_block - t_clean) / 2, 1e-6)

    # Three logits on a normalised scale:
    #   Clean  : high below t_clean, falls linearly
    #   Review : peaks at midpoint between thresholds
    #   Block  : high above t_block, rises linearly
    logits = np.column_stack([
        temperature * (t_clean - risk_noisy) / halfspan,
        temperature * (1.0 - ((risk_noisy - mid) / halfspan) ** 2),
        temperature * (risk_noisy - t_block) / halfspan,
    ])

    probs = softmax(logits, axis=1)
    labels_int = np.array([RNG.choice(3, p=p) for p in probs])

    label_map = {0: "Clean", 1: "Review", 2: "Block"}
    return pd.Series([label_map[i] for i in labels_int], index=df.index)


# ── Step 4: Assemble ───────────────────────────────────────────────────────────
def build_dataset(target_n: int = 2000) -> pd.DataFrame:
    df = generate_raw_features(target_n + 50)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["label"] = assign_labels_probabilistic(df)
    df = df.head(target_n).reset_index(drop=True)
    return df


# ── Step 5: Validate structural constraints ────────────────────────────────────
def validate(df: pd.DataFrame) -> None:
    """
    Structural/arithmetic constraints only.
    Label-rule consistency is intentionally NOT enforced --
    probabilistic labels may violate hard thresholds near boundaries.
    """
    assert (df["critical_fields_missing"] <= df["missing_required_count"]).all(), \
        "critical_fields_missing > missing_required_count"
    assert (df["high_severity_flags"] <= df["total_flags"]).all(), \
        "high_severity_flags > total_flags"
    assert df["completeness_score"].between(0.0, 1.0).all(), \
        "completeness_score out of [0, 1]"
    assert df["extraction_confidence"].between(0.0, 1.0).all(), \
        "extraction_confidence out of [0, 1]"
    assert df["total_flags"].equals(
        df["missing_required_count"]
        + df["missing_optional_count"]
        + df["out_of_range_count"]
        + df["plausibility_issues"]
    ), "total_flags != sum of components"

    # Distribution sanity: every class must be meaningfully populated
    counts = df["label"].value_counts(normalize=True)
    for label, lo in [("Clean", 0.40), ("Review", 0.20), ("Block", 0.08)]:
        assert counts.get(label, 0) >= lo, \
            f"{label} below {lo:.0%} -- calibration drifted, check parameters"

    print("All structural constraint checks passed.")


# ── Step 6: Summary ────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print(f"  ClinOrigin AI -- Training Data Summary (v3 calibrated)")
    print(f"{'='*55}")
    print(f"  Total records : {len(df)}")
    print(f"  Calibration   : noise={NOISE_STD}, temp={TEMPERATURE}, "
          f"percentiles={CLEAN_PERCENTILE}/{BLOCK_PERCENTILE}")

    print(f"\n  Label distribution:")
    for label in ["Clean", "Review", "Block"]:
        count = (df["label"] == label).sum()
        pct = count / len(df) * 100
        print(f"    {label:<8} {count:>5}  ({pct:.1f}%)")

    print(f"\n  Feature means per label:")
    cols = [
        "completeness_score", "missing_required_count", "out_of_range_count",
        "critical_fields_missing", "high_severity_flags",
        "extraction_confidence", "total_flags",
    ]
    print(df.groupby("label")[cols].mean().round(2).to_string())

    # Boundary overlap diagnostic
    n_clean = (df["label"] == "Clean").sum()
    n_block = (df["label"] == "Block").sum()
    clean_violates = df[
        (df["label"] == "Clean")
        & (
            (df["critical_fields_missing"] > 0)
            | (df["high_severity_flags"] >= 4)
            | (df["completeness_score"] < 0.40)
        )
    ]
    block_soft = df[
        (df["label"] == "Block")
        & (df["critical_fields_missing"] == 0)
        & (df["high_severity_flags"] < 4)
        & (df["completeness_score"] >= 0.40)
    ]
    print(f"\n  Boundary overlap (intentional ambiguity):")
    print(f"    Clean records that would violate v1 hard rules : "
          f"{len(clean_violates)} / {n_clean} ({100*len(clean_violates)/max(n_clean,1):.1f}%)")
    print(f"    Block records without any v1 hard trigger      : "
          f"{len(block_soft)} / {n_block} ({100*len(block_soft)/max(n_block,1):.1f}%)")
    print(f"{'='*55}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_dataset(target_n=2000)
    validate(df)
    print_summary(df)

    out_path = Path(__file__).parent / "training_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {list(df.columns)}")