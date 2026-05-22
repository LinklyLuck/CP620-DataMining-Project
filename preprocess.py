from pathlib import Path
import pandas as pd

SEED = 42
N_STRAT = 15_000
N_BAL_PER_CLASS = 5_000

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

RAW_PATH = DATA_DIR / "brfss_raw.csv"
IMBALANCED_PATH = DATA_DIR / "heart_disease_clean.csv"
BALANCED_PATH = DATA_DIR / "heart_disease_balanced.csv"


def main() -> None:
    # Read the original cleaned BRFSS dataset.
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Rename the original target column to match the project notation.
    if "HeartDiseaseorAttack" in df.columns:
        df = df.rename(columns={"HeartDiseaseorAttack": "target"})
    elif "target" not in df.columns:
        raise ValueError(
            "The raw dataset must contain 'HeartDiseaseorAttack' or 'target'."
        )

    df = df.reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    if df.isna().any().any():
        raise ValueError("Missing values found in the raw dataset.")

    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1.")

    print(f"Raw: {len(df):,} rows x {df.shape[1]} cols")
    print(f"  positive rate: {df['target'].mean():.4%}")

    #Draw the stratified 15,000-row secondary imbalanced sample.
    pos_pool = df[df["target"] == 1]
    neg_pool = df[df["target"] == 0]

    positive_rate = len(pos_pool) / len(df)
    n_strat_pos = round(N_STRAT * positive_rate)
    n_strat_neg = N_STRAT - n_strat_pos

    strat_pos = pos_pool.sample(n=n_strat_pos, random_state=SEED)
    strat_neg = neg_pool.sample(n=n_strat_neg, random_state=SEED)

    strat = (
        pd.concat([strat_pos, strat_neg])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    strat.to_csv(IMBALANCED_PATH, index=False)

    print("\n[1] Secondary imbalanced sample (heart_disease_clean.csv):")
    print(
        f"    {len(strat):,} rows, {int(strat['target'].sum()):,} positive "
        f"({strat['target'].mean():.2%})"
    )

    # Remove those source rows, then draw the balanced sample.
    used_idx = set(strat_pos.index) | set(strat_neg.index)

    remaining = df.drop(index=list(used_idx))
    remaining_pos = remaining[remaining["target"] == 1]
    remaining_neg = remaining[remaining["target"] == 0]

    print("\nAfter removing the secondary sample, the remaining pool has:")
    print(
        f"    {len(remaining_pos):,} positives remaining "
        f"(need {N_BAL_PER_CLASS:,})"
    )
    print(
        f"    {len(remaining_neg):,} negatives remaining "
        f"(need {N_BAL_PER_CLASS:,})"
    )

    assert len(remaining_pos) >= N_BAL_PER_CLASS, (
        "Not enough positives for balanced sample."
    )
    assert len(remaining_neg) >= N_BAL_PER_CLASS, (
        "Not enough negatives for balanced sample."
    )

    bal_pos = remaining_pos.sample(n=N_BAL_PER_CLASS, random_state=SEED)
    bal_neg = remaining_neg.sample(n=N_BAL_PER_CLASS, random_state=SEED)

    balanced = (
        pd.concat([bal_pos, bal_neg])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    balanced.to_csv(BALANCED_PATH, index=False)

    print("\n[2] Primary balanced sample (heart_disease_balanced.csv):")
    print(
        f"    {len(balanced):,} rows, {int(balanced['target'].sum()):,} positive "
        f"({balanced['target'].mean():.2%})"
    )

    # Verify size, balance, and zero source-row overlap.
    bal_idx = set(bal_pos.index) | set(bal_neg.index)
    overlap = used_idx & bal_idx

    assert len(strat) == N_STRAT, "Incorrect imbalanced sample size."
    assert len(balanced) == 2 * N_BAL_PER_CLASS, (
        "Incorrect balanced sample size."
    )
    assert int(balanced["target"].sum()) == N_BAL_PER_CLASS, (
        "Balanced sample is not 50/50."
    )
    assert len(overlap) == 0, "Datasets overlap at the source-row level."

    print("\n[3] Disjointness check:")
    print(f"    secondary sample source rows: {len(used_idx):,}")
    print(f"    primary sample source rows:   {len(bal_idx):,}")
    print(f"    overlap:                      {len(overlap)}")
    print("    PASS — the two subsamples share zero source rows.")

    # Identical feature values may still occur because the cleaned release
    # does not contain respondent identifiers.
    content_overlap = (
        set(map(tuple, balanced.to_numpy()))
        & set(map(tuple, strat.to_numpy()))
    )

    print(
        f"    identical-content row patterns across samples: "
        f"{len(content_overlap):,}"
    )
    print(
        "    This is allowed: identical survey-answer profiles may represent "
        "different respondents."
    )

    print("\nSaved files:")
    print(f"    {IMBALANCED_PATH}")
    print(f"    {BALANCED_PATH}")


if __name__ == "__main__":
    main()