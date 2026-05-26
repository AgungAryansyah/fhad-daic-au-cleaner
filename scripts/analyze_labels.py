import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fhad_daic_au_cleaner.config import (
    DEV_LABEL_CSV,
    LABEL_PARTICIPANT_COL,
    LABEL_PHQ_BINARY_COL,
    LABEL_PHQ_SCORE_COL,
    TRAIN_LABEL_CSV,
)
from fhad_daic_au_cleaner.utils import get_logger, load_csv

logger = get_logger("analyze_labels")

PHQ_BINS = [0, 5, 10, 15, 20, 25]
PHQ_BIN_LABELS = ["Minimal (0-4)", "Mild (5-9)", "Moderate (10-14)", "Mod. Severe (15-19)", "Severe (20-24)"]


def print_section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def analyze(df: pd.DataFrame, split: str) -> None:
    print_section(f"{split.upper()} SPLIT  (n={len(df)})")

    print(f"\n  [Binary Distribution]")
    binary_counts = df[LABEL_PHQ_BINARY_COL].value_counts().sort_index()
    for label, count in binary_counts.items():
        name = "Depressed" if label == 1 else "Not Depressed"
        pct = 100 * count / len(df)
        print(f"    {name} (label={label}): {count:>4}  ({pct:.1f}%)")

    print(f"\n  [PHQ Score — Continuous]")
    scores = df[LABEL_PHQ_SCORE_COL]
    print(f"    min:  {scores.min():.1f}")
    print(f"    max:  {scores.max():.1f}")
    print(f"    mean: {scores.mean():.2f}")
    print(f"    std:  {scores.std():.2f}")

    print(f"\n  [PHQ Score — Binned Severity]")
    binned = pd.cut(scores, bins=PHQ_BINS, labels=PHQ_BIN_LABELS, right=False)
    bin_counts = binned.value_counts().reindex(PHQ_BIN_LABELS)
    for label, count in bin_counts.items():
        pct = 100 * count / len(df)
        print(f"    {label:<26}: {count:>4}  ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-label", type=Path, default=TRAIN_LABEL_CSV)
    parser.add_argument("--dev-label", type=Path, default=DEV_LABEL_CSV)
    args = parser.parse_args()

    print("\nPHQ-8 Label Distribution Analysis")

    for path, split in [(args.train_label, "train"), (args.dev_label, "dev")]:
        df = load_csv(path)
        if df is None:
            logger.error("Could not load: %s", path)
            continue
        df.columns = df.columns.str.strip()
        analyze(df, split)

    print()


if __name__ == "__main__":
    main()
