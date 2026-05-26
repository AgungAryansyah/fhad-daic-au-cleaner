import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fhad_daic_au_cleaner.cleaner import clean_session
from fhad_daic_au_cleaner.config import (
    DATA_ROOT,
    DEV_LABEL_CSV,
    LABEL_PARTICIPANT_COL,
    LABEL_PHQ_BINARY_COL,
    LABEL_PHQ_SCORE_COL,
    OUTPUT_ROOT,
    TRAIN_LABEL_CSV,
)
from fhad_daic_au_cleaner.egemaps_cleaner import clean_egemaps_session
from fhad_daic_au_cleaner.utils import get_logger, list_session_ids, load_csv

logger = get_logger("run_cleaning")


def load_labels(label_path: Path) -> dict[int, tuple[float, int]]:
    df = load_csv(label_path)
    if df is None:
        logger.error("Could not load label file: %s", label_path)
        return {}
    df.columns = df.columns.str.strip()
    labels = {}
    for _, row in df.iterrows():
        sid = int(row[LABEL_PARTICIPANT_COL])
        score = float(row[LABEL_PHQ_SCORE_COL])
        binary = int(row[LABEL_PHQ_BINARY_COL])
        labels[sid] = (score, binary)
    return labels


def run(
    data_root: Path,
    train_label: Path,
    dev_label: Path,
    output_root: Path,
    modality: str,
) -> None:
    train_labels = load_labels(train_label)
    dev_labels = load_labels(dev_label)

    all_session_ids = list_session_ids(data_root)

    splits: dict[int, tuple] = {}
    for sid in all_session_ids:
        if sid in train_labels:
            splits[sid] = ("train", *train_labels[sid])
        elif sid in dev_labels:
            splits[sid] = ("dev", *dev_labels[sid])
        else:
            splits[sid] = ("test", None, None)

    au_reports = []
    egemaps_reports = []

    for sid, (split, phq_score, phq_binary) in splits.items():
        logger.info("Processing session %d [%s]", sid, split)
        out_dir = output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        if modality in ("au", "all"):
            cleaned_df, report = clean_session(sid, data_root, phq_score, phq_binary)
            report["split"] = split
            if cleaned_df is not None:
                out_path = out_dir / f"{sid}_clean.csv"
                cleaned_df.to_csv(out_path, index=False)
                logger.info("  [AU] Saved %d frames -> %s", len(cleaned_df), out_path)
            else:
                logger.warning("  [AU] Skipped session %d: %s", sid, report["status"])
            au_reports.append(report)

        if modality in ("egemaps", "all"):
            cleaned_df, report = clean_egemaps_session(sid, data_root, phq_score, phq_binary)
            report["split"] = split
            if cleaned_df is not None:
                out_path = out_dir / f"{sid}_egemaps_clean.csv"
                cleaned_df.to_csv(out_path, index=False)
                logger.info("  [eGeMAPS] Saved %d frames -> %s", len(cleaned_df), out_path)
            else:
                logger.warning("  [eGeMAPS] Skipped session %d: %s", sid, report["status"])
            egemaps_reports.append(report)

    output_root.mkdir(parents=True, exist_ok=True)

    if au_reports:
        report_path = output_root / "cleaning_report_au.csv"
        pd.DataFrame(au_reports).to_csv(report_path, index=False)
        logger.info("AU cleaning report saved to %s", report_path)
        cleaned = sum(1 for r in au_reports if r["status"] == "ok")
        excluded = sum(1 for r in au_reports if r["status"] == "excluded")
        print(f"\n[AU] Done. {cleaned} cleaned, {excluded} excluded, {len(au_reports)-cleaned-excluded} failed.")

    if egemaps_reports:
        report_path = output_root / "cleaning_report_egemaps.csv"
        pd.DataFrame(egemaps_reports).to_csv(report_path, index=False)
        logger.info("eGeMAPS cleaning report saved to %s", report_path)
        cleaned = sum(1 for r in egemaps_reports if r["status"] == "ok")
        excluded = sum(1 for r in egemaps_reports if r["status"] == "excluded")
        print(f"[eGeMAPS] Done. {cleaned} cleaned, {excluded} excluded, {len(egemaps_reports)-cleaned-excluded} failed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--train-label", type=Path, default=TRAIN_LABEL_CSV)
    parser.add_argument("--dev-label", type=Path, default=DEV_LABEL_CSV)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--modality", choices=["au", "egemaps", "all"], default="all")
    args = parser.parse_args()
    run(args.data_root, args.train_label, args.dev_label, args.output_root, args.modality)


if __name__ == "__main__":
    main()
