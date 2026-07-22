import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fhad_daic_au_cleaner.cleaner import clean_session
from fhad_daic_au_cleaner.cnn_cleaner import clean_cnn_session
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
from fhad_daic_au_cleaner.spectrogram_cleaner import clean_spectrogram_session
from fhad_daic_au_cleaner.transcript_cleaner import clean_transcript_session
from fhad_daic_au_cleaner.utils import get_logger, list_session_ids, load_csv
from fhad_daic_au_cleaner.wav2vec_cleaner import clean_wav2vec_session

logger = get_logger("run_cleaning")

MODALITY_TASKS: dict[str, list[dict]] = {
    "au": [{"label": "AU", "suffix": "", "fn": clean_session, "kw": {}}],
    "egemaps": [{"label": "eGeMAPS", "suffix": "_egemaps", "fn": clean_egemaps_session, "kw": {}}],
    "au_egemaps": [
        {"label": "AU", "suffix": "", "fn": clean_session, "kw": {}},
        {"label": "eGeMAPS", "suffix": "_egemaps", "fn": clean_egemaps_session, "kw": {}},
    ],
    "cnn_resnet": [{"label": "CNN_ResNet", "suffix": "_cnn_resnet", "fn": clean_cnn_session, "kw": {"variant": "ResNet"}}],
    "cnn_vgg": [{"label": "CNN_VGG", "suffix": "_cnn_vgg", "fn": clean_cnn_session, "kw": {"variant": "VGG"}}],
    "cnn": [
        {"label": "CNN_ResNet", "suffix": "_cnn_resnet", "fn": clean_cnn_session, "kw": {"variant": "ResNet"}},
        {"label": "CNN_VGG", "suffix": "_cnn_vgg", "fn": clean_cnn_session, "kw": {"variant": "VGG"}},
    ],
    "densenet": [{"label": "DenseNet201", "suffix": "_densenet", "fn": clean_spectrogram_session, "kw": {"variant": "densenet201"}}],
    "vgg16": [{"label": "VGG16", "suffix": "_vgg16", "fn": clean_spectrogram_session, "kw": {"variant": "vgg16"}}],
    "spectrogram": [
        {"label": "DenseNet201", "suffix": "_densenet", "fn": clean_spectrogram_session, "kw": {"variant": "densenet201"}},
        {"label": "VGG16", "suffix": "_vgg16", "fn": clean_spectrogram_session, "kw": {"variant": "vgg16"}},
    ],
    "all": [
        {"label": "AU", "suffix": "", "fn": clean_session, "kw": {}},
        {"label": "eGeMAPS", "suffix": "_egemaps", "fn": clean_egemaps_session, "kw": {}},
        {"label": "CNN_ResNet", "suffix": "_cnn_resnet", "fn": clean_cnn_session, "kw": {"variant": "ResNet"}},
    ],
    "all_files": [
        {"label": "AU", "suffix": "", "fn": clean_session, "kw": {}},
        {"label": "eGeMAPS", "suffix": "_egemaps", "fn": clean_egemaps_session, "kw": {}},
        {"label": "CNN_ResNet", "suffix": "_cnn_resnet", "fn": clean_cnn_session, "kw": {"variant": "ResNet"}},
        {"label": "CNN_VGG", "suffix": "_cnn_vgg", "fn": clean_cnn_session, "kw": {"variant": "VGG"}},
        {"label": "DenseNet201", "suffix": "_densenet", "fn": clean_spectrogram_session, "kw": {"variant": "densenet201"}},
        {"label": "VGG16", "suffix": "_vgg16", "fn": clean_spectrogram_session, "kw": {"variant": "vgg16"}},
    ],
    "wav2vec": [{"label": "Wav2Vec2", "suffix": "_wav2vec", "fn": clean_wav2vec_session, "kw": {}}],
    "transcript": [{"label": "Transcript", "suffix": "_transcript", "fn": clean_transcript_session, "kw": {}}],
}


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
    tasks = MODALITY_TASKS.get(modality)
    if not tasks:
        logger.error("Unknown modality: %s", modality)
        return

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

    task_reports: dict[str, list[dict]] = {t["label"]: [] for t in tasks}

    for sid, (split, phq_score, phq_binary) in splits.items():
        logger.info("Processing session %d [%s]", sid, split)
        out_dir = output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            label = task["label"]
            suffix = task["suffix"]
            cleaned_df, report = task["fn"](sid, data_root, phq_score, phq_binary, **task["kw"])
            report["split"] = split
            if cleaned_df is not None:
                out_path = out_dir / f"{sid}{suffix}_clean.csv"
                cleaned_df.to_csv(out_path, index=False)
                logger.info("  [%s] Saved %d frames -> %s", label, len(cleaned_df), out_path)
            else:
                logger.warning("  [%s] Skipped session %d: %s", label, sid, report["status"])
            task_reports[label].append(report)

    output_root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        label = task["label"]
        reports = task_reports[label]
        if not reports:
            continue
        report_path = output_root / f"cleaning_report_{label}.csv"
        pd.DataFrame(reports).to_csv(report_path, index=False)
        logger.info("%s cleaning report saved to %s", label, report_path)
        cleaned = sum(1 for r in reports if r["status"] == "ok")
        excluded = sum(1 for r in reports if r["status"] == "excluded")
        print(f"[{label}] Done. {cleaned} cleaned, {excluded} excluded, {len(reports) - cleaned - excluded} failed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--train-label", type=Path, default=TRAIN_LABEL_CSV)
    parser.add_argument("--dev-label", type=Path, default=DEV_LABEL_CSV)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--modality",
        choices=[
            "au", "egemaps", "au_egemaps", "cnn_resnet", "cnn_vgg", "cnn",
            "densenet", "vgg16", "spectrogram",
            "all", "all_files", "wav2vec", "transcript",
        ],
        default="all",
    )
    args = parser.parse_args()
    run(args.data_root, args.train_label, args.dev_label, args.output_root, args.modality)


if __name__ == "__main__":
    main()
