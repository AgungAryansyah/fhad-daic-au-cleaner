from pathlib import Path

import pandas as pd

from .config import (
    EXCLUDED_SESSIONS,
    SPECTROGRAM_DENSENET_COL_PREFIX,
    SPECTROGRAM_VGG16_COL_PREFIX,
)
from .utils import get_logger, get_session_dir, get_spectrogram_csv_path, load_csv

logger = get_logger(__name__)

VARIANT_META = {
    "densenet201": {"file_key": "densenet201", "col_prefix": SPECTROGRAM_DENSENET_COL_PREFIX},
    "vgg16": {"file_key": "vgg16", "col_prefix": SPECTROGRAM_VGG16_COL_PREFIX},
}


def clean_spectrogram_session(
    session_id: int,
    data_root: Path,
    phq_score: float | None,
    phq_binary: int | None,
    variant: str,
    excluded_sessions: frozenset[int] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    _excluded = excluded_sessions if excluded_sessions is not None else EXCLUDED_SESSIONS
    meta = VARIANT_META.get(variant, {})
    col_prefix = meta.get("col_prefix", variant)

    report = {
        "participant_id": session_id,
        "modality": f"spectrogram_{variant}",
        "original_frames": 0,
        "final_frames": 0,
        "feature_dim": 0,
        "phq_score": phq_score,
        "phq_binary": phq_binary,
        "status": "ok",
    }

    if session_id in _excluded:
        report["status"] = "excluded"
        return None, report

    session_dir = get_session_dir(data_root, session_id)
    csv_path = get_spectrogram_csv_path(session_dir, session_id, variant=variant)

    df = load_csv(csv_path, header=None)
    if df is None:
        report["status"] = "missing_file"
        return None, report

    if df.empty:
        report["status"] = "empty_file"
        return None, report

    try:
        df = df.apply(pd.to_numeric)
    except Exception:
        logger.warning("Session %d spectrogram %s: non-numeric values found", session_id, variant)

    report["original_frames"] = len(df)
    report["final_frames"] = len(df)
    report["feature_dim"] = df.shape[1]

    df.columns = [f"{col_prefix}_{i}" for i in range(df.shape[1])]

    meta = pd.DataFrame({
        "participant_id": [session_id] * len(df),
        "phq_score": [phq_score] * len(df),
        "phq_binary": [phq_binary] * len(df),
        "frame": range(len(df)),
    })
    df = pd.concat([meta, df], axis=1)

    return df, report
