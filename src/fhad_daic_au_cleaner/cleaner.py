from pathlib import Path

import pandas as pd

from .config import (
    AU_REGRESSION_COLS,
    CONFIDENCE_THRESHOLD,
    EXCLUDED_SESSIONS,
    FEATURE_COLS,
)
from .utils import get_logger, get_openface_csv_path, get_session_dir, load_csv

logger = get_logger(__name__)


def clean_session(
    session_id: int,
    data_root: Path,
    phq_score: float | None,
    phq_binary: int | None,
    excluded_sessions: frozenset[int] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    _excluded = excluded_sessions if excluded_sessions is not None else EXCLUDED_SESSIONS
    report = {
        "participant_id": session_id,
        "original_frames": 0,
        "dropped_low_confidence": 0,
        "dropped_zero_au": 0,
        "final_frames": 0,
        "phq_score": phq_score,
        "phq_binary": phq_binary,
        "status": "ok",
    }

    if session_id in _excluded:
        report["status"] = "excluded"
        return None, report

    session_dir = get_session_dir(data_root, session_id)
    csv_path = get_openface_csv_path(session_dir, session_id)

    df = load_csv(csv_path)
    if df is None:
        report["status"] = "missing_file"
        return None, report

    df.columns = df.columns.str.strip()

    required_cols = ["confidence"] + AU_REGRESSION_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning("Session %d missing columns: %s", session_id, missing)
        report["status"] = f"missing_columns:{','.join(missing)}"
        return None, report

    report["original_frames"] = len(df)

    low_conf_mask = df["confidence"] < CONFIDENCE_THRESHOLD
    report["dropped_low_confidence"] = int(low_conf_mask.sum())
    df = df[~low_conf_mask]

    au_cols_present = [c for c in AU_REGRESSION_COLS if c in df.columns]
    zero_au_mask = (df[au_cols_present] == 0).all(axis=1)
    report["dropped_zero_au"] = int(zero_au_mask.sum())
    df = df[~zero_au_mask]

    if df.empty:
        report["status"] = "empty_after_cleaning"
        return None, report

    cols_to_keep = [c for c in FEATURE_COLS if c in df.columns]
    missing_feature_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feature_cols:
        logger.warning("Session %d: feature columns not found: %s", session_id, missing_feature_cols)
    df = df[cols_to_keep].copy()

    n = len(df)
    meta = pd.DataFrame({
        "participant_id": [session_id] * n,
        "phq_score": [phq_score] * n,
        "phq_binary": [phq_binary] * n,
    })
    df = pd.concat([meta, df], axis=1)

    report["final_frames"] = len(df)
    return df, report
