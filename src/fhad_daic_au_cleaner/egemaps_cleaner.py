from pathlib import Path

import pandas as pd

from .config import (
    EGEMAPS_FEATURE_COLS,
    EGEMAPS_KEEP_COLS,
    EGEMAPS_SENTINEL,
    EXCLUDED_SESSIONS,
)
from .utils import get_egemaps_csv_path, get_logger, get_session_dir, load_csv

logger = get_logger(__name__)


def clean_egemaps_session(
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
        "sentinel_replacements": 0,
        "final_frames": 0,
        "phq_score": phq_score,
        "phq_binary": phq_binary,
        "status": "ok",
    }

    if session_id in _excluded:
        report["status"] = "excluded"
        return None, report

    session_dir = get_session_dir(data_root, session_id)
    csv_path = get_egemaps_csv_path(session_dir, session_id)

    df = load_csv(csv_path, sep=";")
    if df is None:
        report["status"] = "missing_file"
        return None, report

    df.columns = df.columns.str.strip()

    missing = [c for c in EGEMAPS_FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning("Session %d missing eGeMAPS columns: %s", session_id, missing)
        report["status"] = f"missing_columns:{','.join(missing)}"
        return None, report

    report["original_frames"] = len(df)

    sentinel_mask = df[EGEMAPS_FEATURE_COLS] == EGEMAPS_SENTINEL
    report["sentinel_replacements"] = int(sentinel_mask.sum().sum())
    df[EGEMAPS_FEATURE_COLS] = df[EGEMAPS_FEATURE_COLS].where(~sentinel_mask, other=float("nan"))

    cols_to_keep = [c for c in EGEMAPS_KEEP_COLS if c in df.columns]
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
