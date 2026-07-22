from pathlib import Path

import pandas as pd

from .config import (
    EXCLUDED_SESSIONS,
    TRANSCRIPT_KEEP_COLS,
    TRANSCRIPT_SPEAKER_COL,
    TRANSCRIPT_START_TIME_COL,
    TRANSCRIPT_END_TIME_COL,
    TRANSCRIPT_VALUE_COL,
)
from .utils import get_logger, get_session_dir, get_transcript_csv_path, load_csv

logger = get_logger(__name__)


def clean_transcript_session(
    session_id: int,
    data_root: Path,
    phq_score: float | None,
    phq_binary: int | None,
) -> tuple[pd.DataFrame | None, dict]:
    report = {
        "participant_id": session_id,
        "original_utterances": 0,
        "participant_utterances": 0,
        "phq_score": phq_score,
        "phq_binary": phq_binary,
        "status": "ok",
    }

    if session_id in EXCLUDED_SESSIONS:
        report["status"] = "excluded"
        return None, report

    session_dir = get_session_dir(data_root, session_id)
    csv_path = get_transcript_csv_path(session_dir, session_id)

    df = load_csv(csv_path)
    if df is None:
        report["status"] = "missing_file"
        return None, report

    df.columns = df.columns.str.strip()

    required_cols = [TRANSCRIPT_SPEAKER_COL, TRANSCRIPT_VALUE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning("Session %d transcript missing columns: %s", session_id, missing)
        report["status"] = f"missing_columns:{','.join(missing)}"
        return None, report

    report["original_utterances"] = len(df)

    participant_mask = df[TRANSCRIPT_SPEAKER_COL].str.lower() == "participant"
    df = df[participant_mask]

    if df.empty:
        report["status"] = "empty_after_filtering"
        return None, report

    cols_to_keep = [c for c in TRANSCRIPT_KEEP_COLS if c in df.columns]
    df = df[cols_to_keep].copy()

    n = len(df)
    meta = pd.DataFrame({
        "participant_id": [session_id] * n,
        "phq_score": [phq_score] * n,
        "phq_binary": [phq_binary] * n,
    })
    df = pd.concat([meta, df], axis=1)

    report["participant_utterances"] = len(df)
    return df, report
