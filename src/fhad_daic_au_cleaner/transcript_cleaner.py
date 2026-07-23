from pathlib import Path

import pandas as pd

from .config import EXCLUDED_SESSIONS
from .utils import get_logger, get_session_dir, get_transcript_csv_path, load_csv

logger = get_logger(__name__)


def clean_transcript_session(
    session_id: int,
    data_root: Path,
    phq_score: float | None,
    phq_binary: int | None,
    excluded_sessions: frozenset[int] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    _excluded = excluded_sessions if excluded_sessions is not None else EXCLUDED_SESSIONS
    report = {
        "participant_id": session_id,
        "rows": 0,
        "phq_score": phq_score,
        "phq_binary": phq_binary,
        "status": "ok",
    }

    if session_id in _excluded:
        report["status"] = "excluded"
        return None, report

    session_dir = get_session_dir(data_root, session_id)
    csv_path = get_transcript_csv_path(session_dir, session_id)

    df = load_csv(csv_path)
    if df is None:
        report["status"] = "missing_file"
        return None, report

    df.columns = df.columns.str.strip()

    n = len(df)
    meta = pd.DataFrame({
        "participant_id": [session_id] * n,
        "phq_score": [phq_score] * n,
        "phq_binary": [phq_binary] * n,
    })
    df = pd.concat([meta, df], axis=1)

    report["rows"] = len(df)
    return df, report
