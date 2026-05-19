from pathlib import Path

import pandas as pd

from .config import (
    AU_REGRESSION_COLS,
    CONFIDENCE_THRESHOLD,
    EXCLUDED_SESSIONS,
)
from .utils import (
    get_logger,
    get_openface_csv_path,
    get_session_dir,
    list_session_ids,
    load_csv,
)

logger = get_logger(__name__)


def diagnose_session(session_id: int, data_root: Path) -> dict:
    result = {
        "participant_id": session_id,
        "reason": None,
        "original_frames": None,
        "after_confidence_filter": None,
        "after_zero_au_filter": None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "pct_low_confidence": None,
        "pct_zero_au": None,
    }

    if session_id in EXCLUDED_SESSIONS:
        result["reason"] = "excluded_by_config"
        return result

    session_dir = get_session_dir(data_root, session_id)
    csv_path = get_openface_csv_path(session_dir, session_id)

    if not csv_path.exists():
        result["reason"] = "missing_openface_csv"
        return result

    df = load_csv(csv_path)
    if df is None:
        result["reason"] = "csv_load_error"
        return result

    df.columns = df.columns.str.strip()
    result["original_frames"] = len(df)

    required_cols = ["confidence"] + AU_REGRESSION_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        result["reason"] = f"missing_columns:{','.join(missing)}"
        return result

    low_conf_mask = df["confidence"] < CONFIDENCE_THRESHOLD
    n_low_conf = int(low_conf_mask.sum())
    result["pct_low_confidence"] = round(100 * n_low_conf / len(df), 2)
    df = df[~low_conf_mask]
    result["after_confidence_filter"] = len(df)

    if df.empty:
        result["reason"] = "all_frames_dropped_by_confidence"
        return result

    au_cols_present = [c for c in AU_REGRESSION_COLS if c in df.columns]
    zero_au_mask = (df[au_cols_present] == 0).all(axis=1)
    n_zero_au = int(zero_au_mask.sum())
    result["pct_zero_au"] = round(100 * n_zero_au / len(df), 2)
    df = df[~zero_au_mask]
    result["after_zero_au_filter"] = len(df)

    if df.empty:
        result["reason"] = "all_frames_dropped_by_zero_au"
        return result

    result["reason"] = "ok"
    return result


def run_diagnosis(data_root: Path, output_root: Path) -> pd.DataFrame:
    session_ids = list_session_ids(data_root)
    logger.info("Found %d session folders to diagnose.", len(session_ids))

    results = []
    for sid in session_ids:
        cleaned_path = None
        for split in ("train", "dev", "test"):
            p = output_root / split / f"{sid}_clean.csv"
            if p.exists():
                cleaned_path = p
                break

        is_empty_output = (
            cleaned_path is not None and cleaned_path.stat().st_size == 0
        )
        is_missing_output = cleaned_path is None

        if is_empty_output or is_missing_output:
            logger.info("Diagnosing session %d (output: %s)", sid,
                        "empty" if is_empty_output else "missing")
            result = diagnose_session(sid, data_root)
            result["output_status"] = "empty" if is_empty_output else "missing"
        else:
            result = {
                "participant_id": sid,
                "reason": "ok",
                "output_status": "present",
                "original_frames": None,
                "after_confidence_filter": None,
                "after_zero_au_filter": None,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "pct_low_confidence": None,
                "pct_zero_au": None,
            }

        results.append(result)

    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("Diagnosis Summary")
    print("=" * 60)
    print(f"Total sessions: {len(df)}")
    print(f"\nOutput status breakdown:")
    print(df["output_status"].value_counts().to_string())
    print(f"\nReason breakdown (non-ok only):")
    non_ok = df[df["reason"] != "ok"]
    if non_ok.empty:
        print("  All sessions produced valid output.")
    else:
        print(non_ok["reason"].value_counts().to_string())
        print(f"\nPer-session details:")
        print(non_ok[["participant_id", "reason", "original_frames",
                       "pct_low_confidence", "pct_zero_au"]].to_string(index=False))
    print("=" * 60 + "\n")

    return df
