import argparse
from pathlib import Path

import pandas as pd

from .config import (
    AU_REGRESSION_COLS,
    CONFIDENCE_THRESHOLD,
    DATA_ROOT,
    DEV_LABEL_CSV,
    EXCLUDED_SESSIONS,
    TRAIN_LABEL_CSV,
)
from .utils import (
    get_logger,
    get_openface_csv_path,
    get_session_dir,
    list_session_ids,
    load_csv,
)

logger = get_logger(__name__)


def explore(data_root: Path, train_label: Path, dev_label: Path) -> None:
    print("\n" + "=" * 60)
    print("DAIC-WOZ Dataset Exploration Report")
    print("=" * 60)

    session_ids = list_session_ids(data_root)
    print(f"\n[Session Folders]")
    print(f"  Found: {len(session_ids)} sessions")
    print(f"  IDs:   {session_ids}")

    active_ids = [s for s in session_ids if s not in EXCLUDED_SESSIONS]
    excluded_found = [s for s in session_ids if s in EXCLUDED_SESSIONS]
    print(f"  Excluded (found in data): {excluded_found}")
    print(f"  Active sessions: {len(active_ids)}")

    print(f"\n[File Presence Check]")
    missing_openface = []
    for sid in active_ids:
        sdir = get_session_dir(data_root, sid)
        csv_path = get_openface_csv_path(sdir, sid)
        if not csv_path.exists():
            missing_openface.append(sid)

    if missing_openface:
        print(f"  Sessions missing OpenFace CSV: {missing_openface}")
    else:
        print(f"  All active sessions have OpenFace CSV.")

    print(f"\n[Sample OpenFace CSV]")
    sample_id = active_ids[0] if active_ids else None
    if sample_id:
        sdir = get_session_dir(data_root, sample_id)
        csv_path = get_openface_csv_path(sdir, sample_id)
        df = load_csv(csv_path)
        if df is not None:
            df.columns = df.columns.str.strip()
            print(f"  Session {sample_id}: {len(df)} rows x {len(df.columns)} cols")
            print(f"  Columns: {list(df.columns)}")
            if "confidence" in df.columns:
                c = df["confidence"]
                print(f"  confidence — min: {c.min():.3f}, mean: {c.mean():.3f}, max: {c.max():.3f}")
                below_thresh = (c < CONFIDENCE_THRESHOLD).sum()
                print(f"  Frames below confidence {CONFIDENCE_THRESHOLD}: {below_thresh} ({100*below_thresh/len(df):.1f}%)")
            au_cols_present = [c for c in AU_REGRESSION_COLS if c in df.columns]
            missing_au = [c for c in AU_REGRESSION_COLS if c not in df.columns]
            print(f"  AU regression cols present: {len(au_cols_present)}/{len(AU_REGRESSION_COLS)}")
            if missing_au:
                print(f"  Missing AU cols: {missing_au}")
            zero_rows = (df[au_cols_present] == 0).all(axis=1).sum() if au_cols_present else "N/A"
            print(f"  All-zero AU rows (potential scrubbed frames): {zero_rows}")

    print(f"\n[Confidence Distribution — All Active Sessions]")
    low_conf_counts = {}
    for sid in active_ids:
        sdir = get_session_dir(data_root, sid)
        csv_path = get_openface_csv_path(sdir, sid)
        df = load_csv(csv_path)
        if df is not None:
            df.columns = df.columns.str.strip()
            if "confidence" in df.columns:
                low_conf_counts[sid] = (df["confidence"] < CONFIDENCE_THRESHOLD).sum()

    if low_conf_counts:
        total_low = sum(low_conf_counts.values())
        sessions_with_low = sum(1 for v in low_conf_counts.values() if v > 0)
        print(f"  Sessions with low-confidence frames: {sessions_with_low}/{len(active_ids)}")
        print(f"  Total low-confidence frames across all sessions: {total_low}")
        top5 = sorted(low_conf_counts.items(), key=lambda x: -x[1])[:5]
        print(f"  Top 5 sessions by low-confidence frame count: {top5}")

    print(f"\n[Label Files]")
    for label_path, split in [(train_label, "train"), (dev_label, "dev")]:
        df = load_csv(label_path)
        if df is not None:
            print(f"  {split}: {len(df)} rows, columns: {list(df.columns)}")
        else:
            print(f"  {split}: FAILED to load from {label_path}")

    print("\n" + "=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--train-label", type=Path, default=TRAIN_LABEL_CSV)
    parser.add_argument("--dev-label", type=Path, default=DEV_LABEL_CSV)
    args = parser.parse_args()
    explore(args.data_root, args.train_label, args.dev_label)


if __name__ == "__main__":
    main()
