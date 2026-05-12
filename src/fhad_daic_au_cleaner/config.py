from pathlib import Path

DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("output")

TRAIN_LABEL_CSV = Path("labels/train_split_Depression_AVEC2017.csv")
DEV_LABEL_CSV = Path("labels/dev_split_Depression_AVEC2017.csv")

EXCLUDED_SESSIONS = {342, 394, 398, 460, 373, 444, 451, 458, 480, 402}

CONFIDENCE_THRESHOLD = 0.5

AU_REGRESSION_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r",
    "AU17_r", "AU20_r", "AU25_r", "AU26_r",
]

AU_BINARY_COLS = [
    "AU04_c", "AU12_c", "AU15_c", "AU23_c", "AU28_c", "AU45_c",
]

POSE_COLS = [
    "pose_Rx", "pose_Ry", "pose_Rz",
    "pose_Tx", "pose_Ty", "pose_Tz",
]

FEATURE_COLS = AU_REGRESSION_COLS + AU_BINARY_COLS + POSE_COLS

METADATA_COLS = ["participant_id", "phq8_score", "phq8_binary"]
