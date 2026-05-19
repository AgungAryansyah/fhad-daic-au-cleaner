from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output"

TRAIN_LABEL_CSV = PROJECT_ROOT / "labels/train_split_Depression_AVEC2017.csv"
DEV_LABEL_CSV = PROJECT_ROOT / "labels/dev_split_Depression_AVEC2017.csv"

EXCLUDED_SESSIONS = {342, 394, 398, 460, 373, 444, 451, 458, 480, 402}

CONFIDENCE_THRESHOLD = 0.5

LABEL_PARTICIPANT_COL = "Participant_ID"
LABEL_PHQ_SCORE_COL = "PHQ_Score"
LABEL_PHQ_BINARY_COL = "PHQ_Binary"

AU_REGRESSION_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

AU_BINARY_COLS = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]

POSE_COLS = [
    "pose_Rx", "pose_Ry", "pose_Rz",
    "pose_Tx", "pose_Ty", "pose_Tz",
]

FEATURE_COLS = AU_REGRESSION_COLS + AU_BINARY_COLS + POSE_COLS

METADATA_COLS = ["participant_id", "phq_score", "phq_binary"]
