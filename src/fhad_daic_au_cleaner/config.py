from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

_DATASET_ROOT = PROJECT_ROOT
for candidate in (PROJECT_ROOT, PROJECT_ROOT.parent):
    if (candidate / "data").exists() and (candidate / "labels").exists():
        _DATASET_ROOT = candidate
        break

DATA_ROOT = _DATASET_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output"

TRAIN_LABEL_CSV = _DATASET_ROOT / "labels" / "train_split_Depression_AVEC2017.csv"
DEV_LABEL_CSV = _DATASET_ROOT / "labels" / "dev_split_Depression_AVEC2017.csv"

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

FEATURE_COLS = ["frame", "timestamp"] + AU_REGRESSION_COLS + AU_BINARY_COLS + POSE_COLS

METADATA_COLS = ["participant_id", "phq_score", "phq_binary"]

EGEMAPS_SENTINEL = -201

EGEMAPS_FEATURE_COLS = [
    "Loudness_sma3",
    "alphaRatio_sma3",
    "hammarbergIndex_sma3",
    "slope0-500_sma3",
    "slope500-1500_sma3",
    "spectralFlux_sma3",
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
    "F0semitoneFrom27.5Hz_sma3nz",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
    "F1frequency_sma3nz",
    "F1bandwidth_sma3nz",
    "F1amplitudeLogRelF0_sma3nz",
    "F2frequency_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3frequency_sma3nz",
    "F3amplitudeLogRelF0_sma3nz",
]

EGEMAPS_KEEP_COLS = ["frameTime"] + EGEMAPS_FEATURE_COLS

CNN_RESNET_FILE_TEMPLATE = "{sid}_CNN_ResNet.mat"
CNN_VGG_FILE_TEMPLATE = "{sid}_CNN_VGG.mat"

CNN_RESNET_COL_PREFIX = "cnn_resnet"
CNN_VGG_COL_PREFIX = "cnn_vgg"

DENSENET_FILE_TEMPLATE = "{sid}_densenet201.csv"
VGG16_AUDIO_FILE_TEMPLATE = "{sid}_vgg16.csv"

SPECTROGRAM_DENSENET_COL_PREFIX = "dsn"
SPECTROGRAM_VGG16_COL_PREFIX = "vgg16"
