import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_session_dir(data_root: Path, session_id: int) -> Path:
    return data_root / f"{session_id}_P"


def get_openface_csv_path(session_dir: Path, session_id: int) -> Path:
    return session_dir / "features" / f"{session_id}_OpenFace2.1.0_Pose_gaze_AUs.csv"


def get_audio_wav_path(session_dir: Path, session_id: int) -> Path:
    return session_dir / f"{session_id}_AUDIO.wav"


def get_egemaps_csv_path(session_dir: Path, session_id: int) -> Path:
    return session_dir / "features" / f"{session_id}_OpenSMILE2.3.0_egemaps.csv"


def get_cnn_mat_path(session_dir: Path, session_id: int, variant: str = "ResNet") -> Path:
    return session_dir / "features" / f"{session_id}_CNN_{variant}.mat"


def get_transcript_csv_path(session_dir: Path, session_id: int) -> Path:
    return session_dir / f"{session_id}_TRANSCRIPT.csv"


def get_spectrogram_csv_path(session_dir: Path, session_id: int, variant: str) -> Path:
    return session_dir / "features" / f"{session_id}_{variant}.csv"


def load_mat(path: Path) -> dict | None:
    logger = get_logger(__name__)
    try:
        return sio.loadmat(str(path))
    except FileNotFoundError:
        logger.error("File not found: %s", path)
    except Exception as e:
        logger.error("Failed to load .mat %s: %s", path, e)
    return None


def load_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    logger = get_logger(__name__)
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        logger.error("File not found: %s", path)
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
    return None


def list_session_ids(data_root: Path) -> list[int]:
    ids = []
    for p in data_root.iterdir():
        if p.is_dir() and p.name.endswith("_P"):
            try:
                ids.append(int(p.name.split("_")[0]))
            except ValueError:
                pass
    return sorted(ids)
