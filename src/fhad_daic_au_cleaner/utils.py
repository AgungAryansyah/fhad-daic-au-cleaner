import logging
from pathlib import Path

import pandas as pd


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
