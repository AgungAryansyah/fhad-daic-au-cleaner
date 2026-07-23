from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CNN_RESNET_COL_PREFIX,
    CNN_VGG_COL_PREFIX,
    EXCLUDED_SESSIONS,
)
from .utils import get_cnn_mat_path, get_logger, get_session_dir, load_mat

logger = get_logger(__name__)


def _find_embedding_array(mat_data: dict) -> tuple[str, np.ndarray] | None:
    arrays = []
    for key in sorted(mat_data.keys()):
        if key.startswith("__"):
            continue
        val = mat_data[key]
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.size > 0:
            arrays.append((key, val))
    if not arrays:
        return None
    arrays.sort(key=lambda x: x[1].size, reverse=True)
    return arrays[0]


def clean_cnn_session(
    session_id: int,
    data_root: Path,
    phq_score: float | None,
    phq_binary: int | None,
    variant: str = "ResNet",
    col_prefix: str | None = None,
    excluded_sessions: frozenset[int] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    _excluded = excluded_sessions if excluded_sessions is not None else EXCLUDED_SESSIONS
    if col_prefix is None:
        col_prefix = CNN_RESNET_COL_PREFIX if variant == "ResNet" else CNN_VGG_COL_PREFIX

    report = {
        "participant_id": session_id,
        "modality": f"cnn_{variant.lower()}",
        "original_frames": 0,
        "final_frames": 0,
        "embedding_dim": 0,
        "phq_score": phq_score,
        "phq_binary": phq_binary,
        "status": "ok",
    }

    if session_id in _excluded:
        report["status"] = "excluded"
        return None, report

    session_dir = get_session_dir(data_root, session_id)
    mat_path = get_cnn_mat_path(session_dir, session_id, variant=variant)

    mat_data = load_mat(mat_path)
    if mat_data is None:
        report["status"] = "missing_file"
        return None, report

    result = _find_embedding_array(mat_data)
    if result is None:
        logger.warning("Session %d CNN %s: no valid 2D array found in .mat", session_id, variant)
        report["status"] = "no_valid_array"
        return None, report

    key_name, arr = result
    n_frames, embedding_dim = arr.shape
    report["original_frames"] = n_frames
    report["final_frames"] = n_frames
    report["embedding_dim"] = embedding_dim

    cols = [f"{col_prefix}_{i}" for i in range(embedding_dim)]
    df = pd.DataFrame(arr, columns=cols)

    meta = pd.DataFrame({
        "participant_id": [session_id] * n_frames,
        "phq_score": [phq_score] * n_frames,
        "phq_binary": [phq_binary] * n_frames,
        "frame": range(n_frames),
    })
    df = pd.concat([meta, df], axis=1)

    return df, report
