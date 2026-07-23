from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from torchaudio.functional import resample

from .config import EXCLUDED_SESSIONS, WAV2VEC_CHUNK_SECONDS, WAV2VEC_COL_PREFIX, WAV2VEC_MODEL_ID
from .utils import get_audio_wav_path, get_logger, get_session_dir

logger = get_logger(__name__)


def _load_wav2vec_model(device: torch.device) -> tuple:
    from transformers import Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL_ID)
    model = model.to(device)
    model.eval()
    return model, model.config.hidden_size


def clean_wav2vec_session(
    session_id: int,
    data_root: Path,
    phq_score: float | None,
    phq_binary: int | None,
    excluded_sessions: frozenset[int] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    _excluded = excluded_sessions if excluded_sessions is not None else EXCLUDED_SESSIONS
    report = {
        "participant_id": session_id,
        "duration_seconds": 0.0,
        "embedding_frames": 0,
        "embedding_dim": 0,
        "phq_score": phq_score,
        "phq_binary": phq_binary,
        "status": "ok",
    }

    if session_id in _excluded:
        report["status"] = "excluded"
        return None, report

    session_dir = get_session_dir(data_root, session_id)
    wav_path = get_audio_wav_path(session_dir, session_id)

    if not wav_path.exists():
        report["status"] = "missing_file"
        return None, report

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, hidden_size = _load_wav2vec_model(device)

    sample_rate, data = wavfile.read(str(wav_path))

    if data.dtype.kind != "f":
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)

    if data.ndim > 1:
        data = data.mean(axis=1)

    waveform = torch.from_numpy(data).unsqueeze(0)

    if sample_rate != 16000:
        waveform = resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    report["duration_seconds"] = waveform.shape[1] / sample_rate

    chunk_samples = WAV2VEC_CHUNK_SECONDS * sample_rate
    total_samples = waveform.shape[1]
    embeddings = []

    with torch.no_grad():
        for start in range(0, total_samples, chunk_samples):
            chunk = waveform[:, start : start + chunk_samples].to(device)
            if chunk.shape[1] < 400:
                continue
            output = model(chunk, output_hidden_states=True)
            hidden = output.last_hidden_state.squeeze(0)
            embeddings.append(hidden.cpu().numpy())

    if not embeddings:
        report["status"] = "empty_after_cleaning"
        return None, report

    all_embeddings = np.concatenate(embeddings, axis=0)
    n_frames = all_embeddings.shape[0]

    col_names = [f"{WAV2VEC_COL_PREFIX}_{i}" for i in range(hidden_size)]
    df = pd.DataFrame(all_embeddings, columns=col_names)

    meta = pd.DataFrame({
        "participant_id": [session_id] * n_frames,
        "phq_score": [phq_score] * n_frames,
        "phq_binary": [phq_binary] * n_frames,
    })
    df = pd.concat([meta, df], axis=1)

    report["embedding_frames"] = n_frames
    report["embedding_dim"] = hidden_size
    return df, report
