"""
Audio feature extraction for ASR.

Extracts MFCC features from raw audio waveforms using librosa.
Supports loading from file paths or raw numpy arrays.
"""

from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

from config import AudioConfig, audio_config


def load_audio(
    path: str,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and convert to mono with the target sample rate.

    Args:
        path: Path to audio file (wav, flac, mp3, etc.).
        target_sr: Target sample rate. If None, uses config default.
        mono: Whether to convert to mono.

    Returns:
        (waveform, sample_rate) where waveform shape is (n_samples,).
    """
    if target_sr is None:
        target_sr = audio_config.sample_rate

    waveform, sr = sf.read(path)
    # Convert to mono if stereo
    if mono and waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Resample if needed
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return waveform, sr


def extract_mfcc(
    waveform: np.ndarray,
    sr: Optional[int] = None,
    n_mfcc: Optional[int] = None,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
) -> np.ndarray:
    """
    Extract MFCC features from a waveform.

    Args:
        waveform: 1-D audio signal.
        sr: Sample rate. Default from config.
        n_mfcc: Number of MFCC coefficients. Default from config.
        n_fft: FFT window size. Default from config.
        hop_length: Hop length in samples. Default from config.
        win_length: Window length in samples. Default from config.

    Returns:
        MFCC matrix of shape (time_steps, n_mfcc).
    """
    cfg = audio_config
    sr = sr or cfg.sample_rate
    n_mfcc = n_mfcc or cfg.n_mfcc
    n_fft = n_fft or cfg.n_fft
    hop_length = hop_length or cfg.hop_length
    win_length = win_length or cfg.win_length

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    # Transpose to (time, features) – the convention for RNNs
    return mfcc.T  # shape: (T, n_mfcc)


def add_delta_deltas(features: np.ndarray) -> np.ndarray:
    """
    Append delta and delta-delta features.

    Args:
        features: (T, D) array (e.g. MFCCs).

    Returns:
        (T, 3*D) array with [MFCC, delta(MFCC), delta-delta(MFCC)].
    """
    delta = librosa.feature.delta(features.T)
    delta2 = librosa.feature.delta(features.T, order=2)
    # Stack along feature dimension
    return np.concatenate([features.T, delta, delta2], axis=0).T  # (T, 3*D)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Per-feature mean & variance normalisation (per utterance).

    Args:
        features: (T, D) array.

    Returns:
        Normalised (T, D) array (zero mean, unit variance).
    """
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True) + 1e-10
    return (features - mean) / std


def pad_or_truncate(
    features: np.ndarray,
    max_time_steps: int,
) -> np.ndarray:
    """
    Pad (with zeros) or truncate features to a fixed time length.

    Args:
        features: (T, D) array.
        max_time_steps: Desired number of time steps.

    Returns:
        (max_time_steps, D) array.
    """
    T, D = features.shape
    if T >= max_time_steps:
        return features[:max_time_steps, :]
    # Pad
    pad_width = max_time_steps - T
    return np.pad(features, ((0, pad_width), (0, 0)), mode="constant")


def process_file(
    path: str,
    use_deltas: bool = False,
    normalize: bool = True,
    max_time_steps: Optional[int] = None,
) -> np.ndarray:
    """
    Full pipeline: load audio -> MFCC -> (optional deltas) -> (optional norm)
    -> (optional pad/truncate).

    Args:
        path: Path to audio file.
        use_deltas: Whether to append delta & delta-delta features.
        normalize: Whether to apply per-utterance mean-variance norm.
        max_time_steps: If set, pad/truncate to this many time steps.

    Returns:
        Feature matrix of shape (T', D') or (max_time_steps, D').
    """
    waveform, sr = load_audio(path)
    features = extract_mfcc(waveform, sr)

    if use_deltas:
        features = add_delta_deltas(features)

    if normalize:
        features = normalize_features(features)

    if max_time_steps is not None:
        features = pad_or_truncate(features, max_time_steps)

    return features


def compute_max_time_steps(
    audio_paths: list,
    sr: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> int:
    """
    Compute the maximum number of MFCC time steps across a list of audio files.
    Useful for determining ``max_time_steps`` for padding.

    Args:
        audio_paths: List of audio file paths.
        sr: Sample rate. Default from config.
        hop_length: Hop length. Default from config.

    Returns:
        Maximum time steps across all files.
    """
    cfg = audio_config
    sr = sr or cfg.sample_rate
    hop_length = hop_length or cfg.hop_length

    max_T = 0
    for path in audio_paths:
        waveform, _ = load_audio(path, target_sr=sr)
        T = (len(waveform) + hop_length - 1) // hop_length
        if T > max_T:
            max_T = T
    return max_T
