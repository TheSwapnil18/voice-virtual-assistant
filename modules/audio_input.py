"""
Step 1: Audio Input Analysis Module.

Inspects raw WAV audio bytes recorded by the user's microphone:
- Sample rate (Hz), bit depth, channels, and duration
- Root Mean Square (RMS) loudness & decibels (dBFS)
- Downsampled amplitude peaks for simple waveform visualization
"""

from __future__ import annotations

import io
import wave
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class AudioAnalysis:
    sample_rate_hz: int
    channels: int
    sample_width_bits: int
    duration_sec: float
    rms_level: float
    db_fs: float
    quality_label: str
    waveform_peaks: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_loudness(rms: float) -> str:
    """Return a plain-English label for microphone loudness."""
    if rms < 0.008:
        return "Too quiet (try speaking closer to mic)"
    if rms > 0.45:
        return "Very loud (possible clipping)"
    return "Good"


def _compute_peaks(samples: np.ndarray, num_bars: int = 36) -> list[float]:
    """Downsample normalized audio into small amplitude bars for the UI."""
    if len(samples) == 0:
        return [0.0] * num_bars
    abs_samples = np.abs(samples)
    chunks = np.array_split(abs_samples, num_bars)
    return [round(float(np.max(chunk)), 3) if len(chunk) else 0.0 for chunk in chunks]


def analyze_wav_bytes(wav_bytes: bytes) -> AudioAnalysis:
    """Parse WAV bytes and extract signal characteristics for the case study."""
    if not wav_bytes:
        raise ValueError("Empty audio data received.")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_frames = wf.readframes(n_frames)

    if n_frames == 0 or sample_rate == 0:
        raise ValueError("Audio recording has zero duration.")

    duration = round(n_frames / float(sample_rate), 2)

    # Convert PCM frames to normalized [-1.0, 1.0] float array
    if sample_width == 2:
        pcm = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        pcm = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        pcm = np.zeros(n_frames, dtype=np.float32)

    if channels > 1 and len(pcm) >= channels:
        pcm = pcm.reshape(-1, channels).mean(axis=1)

    rms = float(np.sqrt(np.mean(pcm**2))) if len(pcm) else 0.0
    db_fs = round(float(20.0 * np.log10(max(rms, 1e-6))), 1)

    return AudioAnalysis(
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width_bits=sample_width * 8,
        duration_sec=duration,
        rms_level=round(rms, 4),
        db_fs=db_fs,
        quality_label=_classify_loudness(rms),
        waveform_peaks=_compute_peaks(pcm),
    )
