"""
Step 2: Speech Recognition (ASR) Module.

Converts recorded WAV audio into text using the SpeechRecognition library
(Google Web Speech API backend) and records processing latency.
"""

from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass

import speech_recognition as sr


@dataclass
class RecognitionResult:
    success: bool
    transcript: str
    engine: str
    latency_ms: float
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def transcribe_wav_bytes(wav_bytes: bytes, language: str = "en-US") -> RecognitionResult:
    """Transcribe WAV bytes to text and measure recognition latency."""
    start = time.perf_counter()
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language=language)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        return RecognitionResult(
            success=True,
            transcript=text.strip(),
            engine="Google Speech Recognition",
            latency_ms=elapsed_ms,
        )
    except sr.UnknownValueError:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        return RecognitionResult(
            success=False,
            transcript="",
            engine="Google Speech Recognition",
            latency_ms=elapsed_ms,
            error_message="Could not understand the audio. Please speak clearly and try again.",
        )
    except sr.RequestError as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        return RecognitionResult(
            success=False,
            transcript="",
            engine="Google Speech Recognition",
            latency_ms=elapsed_ms,
            error_message=f"Speech recognition service unavailable: {exc}",
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        return RecognitionResult(
            success=False,
            transcript="",
            engine="Google Speech Recognition",
            latency_ms=elapsed_ms,
            error_message=f"Audio decoding failed: {exc}",
        )
