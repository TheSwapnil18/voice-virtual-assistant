"""
Step 4: Speech Synthesis (TTS) Module.

Converts the assistant's text reply into spoken MP3 audio using gTTS
and returns it base64-encoded so any browser can play it immediately.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import asdict, dataclass

from gtts import gTTS


@dataclass
class SynthesisResult:
    success: bool
    audio_base64: str
    mime_type: str
    engine: str
    latency_ms: float
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def synthesize_speech(text: str, lang: str = "en") -> SynthesisResult:
    """Convert reply text into base64-encoded MP3 audio."""
    start = time.perf_counter()
    if not text.strip():
        return SynthesisResult(
            success=False,
            audio_base64="",
            mime_type="audio/mpeg",
            engine="gTTS",
            latency_ms=0.0,
            error_message="Empty reply text.",
        )

    try:
        buffer = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        return SynthesisResult(
            success=True,
            audio_base64=encoded,
            mime_type="audio/mpeg",
            engine="gTTS (Google Text-to-Speech)",
            latency_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        return SynthesisResult(
            success=False,
            audio_base64="",
            mime_type="audio/mpeg",
            engine="gTTS (Google Text-to-Speech)",
            latency_ms=elapsed_ms,
            error_message=f"Speech synthesis unavailable: {exc}",
        )
