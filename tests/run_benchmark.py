"""
Reproducible Case Study Benchmark Script.

Generates clean and noisy 16 kHz WAV test audio (using Windows SAPI5 voices)
for 8 standard assistant commands, runs the full assistant pipeline
(Audio Input -> ASR -> NLP -> TTS -> WER), and prints measured metrics.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.audio_input import analyze_wav_bytes
from modules.nlp_processor import IntentClassifier
from modules.speech_recognition import transcribe_wav_bytes
from modules.speech_synthesis import synthesize_speech
from modules.wer_evaluator import calculate_wer

TEST_COMMANDS = [
    ("open youtube", "open_website"),
    ("search youtube for python tutorial", "search_youtube"),
    ("open google", "open_website"),
    ("open notepad", "open_app"),
    ("open calculator", "open_app"),
    ("increase the volume", "volume_up"),
    ("decrease the volume", "volume_down"),
    ("set volume to 50 percent", "volume_set"),
]


def synthesize_test_wav(text: str, out_path: Path) -> None:
    """Use Windows SAPI5 to render a 16 kHz mono 16-bit WAV file for testing."""
    ps_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        "16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{out_path}', $fmt); "
        f"$s.Speak('{text}'); "
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True)


def add_background_noise(wav_bytes: bytes, noise_level: float = 0.04) -> bytes:
    """Mix mild Gaussian noise into WAV bytes to simulate noisy environment."""
    rng = np.random.default_rng(42)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        params = wf.getparams()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    noisy = np.clip(pcm + rng.normal(0.0, noise_level, size=pcm.shape), -1.0, 1.0)
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setparams(params)
        wf.writeframes((noisy * 32767).astype(np.int16).tobytes())
    return out.getvalue()


def main() -> None:
    if sys.platform != "win32":
        print("System.Speech benchmark generation requires Windows.")
        return

    classifier = IntentClassifier(ROOT / "data" / "intents.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        conditions = [("Clean", 0.0), ("Mild Noise (SNR ~14dB)", 0.05), ("Heavy Noise (SNR ~4dB)", 0.16)]
        for condition, noise_lvl in conditions:
            print(f"\n=== Condition: {condition} ===")
            print(f"{'Reference':35} | {'Recognized':35} | {'WER%':6} | {'Intent OK':9} | {'ASR ms':7} | {'NLP ms':7} | {'TTS ms':7}")
            print("-" * 120)

            wers = []
            intent_hits = 0
            asr_latencies = []
            nlp_latencies = []
            tts_latencies = []

            for i, (phrase, expected_intent) in enumerate(TEST_COMMANDS):
                wav_file = Path(tmpdir) / f"cmd_{i}.wav"
                synthesize_test_wav(phrase, wav_file)
                wav_bytes = wav_file.read_bytes()
                if noise_lvl > 0:
                    wav_bytes = add_background_noise(wav_bytes, noise_level=noise_lvl)

                _audio = analyze_wav_bytes(wav_bytes)
                asr = transcribe_wav_bytes(wav_bytes, language="en-US")
                hyp = asr.transcript if asr.success else ""
                wer_res = calculate_wer(phrase, hyp)
                nlp = classifier.predict(hyp) if hyp else None
                intent_ok = bool(nlp and nlp.intent == expected_intent)
                tts = synthesize_speech(f"Running {phrase}" if i == 0 else "Okay") if i < 2 else None

                wers.append(wer_res.wer_percent)
                if intent_ok:
                    intent_hits += 1
                asr_latencies.append(asr.latency_ms)
                if nlp:
                    nlp_latencies.append(nlp.latency_ms)
                if tts and tts.success:
                    tts_latencies.append(tts.latency_ms)

                tts_str = f"{tts.latency_ms:.1f}" if (tts and tts.success) else "-"
                nlp_str = f"{nlp.latency_ms:.2f}" if nlp else "-"
                print(
                    f"{phrase[:35]:35} | {(hyp or '<none>')[:35]:35} | {wer_res.wer_percent:6.2f} | "
                    f"{str(intent_ok):9} | {asr.latency_ms:7.1f} | {nlp_str:7} | {tts_str:7}"
                )

            avg_wer = sum(wers) / len(wers)
            avg_asr = sum(asr_latencies) / len(asr_latencies)
            avg_nlp = sum(nlp_latencies) / len(nlp_latencies) if nlp_latencies else 0.0
            print(
                f"Summary ({condition}): Avg WER = {avg_wer:.2f}% | "
                f"Intent Accuracy = {intent_hits}/{len(TEST_COMMANDS)} ({intent_hits/len(TEST_COMMANDS)*100:.1f}%) | "
                f"Avg ASR Latency = {avg_asr:.1f} ms | Avg NLP Latency = {avg_nlp:.2f} ms"
            )


if __name__ == "__main__":
    main()
