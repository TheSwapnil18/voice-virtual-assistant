# Voice-Based Virtual Assistant (Case Study Project)

A small, clean, college-level voice assistant built with Python and FastAPI that demonstrates the complete speech pipeline in a transparent UI:

1. **Audio Input** (`modules/audio_input.py`) — Captures 16 kHz, 16-bit mono PCM `.wav` directly in the browser (zero `ffmpeg` needed) and measures duration, RMS loudness, dBFS, and waveform peaks.
2. **Speech Recognition / ASR** (`modules/speech_recognition.py`) — Transcribes spoken WAV audio into text via `SpeechRecognition` (Google Web Speech backend) and records latency.
3. **Natural Language Processing / NLP** (`modules/nlp_processor.py`) — Uses TF-IDF vectorization + Cosine Similarity against `data/intents.json` along with regex/keyword slot extraction to classify intents and extract websites, apps, search queries, and volume percentages.
4. **Speech Synthesis / TTS & Execution** (`modules/speech_synthesis.py`, `modules/task_executor.py`) — Generates spoken MP3 replies via `gTTS` and performs browser tasks (YouTube, Google, Wikipedia, etc.) plus real Windows OS tasks (Notepad, Calculator, Paint, Chrome, Volume Up/Down/Mute/Set) when running locally on Windows.
5. **Word Error Rate / WER Lab** (`modules/wer_evaluator.py`) — Implements word-level Levenshtein dynamic programming to calculate Substitutions ($S$), Deletions ($D$), Insertions ($I$), Reference Words ($N$), and $\text{WER} = \frac{S + D + I}{N}$, with visual word alignment and CSV export.

---

## Project Structure

```text
NLP CS/
├── app.py                   # FastAPI web server & REST endpoints
├── data/intents.json        # Small training dataset of voice command examples
├── modules/
│   ├── audio_input.py       # WAV parsing, loudness (dBFS), waveform peaks
│   ├── speech_recognition.py# Audio-to-text transcription
│   ├── nlp_processor.py     # TF-IDF intent matching + slot extraction
│   ├── speech_synthesis.py  # Text-to-speech MP3 generation (gTTS)
│   ├── task_executor.py     # Web URL builder & Windows apps/volume control
│   └── wer_evaluator.py     # Dynamic programming WER & alignment
├── static/index.html        # Clean, non-technical friendly web interface
├── tests/
│   ├── test_pipeline.py     # Automated unit & API tests
│   └── run_benchmark.py     # Reproducible clean vs noisy WER benchmark
├── CASE_STUDY_REPORT.md     # Complete academic case study report
├── start_assistant.bat      # One-click Windows launcher
├── .python-version          # Pins Python 3.11.11 for cloud deployment (Render)
└── requirements.txt         # Cross-platform dependencies
```

---

## How to Run Locally on Windows

> **Note:** Python 3.11 is recommended (especially for `SpeechRecognition`, since Python 3.13+ removed `aifc`/`audioop`).

Double-click `start_assistant.bat`, or run in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in Chrome or Edge.

### Run Tests & WER Benchmark
```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe tests\run_benchmark.py
```

---

## How to Create Your Global Link (100% Free, No Docker Needed)

### Option 1: Free Permanent Link on Render (Native Python 3)
1. Upload all project files (except `.venv/`, `__pycache__/`, and `.pytest_cache/`) to a GitHub repository. (`.python-version` pins Python `3.11.11` automatically.)
2. Go to [render.com](https://render.com), log in with GitHub, and click **New → Web Service**.
3. Select your repository and configure:
   - **Language / Runtime:** `Python 3` (not Docker)
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
4. Click **Deploy Web Service**. You get a permanent HTTPS link like `https://voice-assistant-xxxx.onrender.com`.
   - Web commands (`Open YouTube`, `Search YouTube`, `Open Google`), voice ASR, NLP, TTS, and WER Check work live for everyone.
   - Windows-only commands (`Open Notepad`, `Volume Up`) automatically show **Cloud Demo** simulation mode.
   *(Note: Render's free tier sleeps after 15 minutes of inactivity and wakes in ~30–50 seconds on the first visit.)*

### Option 2: Instant Free Global Link Controlling Your Windows PC (Live Presentation)
If you want your global URL to open real apps and change volume on your Windows laptop during a live demo (no GitHub or Docker needed):
1. Double-click `start_assistant.bat` so the assistant is running on `http://127.0.0.1:8000`.
2. Install Cloudflare Tunnel once: `winget install Cloudflare.cloudflared`
3. In a new terminal, run:
   ```powershell
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
4. Share the generated `https://*.trycloudflare.com` URL (HTTPS enables browser microphone access from anywhere).
