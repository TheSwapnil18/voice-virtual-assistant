"""
Step 4b: Task Executor Module.

Handles two categories of tasks cleanly:
1. Web Tasks (YouTube, Google, Wikipedia, etc.) -> returns a URL for the browser to open.
2. Windows OS Tasks (Apps & System Volume) -> runs natively on Windows via pycaw/os.startfile,
   or safely simulates when hosted globally on a Linux cloud server.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from urllib.parse import quote_plus

IS_WINDOWS = sys.platform == "win32"

WEBSITE_URLS = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "wikipedia": "https://www.wikipedia.org",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "spotify": "https://open.spotify.com",
    "chatgpt": "https://chat.openai.com",
    "instagram": "https://www.instagram.com",
    "linkedin": "https://www.linkedin.com",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "stackoverflow": "https://stackoverflow.com",
    "reddit": "https://www.reddit.com",
    "netflix": "https://www.netflix.com",
    "amazon": "https://www.amazon.com",
}

# Safe whitelist of common Windows apps so arbitrary shell commands are never executed
WINDOWS_APPS = {
    "notepad": ("Notepad", "notepad.exe"),
    "calculator": ("Calculator", "calc.exe"),
    "calc": ("Calculator", "calc.exe"),
    "paint": ("Paint", "mspaint.exe"),
    "mspaint": ("Paint", "mspaint.exe"),
    "cmd": ("Command Prompt", "cmd.exe"),
    "command prompt": ("Command Prompt", "cmd.exe"),
    "terminal": ("Command Prompt", "cmd.exe"),
    "explorer": ("File Explorer", "explorer.exe"),
    "file explorer": ("File Explorer", "explorer.exe"),
    "files": ("File Explorer", "explorer.exe"),
    "task manager": ("Task Manager", "taskmgr.exe"),
    "control panel": ("Control Panel", "control.exe"),
    "snipping tool": ("Snipping Tool", "SnippingTool.exe"),
    "settings": ("Windows Settings", "ms-settings:"),
    "windows settings": ("Windows Settings", "ms-settings:"),
    "clock": ("Clock", "ms-clock:"),
    "chrome": ("Google Chrome", "chrome"),
    "google chrome": ("Google Chrome", "chrome"),
    "edge": ("Microsoft Edge", "msedge"),
    "microsoft edge": ("Microsoft Edge", "msedge"),
}


@dataclass
class ExecutionResult:
    action_type: str
    status: str
    reply_text: str
    details: str
    target_url: str = ""
    simulated: bool = False
    volume_percent: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def is_local_windows_mode() -> bool:
    """Return True when running on Windows and cloud simulation is not forced."""
    force_sim = os.getenv("SIMULATE_WINDOWS", "").lower() in {"1", "true", "yes"}
    return IS_WINDOWS and not force_sim


def _adjust_windows_volume(mode: str, target_percent: int | None = None) -> tuple[bool, int | None, str]:
    """Adjust Windows master audio volume using pycaw Core Audio APIs."""
    import comtypes
    from pycaw.pycaw import AudioUtilities

    comtypes.CoInitialize()
    try:
        speaker = AudioUtilities.GetSpeakers()
        endpoint = getattr(speaker, "EndpointVolume", None)
        if endpoint is None:
            return False, None, "Audio endpoint not available."

        current = round(float(endpoint.GetMasterVolumeLevelScalar()) * 100)

        if mode == "up":
            new_pct = min(100, current + 10)
            endpoint.SetMute(0, None)
            endpoint.SetMasterVolumeLevelScalar(new_pct / 100.0, None)
            return True, new_pct, f"Windows volume raised from {current}% to {new_pct}%."

        if mode == "down":
            new_pct = max(0, current - 10)
            endpoint.SetMasterVolumeLevelScalar(new_pct / 100.0, None)
            return True, new_pct, f"Windows volume lowered from {current}% to {new_pct}%."

        if mode == "set" and target_percent is not None:
            new_pct = max(0, min(100, int(target_percent)))
            if new_pct > 0:
                endpoint.SetMute(0, None)
            endpoint.SetMasterVolumeLevelScalar(new_pct / 100.0, None)
            return True, new_pct, f"Windows volume set to {new_pct}%."

        if mode == "mute":
            endpoint.SetMute(1, None)
            return True, current, "Windows system audio muted."

        if mode == "unmute":
            endpoint.SetMute(0, None)
            return True, current, "Windows system audio unmuted."

        return False, current, "Unsupported volume action."
    except Exception as exc:
        return False, None, f"Could not change Windows volume: {exc}"
    finally:
        comtypes.CoUninitialize()


def _launch_windows_app(app_key: str) -> tuple[bool, str, str]:
    """Launch a whitelisted Windows application cleanly in a visible window."""
    app_info = WINDOWS_APPS.get(app_key.lower().strip())
    if not app_info:
        return False, app_key, f"'{app_key}' is not in the safe Windows app list."

    display_name, command = app_info
    try:
        if command == "cmd.exe":
            subprocess.Popen(["cmd.exe"], creationflags=subprocess.CREATE_NEW_CONSOLE)
            return True, display_name, f"Launched {display_name} ({command}) on Windows."

        import ctypes

        # SW_SHOWNORMAL (1) prevents hidden-window inheritance when started from tools
        rc = ctypes.windll.shell32.ShellExecuteW(None, "open", command, None, None, 1)
        if rc <= 32:
            if rc in (2, 3):
                return False, display_name, f"{display_name} was not found on this PC."
            return False, display_name, f"Could not launch {display_name} (error code {rc})."
        return True, display_name, f"Launched {display_name} ({command}) on Windows."
    except FileNotFoundError:
        return False, display_name, f"{display_name} was not found on this PC."
    except Exception as exc:
        return False, display_name, f"Could not launch {display_name}: {exc}"


def execute_intent(intent: str, entities: dict) -> ExecutionResult:
    """Execute the understood intent and produce a friendly reply and status."""
    local_win = is_local_windows_mode()

    if intent == "open_website":
        site = entities.get("site", "google").lower().strip()
        if site in WEBSITE_URLS:
            url = WEBSITE_URLS[site]
            label = site.title()
        elif "." in site and " " not in site:
            url = site if site.startswith(("http://", "https://")) else f"https://{site}"
            label = site
        else:
            url = f"https://www.google.com/search?q={quote_plus(site)}"
            label = site
        return ExecutionResult(
            action_type="open_url",
            status="Ready to open in browser",
            reply_text=f"Opening {label}.",
            details=f"Target URL: {url}",
            target_url=url,
        )

    if intent == "search_youtube":
        query = entities.get("query", "").strip()
        if not query:
            url = WEBSITE_URLS["youtube"]
            return ExecutionResult(
                action_type="open_url",
                status="Ready to open YouTube",
                reply_text="Opening YouTube.",
                details=f"Target URL: {url}",
                target_url=url,
            )
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        return ExecutionResult(
            action_type="open_url",
            status="YouTube search ready",
            reply_text=f"Searching YouTube for {query}.",
            details=f"Target URL: {url}",
            target_url=url,
        )

    if intent == "search_google":
        query = entities.get("query", "").strip()
        if not query:
            url = WEBSITE_URLS["google"]
            return ExecutionResult(
                action_type="open_url",
                status="Ready to open Google",
                reply_text="Opening Google.",
                details=f"Target URL: {url}",
                target_url=url,
            )
        url = f"https://www.google.com/search?q={quote_plus(query)}"
        return ExecutionResult(
            action_type="open_url",
            status="Google search ready",
            reply_text=f"Searching Google for {query}.",
            details=f"Target URL: {url}",
            target_url=url,
        )

    if intent == "open_app":
        app_key = entities.get("app", "").strip()
        if not app_key:
            return ExecutionResult(
                action_type="windows_app",
                status="App name not specified",
                reply_text="Please tell me which app to open, like Notepad or Calculator.",
                details="Supported apps: Notepad, Calculator, Paint, File Explorer, Chrome, Edge, Settings.",
            )
        display_name = WINDOWS_APPS.get(app_key.lower(), (app_key.title(), ""))[0]
        if not local_win:
            return ExecutionResult(
                action_type="windows_app",
                status="Simulated (Cloud Demo Mode)",
                reply_text=f"In Windows mode, I would open {display_name}.",
                details=f"Cloud server cannot launch desktop apps. Run locally on Windows to open {display_name}.",
                simulated=True,
            )
        ok, display_name, msg = _launch_windows_app(app_key)
        return ExecutionResult(
            action_type="windows_app",
            status="Executed on Windows" if ok else "App launch failed",
            reply_text=f"Opening {display_name}." if ok else msg,
            details=msg,
        )

    if intent in {"volume_up", "volume_down", "volume_set", "volume_mute", "volume_unmute"}:
        mode = intent.replace("volume_", "")
        target_pct = entities.get("volume_percent")
        if not local_win:
            sim_msg = {
                "up": "Volume increased by 10% (Simulated in Cloud Demo Mode).",
                "down": "Volume decreased by 10% (Simulated in Cloud Demo Mode).",
                "set": f"Volume set to {target_pct}% (Simulated in Cloud Demo Mode).",
                "mute": "System audio muted (Simulated in Cloud Demo Mode).",
                "unmute": "System audio unmuted (Simulated in Cloud Demo Mode).",
            }[mode]
            return ExecutionResult(
                action_type="windows_volume",
                status="Simulated (Cloud Demo Mode)",
                reply_text=sim_msg.replace(" (Simulated in Cloud Demo Mode)", ""),
                details=sim_msg,
                simulated=True,
                volume_percent=target_pct,
            )

        ok, new_pct, msg = _adjust_windows_volume(mode, target_pct)
        if not ok:
            return ExecutionResult(
                action_type="windows_volume",
                status="Volume control error",
                reply_text=msg,
                details=msg,
            )
        spoken = {
            "up": f"Volume increased to {new_pct} percent.",
            "down": f"Volume decreased to {new_pct} percent.",
            "set": f"Volume set to {new_pct} percent.",
            "mute": "Audio muted.",
            "unmute": "Audio unmuted.",
        }[mode]
        return ExecutionResult(
            action_type="windows_volume",
            status="Executed on Windows",
            reply_text=spoken,
            details=msg,
            volume_percent=new_pct,
        )

    if intent == "get_time":
        now = datetime.now()
        formatted = now.strftime("%I:%M %p on %A, %B %d, %Y").lstrip("0")
        return ExecutionResult(
            action_type="info",
            status="Success",
            reply_text=f"It is {formatted}.",
            details=f"Server time: {formatted}",
        )

    if intent == "greeting":
        return ExecutionResult(
            action_type="chat",
            status="Success",
            reply_text="Hello! I am your voice assistant. Ask me to open YouTube, search the web, open Notepad, or change the volume.",
            details="Ready for voice or text commands.",
        )

    if intent == "help":
        return ExecutionResult(
            action_type="chat",
            status="Success",
            reply_text="You can say: Open YouTube, Search YouTube for Python, Open Notepad, Open Calculator, Volume up, Volume down, or Set volume to 50 percent.",
            details="Listed supported web and Windows tasks.",
        )

    return ExecutionResult(
        action_type="unknown",
        status="Command not recognized",
        reply_text="Sorry, I didn't understand that command. Try saying 'open YouTube', 'open Notepad', or 'volume up'.",
        details="Confidence below threshold. See quick command examples above.",
    )
