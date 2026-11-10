"""
Step 3: Natural Language Processing (NLP) Module.

Pipeline:
1. Text normalization (lowercasing and punctuation cleanup)
2. TF-IDF vectorization + Cosine Similarity against data/intents.json
3. Lightweight regex/keyword entity (slot) extraction for websites,
   search queries, Windows apps, and volume percentages.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWN_WEBSITES = {
    "youtube",
    "google",
    "wikipedia",
    "github",
    "gmail",
    "spotify",
    "chatgpt",
    "instagram",
    "linkedin",
    "maps",
    "google maps",
    "stackoverflow",
    "reddit",
    "netflix",
    "amazon",
}

KNOWN_APPS = {
    "notepad",
    "calculator",
    "calc",
    "paint",
    "mspaint",
    "cmd",
    "command prompt",
    "terminal",
    "explorer",
    "file explorer",
    "files",
    "chrome",
    "google chrome",
    "edge",
    "microsoft edge",
    "task manager",
    "settings",
    "windows settings",
    "control panel",
    "snipping tool",
    "clock",
}

INTENT_LABELS = {
    "open_website": "Open a Website",
    "search_youtube": "Search on YouTube",
    "search_google": "Search on Google",
    "open_app": "Open Windows App",
    "volume_up": "Increase Volume",
    "volume_down": "Decrease Volume",
    "volume_set": "Set Specific Volume",
    "volume_mute": "Mute Volume",
    "volume_unmute": "Unmute Volume",
    "get_time": "Check Date & Time",
    "greeting": "Greeting",
    "help": "Show Help / Commands",
    "unknown": "Unknown Command",
}


@dataclass
class NLPResult:
    cleaned_text: str
    intent: str
    intent_label: str
    confidence: float
    entities: dict
    matched_example: str
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


ASR_ALIASES = {
    "note pad": "notepad",
    "you tube": "youtube",
    "chat gpt": "chatgpt",
    "ms paint": "mspaint",
    "file manager": "file explorer",
}


def normalize_text(text: str) -> str:
    """Lowercase, clean punctuation, and merge common ASR split-word phrases."""
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[^\w\s.%]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for split_phrase, canonical in ASR_ALIASES.items():
        cleaned = re.sub(rf"\b{re.escape(split_phrase)}\b", canonical, cleaned)
    return cleaned


class IntentClassifier:
    """TF-IDF + Cosine Similarity intent classifier with slot extraction."""

    def __init__(self, intents_path: Path, threshold: float = 0.25) -> None:
        self.threshold = threshold
        raw = json.loads(intents_path.read_text(encoding="utf-8"))

        self.examples: list[str] = []
        self.tags: list[str] = []
        for item in raw["intents"]:
            for example in item["examples"]:
                self.examples.append(normalize_text(example))
                self.tags.append(item["tag"])

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.example_matrix = self.vectorizer.fit_transform(self.examples)

    def _extract_entities_and_refine(self, cleaned: str, intent: str) -> tuple[str, dict]:
        """Extract command parameters (website, app, query, volume %) and refine intent."""
        entities: dict = {}

        # 1. Explicit volume percentage (e.g., "set volume to 50", "volume 40%")
        if "volume" in cleaned or "sound" in cleaned or "audio" in cleaned:
            num_match = re.search(r"\b(\d{1,3})\s*(?:%|percent)?\b", cleaned)
            if num_match:
                level = max(0, min(100, int(num_match.group(1))))
                return "volume_set", {"volume_percent": level}

        # 2. YouTube search queries
        yt_patterns = [
            r"(?:search|find|look up)\s+(?:on\s+)?youtube\s+(?:for\s+)?(.+)",
            r"youtube\s+search\s+(?:for\s+)?(.+)",
            r"play\s+(.+?)\s+on\s+youtube",
            r"search\s+(.+?)\s+on\s+youtube",
        ]
        for pat in yt_patterns:
            m = re.search(pat, cleaned)
            if m:
                return "search_youtube", {"query": m.group(1).strip()}

        # 3. Google / web search queries
        google_patterns = [
            r"(?:search|look up)\s+(?:on\s+)?google\s+(?:for\s+)?(.+)",
            r"^google\s+(.+)",
            r"(?:search|look up)\s+(?:the\s+web\s+)?(?:for\s+)?(.+)",
        ]
        for pat in google_patterns:
            m = re.search(pat, cleaned)
            if m and "youtube" not in cleaned:
                return "search_google", {"query": m.group(1).strip()}

        # 4. Disambiguate "open <app>" vs "open <website>"
        for app in sorted(KNOWN_APPS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(app)}\b", cleaned):
                return "open_app", {"app": app}

        for site in sorted(KNOWN_WEBSITES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(site)}\b", cleaned):
                return "open_website", {"site": site}

        domain_match = re.search(r"\b([a-z0-9-]+\.(?:com|org|net|in|edu|io|ai))\b", cleaned)
        if domain_match:
            return "open_website", {"site": domain_match.group(1)}

        # Fallback target if user literally said "open <something>"
        open_match = re.search(r"^(?:open|launch|visit|go to|start)\s+(.+)$", cleaned)
        if open_match:
            target = open_match.group(1).strip()
            if intent == "open_app":
                entities["app"] = target
            else:
                intent = "open_website"
                entities["site"] = target

        return intent, entities

    def predict(self, text: str) -> NLPResult:
        start = time.perf_counter()
        cleaned = normalize_text(text)

        if not cleaned:
            return NLPResult(
                cleaned_text="",
                intent="unknown",
                intent_label=INTENT_LABELS["unknown"],
                confidence=0.0,
                entities={},
                matched_example="",
                latency_ms=0.0,
            )

        query_vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(query_vec, self.example_matrix)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        best_tag = self.tags[best_idx]
        matched_example = self.examples[best_idx]

        # Refine with slot/entity extraction before applying fallback threshold
        refined_intent, entities = self._extract_entities_and_refine(cleaned, best_tag)

        # If entity extractor matched a clear slot, ensure reasonable minimum confidence
        if entities and best_score < self.threshold:
            best_score = max(best_score, 0.55)

        if best_score < self.threshold and not entities:
            refined_intent = "unknown"

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return NLPResult(
            cleaned_text=cleaned,
            intent=refined_intent,
            intent_label=INTENT_LABELS.get(refined_intent, refined_intent),
            confidence=round(best_score, 2),
            entities=entities,
            matched_example=matched_example,
            latency_ms=elapsed_ms,
        )
