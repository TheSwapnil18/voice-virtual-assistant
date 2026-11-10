"""
Step 5: Word Error Rate (WER) Evaluation Module.

Computes standard ASR evaluation metrics using word-level Levenshtein
dynamic programming:
    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
Also returns word-by-word alignment so non-technical users can visually see
exact matches, wrong words, missed words, and extra words.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass
class WERResult:
    reference_clean: str
    hypothesis_clean: str
    total_ref_words: int
    correct_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float
    wer_percent: float
    word_accuracy_percent: float
    alignment: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_for_wer(text: str) -> str:
    """Lowercase, expand '%' to 'percent', and strip punctuation for fair comparison."""
    lowered = re.sub(r"(\d)\s*%", r"\1 percent", text.lower())
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _levenshtein_align(ref_tokens: list[str], hyp_tokens: list[str]) -> tuple[int, int, int, int, list[dict]]:
    """Compute minimum edit operations (S, D, I, Matches) and backtrace alignment."""
    n = len(ref_tokens)
    m = len(hyp_tokens)

    # dp[i][j] holds minimum edits to convert ref[:i] -> hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "match"
            else:
                sub_cost = dp[i - 1][j - 1] + 1
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1
                best = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = best
                if best == sub_cost:
                    op[i][j] = "sub"
                elif best == del_cost:
                    op[i][j] = "del"
                else:
                    op[i][j] = "ins"

    # Backtrace to build readable word-by-word alignment
    alignment: list[dict] = []
    matches = subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        step = op[i][j]
        if step == "match":
            matches += 1
            alignment.append({"op": "match", "ref": ref_tokens[i - 1], "hyp": hyp_tokens[j - 1]})
            i -= 1
            j -= 1
        elif step == "sub":
            subs += 1
            alignment.append({"op": "sub", "ref": ref_tokens[i - 1], "hyp": hyp_tokens[j - 1]})
            i -= 1
            j -= 1
        elif step == "del":
            dels += 1
            alignment.append({"op": "del", "ref": ref_tokens[i - 1], "hyp": ""})
            i -= 1
        else:
            ins += 1
            alignment.append({"op": "ins", "ref": "", "hyp": hyp_tokens[j - 1]})
            j -= 1

    alignment.reverse()
    return matches, subs, dels, ins, alignment


def calculate_wer(reference: str, hypothesis: str) -> WERResult:
    """Calculate Word Error Rate and word alignment between reference and hypothesis."""
    ref_clean = normalize_for_wer(reference)
    hyp_clean = normalize_for_wer(hypothesis)

    ref_tokens = ref_clean.split() if ref_clean else []
    hyp_tokens = hyp_clean.split() if hyp_clean else []

    if not ref_tokens:
        raise ValueError("Reference sentence cannot be empty.")

    matches, subs, dels, ins, alignment = _levenshtein_align(ref_tokens, hyp_tokens)
    total_ref = len(ref_tokens)
    wer = (subs + dels + ins) / float(total_ref)
    wer_pct = round(wer * 100.0, 2)
    accuracy_pct = round(max(0.0, 100.0 - wer_pct), 2)

    return WERResult(
        reference_clean=ref_clean,
        hypothesis_clean=hyp_clean,
        total_ref_words=total_ref,
        correct_words=matches,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        wer=round(wer, 4),
        wer_percent=wer_pct,
        word_accuracy_percent=accuracy_pct,
        alignment=alignment,
    )
