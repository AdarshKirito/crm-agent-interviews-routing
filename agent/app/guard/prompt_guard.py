"""Prompt-injection check with Llama Prompt Guard 2 (86M) served by Groq.

The model has a 512-token window, so long text is checked in chunks and the highest
score wins. Without GROQ_API_KEY the check is skipped and reported as unavailable.
"""

import logging
import re

import httpx

from ..config import GROQ_API_KEY, PROMPT_GUARD_MODEL

logger = logging.getLogger(__name__)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CHUNK_CHARS = 1500  # comfortably under 512 tokens


def _chunks(text: str) -> list[str]:
    text = text.strip()
    return [text[i : i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)] or [""]


def parse_score(content: str) -> float | None:
    """Prompt Guard returns the probability that the text is an attack; accept a
    bare number or a label."""
    content = (content or "").strip()
    number = re.search(r"\d*\.?\d+(?:[eE]-?\d+)?", content)
    if number:
        try:
            return max(0.0, min(1.0, float(number.group(0))))
        except ValueError:
            pass
    upper = content.upper()
    if any(w in upper for w in ("MALICIOUS", "INJECTION", "JAILBREAK")):
        return 1.0
    if "BENIGN" in upper:
        return 0.0
    return None


async def injection_score(text: str, timeout: float = 10.0) -> float | None:
    if not GROQ_API_KEY or not text.strip():
        return None
    best: float | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for chunk in _chunks(text):
            try:
                res = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": PROMPT_GUARD_MODEL, "messages": [{"role": "user", "content": chunk}]},
                )
                res.raise_for_status()
                score = parse_score(res.json()["choices"][0]["message"]["content"])
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as err:
                logger.warning("prompt guard unavailable: %s", err)
                return best
            if score is not None:
                best = score if best is None else max(best, score)
    return best
