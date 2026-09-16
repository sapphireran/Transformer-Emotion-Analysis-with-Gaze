"""Tiny tokenizer used by the sklearn examples (no NLTK download)."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercased alphabetic tokens, keeping simple apostrophes."""
    return [match.group(0).lower() for match in _TOKEN.finditer(text or "")]


def token_count(text: str) -> int:
    return len(tokenize(text))


def sklearn_tokenizer(text: str) -> list[str]:
    """Alias so TfidfVectorizer can take `tokenizer=tokenize`."""
    return tokenize(text)
