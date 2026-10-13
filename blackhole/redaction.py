"""Recursive credential redaction shared by persisted and displayed errors."""

from __future__ import annotations

import re
from typing import Any, Mapping

# Match sensitive key *segments*, not arbitrary substrings. This keeps keys such
# as access_token / api_key protected without corrupting benign fields such as
# max_new_tokens.
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_.-])(?:token|secret|password|authorization|cookie|api[-_]?key)(?:$|[_.-])",
    re.I,
)
_PREFIX = re.compile(r"(?:hf_|ghp_|gho_|github_pat_)[A-Za-z0-9_-]{8,}")
_AUTH = re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", re.I)
_QUERY = re.compile(r"([?&](?:token|access_token|api_key|key|secret|password|auth)=)[^&#\s]+", re.I)


def redact_text(value: object, *, limit: int | None = None) -> str:
    text = _PREFIX.sub("[REDACTED]", str(value))
    text = _AUTH.sub("[REDACTED]", text)
    text = _QUERY.sub(r"\1[REDACTED]", text)
    return text if limit is None or len(text) <= limit else text[:limit] + "…"


def redact_value(value: Any, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): redact_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value
