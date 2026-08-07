import re

_PATTERNS = (
    (re.compile(r"(?i)\b(?:bearer|token|api[_-]?key)\s*[:=]\s*[^\s,;]+"), "[REDACTED:API_TOKEN]"),
    (re.compile(r"(?i)(https?://[^\s/@]+):[^\s/@]+@"), r"\1:[REDACTED:PASSWORD]@"),
    (re.compile(r"(?i)\bpassword\s*[:=]\s*[^\s,;]+"), "password=[REDACTED:PASSWORD]"),
    (
        re.compile(r"-----BEGIN [^-]+ PRIVATE KEY-----.*?-----END [^-]+ PRIVATE KEY-----", re.S),
        "[REDACTED:PRIVATE_KEY]",
    ),
)


def redact_secrets(content: str) -> str:
    """Remove common credentials before any scoring, provider, or storage step."""
    redacted = content
    for pattern, replacement in _PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted
