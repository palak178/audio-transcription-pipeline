"""Output formats for downstream consumers.

JSON is the machine contract. SRT and WebVTT exist because the overwhelmingly common
downstream use of a timestamped transcript is captioning, and both are trivial to produce
from the same segment list. Plain text is for search indexing and anything that wants prose.
"""

from __future__ import annotations

import textwrap
from enum import StrEnum

from transcription.models import TranscriptionResult


class ResponseFormat(StrEnum):
    """The available output formats.

    An enum rather than a bare string so the API documents the choices: Swagger renders a
    dropdown, and an invalid value is rejected by validation before any work starts.
    """

    json = "json"
    srt = "srt"
    vtt = "vtt"
    txt = "txt"


# Derived from the enum so the two can never drift apart.
FORMATS: tuple[str, ...] = tuple(fmt.value for fmt in ResponseFormat)

_EXTENSIONS = {"json": ".json", "srt": ".srt", "vtt": ".vtt", "txt": ".txt"}

# Media type per format, for HTTP responses.
MEDIA_TYPES = {
    "json": "application/json",
    "srt": "text/plain; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
}


def extension_for(fmt: str) -> str:
    return _EXTENSIONS[fmt]


def to_json(result: TranscriptionResult, *, indent: int | None = 2) -> str:
    """The canonical machine-readable form, including computed fields."""
    return result.model_dump_json(indent=indent)


def format_timecode(seconds: float, *, separator: str = ",") -> str:
    """Render seconds as ``HH:MM:SS,mmm``.

    Hours are always included because both formats require the field, and long recordings
    genuinely cross the one-hour boundary. Negative values are clamped: a malformed negative
    timestamp should produce a valid caption file rather than an unparseable one.
    """
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _cue_text(text: str) -> str:
    """Caption text must be non-empty, or players treat the cue as malformed."""
    return text.strip() or "[inaudible]"


def to_srt(result: TranscriptionResult) -> str:
    blocks = []
    for index, segment in enumerate(result.segments, start=1):
        start = format_timecode(segment.start)
        # Guard against a zero-or-negative-length cue, which some players drop silently.
        end = format_timecode(max(segment.end, segment.start + 0.001))
        blocks.append(f"{index}\n{start} --> {end}\n{_cue_text(segment.text)}\n")
    return "\n".join(blocks)


def to_vtt(result: TranscriptionResult) -> str:
    blocks = ["WEBVTT\n"]
    for segment in result.segments:
        start = format_timecode(segment.start, separator=".")
        end = format_timecode(max(segment.end, segment.start + 0.001), separator=".")
        blocks.append(f"{start} --> {end}\n{_cue_text(segment.text)}\n")
    return "\n".join(blocks)


def to_txt(result: TranscriptionResult, *, width: int = 100) -> str:
    """Prose transcript, wrapped for readability."""
    text = result.text
    if not text:
        return ""
    return textwrap.fill(text, width=width) + "\n"


def render(result: TranscriptionResult, fmt: str) -> str:
    try:
        renderer = {"json": to_json, "srt": to_srt, "vtt": to_vtt, "txt": to_txt}[fmt]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}. Choose from: {', '.join(FORMATS)}") from None
    return renderer(result)
