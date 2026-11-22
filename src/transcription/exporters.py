"""Output formats for downstream consumers.

JSON is the machine contract. SRT and WebVTT exist because the overwhelmingly common
downstream use of a timestamped transcript is captioning, and both are trivial to produce
from the same segment list. Plain text is for search indexing and anything that wants prose.

**Speaker labels are lossless in exactly one format.** JSON carries them as data. WebVTT has a
real mechanism for them — the ``<v>`` voice span — so a conformant player styles or strips the
tag. SRT has no speaker field at all, so a label can only be prefixed into the cue text, where
it becomes indistinguishable from the transcript and cannot be separated out again. Plain text
lost the timings already. So JSON is the artefact to keep; the caption formats are renderings
of it, and two of them are one-way.

Output is byte-identical to a pre-diarisation run whenever no segment carries a label, which
is what keeps an additive feature from silently changing every existing consumer's captions.
"""

from __future__ import annotations

import textwrap
from enum import StrEnum

from transcription.diarisation import iter_turn_blocks
from transcription.models import Segment, TranscriptionResult


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


def _label(segment: Segment, speaker_labels: bool) -> str | None:
    return segment.speaker if speaker_labels and segment.speaker else None


def to_srt(result: TranscriptionResult, *, speaker_labels: bool = True) -> str:
    """SubRip captions.

    A speaker label has to be prefixed into the cue text, because SRT has nowhere else to put
    one. That makes it lossy — the label is now part of the transcript as far as any consumer
    is concerned. The ``SPEAKER_00:`` prefix is the de facto convention rather than anything
    in a specification, and it spends about eleven characters of a caption line that
    convention keeps to 37-42. Pass ``speaker_labels=False`` for clean captions out of a
    diarised job.
    """
    blocks = []
    for index, segment in enumerate(result.segments, start=1):
        start = format_timecode(segment.start)
        # Guard against a zero-or-negative-length cue, which some players drop silently.
        end = format_timecode(max(segment.end, segment.start + 0.001))
        text = _cue_text(segment.text)
        speaker = _label(segment, speaker_labels)
        if speaker:
            text = f"{speaker}: {text}"
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(result: TranscriptionResult, *, speaker_labels: bool = True) -> str:
    """WebVTT captions.

    The only caption format here that carries a speaker properly: ``<v Name>`` is a cue span
    the specification defines, so a conformant player styles it or strips it rather than
    showing the markup as text. That makes WebVTT the one lossless caption option, and the
    reason to prefer it over SRT whenever the consumer supports it.
    """
    blocks = ["WEBVTT\n"]
    for segment in result.segments:
        start = format_timecode(segment.start, separator=".")
        end = format_timecode(max(segment.end, segment.start + 0.001), separator=".")
        text = _cue_text(segment.text)
        speaker = _label(segment, speaker_labels)
        if speaker:
            text = f"<v {speaker}>{text}</v>"
        blocks.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_txt(result: TranscriptionResult, *, width: int = 100, speaker_labels: bool = True) -> str:
    """Prose transcript, wrapped for readability.

    This is the format diarisation improves most, and the reason is the merging rather than the
    labels: consecutive segments from one speaker become a single block, which is the
    difference between a wall of short lines and something a person can read. Timings were
    already gone here, so nothing is lost that was not lost before.
    """
    if not speaker_labels or not any(segment.speaker for segment in result.segments):
        text = result.text
        if not text:
            return ""
        return textwrap.fill(text, width=width) + "\n"

    blocks = []
    for speaker, segments in iter_turn_blocks(result.segments):
        body = " ".join(segment.text.strip() for segment in segments).strip()
        if not body:
            continue
        prefix = f"{speaker}: " if speaker else ""
        blocks.append(
            textwrap.fill(
                prefix + body,
                width=width,
                # Indent the wrapped lines so the label stays visually attached to its block.
                subsequent_indent=" " * len(prefix),
            )
        )
    return "\n\n".join(blocks) + "\n" if blocks else ""


def render(result: TranscriptionResult, fmt: str, *, speaker_labels: bool = True) -> str:
    if fmt == "json":
        return to_json(result)
    try:
        renderer = {"srt": to_srt, "vtt": to_vtt, "txt": to_txt}[fmt]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}. Choose from: {', '.join(FORMATS)}") from None
    return renderer(result, speaker_labels=speaker_labels)
