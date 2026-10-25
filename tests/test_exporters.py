"""Tests for the downstream output formats."""

from __future__ import annotations

import json

import pytest

from transcription import exporters
from transcription.models import AudioInfo, ModelInfo, Segment, TranscriptionResult


def build_result(segments: list[Segment]) -> TranscriptionResult:
    return TranscriptionResult(
        id="a" * 64,
        language="en",
        language_probability=0.98,
        audio=AudioInfo(
            filename="clip.mp3",
            sha256="a" * 64,
            size_bytes=1234,
            duration_seconds=10.0,
        ),
        model=ModelInfo(
            name="base",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            vad_filter=True,
            word_timestamps=False,
        ),
        segments=segments,
        processing_seconds=2.5,
    )


@pytest.fixture
def result() -> TranscriptionResult:
    return build_result(
        [
            Segment(id=0, start=0.0, end=2.5, text="Hello world.", avg_logprob=-0.2),
            Segment(id=1, start=2.5, end=4.25, text="Second line.", avg_logprob=-0.5),
        ]
    )


# --- Timecodes ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (0.001, "00:00:00,001"),
        (1.5, "00:00:01,500"),
        (61.25, "00:01:01,250"),
        (3600.0, "01:00:00,000"),  # long recordings do cross the hour mark
        (7265.123, "02:01:05,123"),
        (-5.0, "00:00:00,000"),  # clamped, so a bad value cannot corrupt the file
    ],
)
def test_format_timecode(seconds, expected) -> None:
    assert exporters.format_timecode(seconds) == expected


def test_vtt_uses_a_decimal_point() -> None:
    assert exporters.format_timecode(1.5, separator=".") == "00:00:01.500"


# --- SRT ------------------------------------------------------------------------------


def test_srt_structure(result: TranscriptionResult) -> None:
    srt = exporters.to_srt(result)

    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nHello world.\n")
    # Cue numbering is 1-based and contiguous.
    assert "\n2\n00:00:02,500 --> 00:00:04,250\nSecond line.\n" in srt


def test_srt_gives_empty_text_a_placeholder() -> None:
    """A blank cue makes the file malformed for many players."""
    result = build_result([Segment(id=0, start=0.0, end=1.0, text="   ")])

    assert "[inaudible]" in exporters.to_srt(result)


def test_srt_never_emits_a_zero_length_cue() -> None:
    result = build_result([Segment(id=0, start=5.0, end=5.0, text="Blip.")])

    body = exporters.to_srt(result)

    start, end = body.splitlines()[1].split(" --> ")
    assert start != end


def test_srt_of_empty_transcript_is_empty() -> None:
    assert exporters.to_srt(build_result([])) == ""


# --- WebVTT ---------------------------------------------------------------------------


def test_vtt_has_required_header(result: TranscriptionResult) -> None:
    vtt = exporters.to_vtt(result)

    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:02.500" in vtt


# --- JSON and text --------------------------------------------------------------------


def test_json_carries_the_downstream_contract(result: TranscriptionResult) -> None:
    payload = json.loads(exporters.to_json(result))

    assert payload["schema_version"] == "1.0"
    assert payload["id"] == "a" * 64
    assert payload["text"] == "Hello world. Second line."
    assert payload["real_time_factor"] == 0.25
    # Provenance: enough to reproduce the run.
    assert payload["model"]["name"] == "base"
    assert payload["model"]["beam_size"] == 5
    # Per-segment timing and confidence for downstream routing.
    first = payload["segments"][0]
    assert (first["start"], first["end"], first["duration"]) == (0.0, 2.5, 2.5)
    assert 0.0 < first["confidence"] <= 1.0


def test_confidence_is_higher_for_higher_logprob(result: TranscriptionResult) -> None:
    assert result.segments[0].confidence > result.segments[1].confidence


def test_txt_is_wrapped_prose(result: TranscriptionResult) -> None:
    assert exporters.to_txt(result).strip() == "Hello world. Second line."


def test_txt_of_empty_transcript_is_empty() -> None:
    assert exporters.to_txt(build_result([])) == ""


def test_unicode_survives_every_format() -> None:
    result = build_result([Segment(id=0, start=0.0, end=1.0, text="Café — naïve 日本語")])

    for fmt in exporters.FORMATS:
        assert "Café" in exporters.render(result, fmt)


def test_render_rejects_unknown_format(result: TranscriptionResult) -> None:
    with pytest.raises(ValueError, match="Unknown format"):
        exporters.render(result, "docx")
