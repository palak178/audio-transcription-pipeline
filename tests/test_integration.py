"""End-to-end tests against a real Whisper model.

Marked ``slow`` and excluded from the default run: the first execution downloads ~40 MB of
weights. Run with ``make test-all`` or ``pytest -m slow``.

These are the tests that would catch a regression the fake engine cannot see — a wrong
sample rate reaching the model, a broken float conversion, VAD misconfiguration.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.conftest import ffmpeg_required

from transcription.config import Settings
from transcription.pipeline import transcribe_file
from transcription.transcriber import FasterWhisperTranscriber

pytestmark = [pytest.mark.slow, ffmpeg_required]

# `tiny` keeps CI honest about cost; accuracy assertions stay loose enough to suit it.
INTEGRATION_MODEL = "tiny"


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    return Settings(whisper_model=INTEGRATION_MODEL, max_duration_seconds=3600)


@pytest.fixture(scope="module")
def real_transcriber(real_settings: Settings) -> FasterWhisperTranscriber:
    engine = FasterWhisperTranscriber(real_settings)
    engine.warm()  # pay the download/load cost once for the module
    return engine


def test_transcribes_real_speech(speech_sample, real_transcriber, real_settings) -> None:
    result = transcribe_file(
        speech_sample, transcriber=real_transcriber, settings=real_settings
    )

    text = result.text.lower()
    assert "quick brown fox" in text
    assert "lazy dog" in text
    assert result.language == "en"
    assert result.language_probability > 0.5


def test_segments_are_ordered_and_within_the_audio(
    speech_sample, real_transcriber, real_settings
) -> None:
    result = transcribe_file(
        speech_sample, transcriber=real_transcriber, settings=real_settings
    )

    assert result.segments
    previous_end = 0.0
    for segment in result.segments:
        assert segment.start >= 0
        assert segment.end > segment.start
        assert segment.start >= previous_end - 0.5  # allow minor overlap between cues
        assert segment.end <= result.audio.duration_seconds + 1.0
        previous_end = segment.end


def test_word_timestamps_align_inside_their_segment(
    speech_sample, real_transcriber, real_settings
) -> None:
    result = transcribe_file(
        speech_sample,
        transcriber=real_transcriber,
        settings=real_settings,
        word_timestamps=True,
    )

    segment = result.segments[0]
    assert segment.words
    for word in segment.words:
        assert segment.start - 0.5 <= word.start <= segment.end + 0.5


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("converted.wav", ["-c:a", "pcm_s16le", "-ar", "8000"]),
        ("converted.flac", ["-c:a", "flac", "-ar", "44100"]),
        ("converted.m4a", ["-c:a", "aac", "-ar", "48000", "-ac", "2"]),
    ],
)
def test_transcript_is_independent_of_input_format(
    speech_sample, real_transcriber, real_settings, tmp_path: Path, name, extra
) -> None:
    """The normalisation step means the container and codec cannot change the output."""
    converted = tmp_path / name
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(speech_sample), *extra, str(converted),
        ],
        check=True,
        capture_output=True,
    )

    result = transcribe_file(converted, transcriber=real_transcriber, settings=real_settings)

    assert "quick brown fox" in result.text.lower()


def test_silence_does_not_hallucinate(tmp_path: Path, real_transcriber, real_settings) -> None:
    """Whisper is known to invent text over silence; the VAD filter is what prevents it."""
    silence = tmp_path / "silence.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "20",
            "-c:a", "pcm_s16le", str(silence),
        ],
        check=True,
        capture_output=True,
    )

    result = transcribe_file(silence, transcriber=real_transcriber, settings=real_settings)

    assert result.segments == []
    assert result.text == ""


def test_speech_followed_by_long_silence_does_not_repeat(
    speech_sample, tmp_path: Path, real_transcriber, real_settings
) -> None:
    """The specific failure this guards: a trailing silence triggering a repetition loop."""
    padded = tmp_path / "padded.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(speech_sample),
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-filter_complex",
            "[0:a]aformat=sample_rates=16000:channel_layouts=mono[a];"
            "[1:a]atrim=0:45[b];[a][b]concat=n=2:v=0:a=1",
            "-c:a", "pcm_s16le", str(padded),
        ],
        check=True,
        capture_output=True,
    )

    result = transcribe_file(padded, transcriber=real_transcriber, settings=real_settings)

    texts = [segment.text.strip().lower() for segment in result.segments]
    assert len(texts) == len(set(texts)), f"repeated segments suggest a loop: {texts}"
    assert "quick brown fox" in result.text.lower()


def test_real_time_factor_is_recorded(speech_sample, real_transcriber, real_settings) -> None:
    result = transcribe_file(
        speech_sample, transcriber=real_transcriber, settings=real_settings
    )

    assert result.real_time_factor is not None
    assert result.processing_seconds > 0


def test_windowed_long_audio_against_a_real_model(speech_sample, tmp_path: Path) -> None:
    """Windowing must not corrupt the transcript or the timeline.

    The sample is looped to ~36 s and the window forced down to 8 s, so the file crosses
    several real window boundaries with a real model decoding each one.
    """
    looped = tmp_path / "looped.mp3"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-stream_loop", "4", "-i", str(speech_sample),
            "-c:a", "libmp3lame", "-b:a", "48k", str(looped),
        ],
        check=True,
        capture_output=True,
    )
    windowed = Settings(
        whisper_model=INTEGRATION_MODEL,
        window_seconds=8.0,
        window_search_seconds=2.0,
    )
    engine = FasterWhisperTranscriber(windowed)

    result = transcribe_file(looped, transcriber=engine, settings=windowed)

    # The phrase recurs once per loop, so crossing boundaries must not lose it.
    assert result.text.lower().count("quick brown fox") >= 4
    # The timeline stays ordered and inside the audio across every seam.
    previous_end = 0.0
    for segment in result.segments:
        assert segment.start >= previous_end - 0.5
        assert segment.end <= result.audio.duration_seconds + 1.0
        previous_end = segment.end
    # Coverage reaches the end of the file: no window was silently skipped.
    assert previous_end > result.audio.duration_seconds * 0.8
    assert [s.id for s in result.segments] == list(range(len(result.segments)))
