"""Tests for pipeline orchestration, using a fake engine so these stay fast."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from tests.conftest import FakeTranscriber, ffmpeg_required

from transcription.config import Settings
from transcription.errors import AudioTooLongError, CorruptAudioError, TranscriptionFailedError
from transcription.models import Segment
from transcription.pipeline import safe_filename, transcribe_file
from transcription.transcriber import TranscribeMeta

pytestmark = ffmpeg_required


def test_result_carries_the_full_contract(make_audio, fake_transcriber, settings) -> None:
    source = make_audio("clip.mp3", seconds=2.0, extra=["-c:a", "libmp3lame"])

    result = transcribe_file(source, transcriber=fake_transcriber, settings=settings)

    assert result.schema_version == "1.0"
    assert len(result.id) == 64  # content hash, stable across runs
    assert result.text == "Hello world. Second segment."
    assert result.language == "en"
    assert result.model.name == "fake"
    assert result.audio.codec == "mp3"
    assert result.audio.duration_seconds == pytest.approx(2.0, abs=0.2)
    assert [s.id for s in result.segments] == [0, 1]


def test_id_is_stable_for_identical_audio(make_audio, settings, tmp_path: Path) -> None:
    """Two uploads of the same bytes get the same id, which is what makes caching and
    idempotent retries possible downstream."""
    source = make_audio("a.wav", extra=["-c:a", "pcm_s16le"])
    duplicate = tmp_path / "renamed.wav"
    duplicate.write_bytes(source.read_bytes())

    first = transcribe_file(source, transcriber=FakeTranscriber(), settings=settings)
    second = transcribe_file(duplicate, transcriber=FakeTranscriber(), settings=settings)

    assert first.id == second.id


def test_engine_receives_canonical_samples_not_the_original(make_audio, settings) -> None:
    """The engine only ever sees normalised 16 kHz mono float32 samples.

    Whatever the input codec and sample rate, the engine gets the same representation — the
    invariant that makes the rest of the system format-agnostic.
    """
    transcriber = FakeTranscriber()
    source = make_audio("clip.ogg", seconds=2.0, extra=["-c:a", "libopus", "-ar", "48000"])

    transcribe_file(source, transcriber=transcriber, settings=settings)

    samples = transcriber.calls[0]["samples"]
    assert samples.dtype.name == "float32"
    assert abs(float(samples.max())) <= 1.0
    # 2 seconds at 16 kHz, whatever the source rate was.
    assert len(samples) == pytest.approx(2 * 16_000, rel=0.1)


def test_options_reach_the_engine(make_audio, settings) -> None:
    transcriber = FakeTranscriber()
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    transcribe_file(
        source,
        transcriber=transcriber,
        settings=settings,
        language="fr",
        word_timestamps=True,
    )

    assert transcriber.calls[0]["language"] == "fr"
    assert transcriber.calls[0]["word_timestamps"] is True


def test_temporary_files_are_cleaned_up(make_audio, fake_transcriber, settings) -> None:
    """The normalised WAV must not outlive the call that created it."""
    scratch = Path(tempfile.gettempdir())
    before = set(scratch.glob("transcribe-*"))
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    transcribe_file(source, transcriber=fake_transcriber, settings=settings)

    assert set(scratch.glob("transcribe-*")) == before


# --- Long audio -----------------------------------------------------------------------


def test_rejects_audio_over_the_limit(make_audio, fake_transcriber, settings) -> None:
    source = make_audio("long.wav", seconds=5.0, extra=["-c:a", "pcm_s16le"])

    with pytest.raises(AudioTooLongError) as caught:
        transcribe_file(
            source,
            transcriber=fake_transcriber,
            settings=settings,
            max_duration_seconds=2.0,
        )

    assert caught.value.http_status == 413
    assert caught.value.details["limit_seconds"] == 2.0
    # Rejected on metadata alone: the engine was never invoked, so no CPU was spent.
    assert fake_transcriber.calls == []


def test_accepts_audio_within_the_limit(make_audio, fake_transcriber, settings) -> None:
    source = make_audio("short.wav", seconds=1.0, extra=["-c:a", "pcm_s16le"])

    result = transcribe_file(
        source,
        transcriber=fake_transcriber,
        settings=settings,
        max_duration_seconds=2.0,
    )

    assert result.segments


def test_progress_is_reported_and_completes(make_audio, fake_transcriber, settings) -> None:
    source = make_audio("clip.wav", seconds=3.0, extra=["-c:a", "pcm_s16le"])
    updates: list[tuple[float, float]] = []

    transcribe_file(
        source,
        transcriber=fake_transcriber,
        settings=settings,
        progress=lambda done, total: updates.append((done, total)),
    )

    assert len(updates) >= 2
    assert updates[-1][0] == updates[-1][1]  # a finished run reports 100%
    assert all(done <= total for done, total in updates)


def test_duration_comes_from_decoded_audio_when_metadata_lies(
    make_audio, fake_transcriber, settings, tmp_path: Path
) -> None:
    """Some containers declare no duration; the normalised WAV is authoritative."""
    import subprocess

    raw = tmp_path / "stream.ogg"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:r=16000", "-t", "2",
            "-c:a", "libopus", str(raw),
        ],
        check=True,
        capture_output=True,
    )

    result = transcribe_file(raw, transcriber=fake_transcriber, settings=settings)

    assert result.audio.duration_seconds == pytest.approx(2.0, abs=0.3)


# --- Error propagation ----------------------------------------------------------------


def test_invalid_input_fails_before_the_engine(tmp_path: Path, fake_transcriber, settings) -> None:
    broken = tmp_path / "nope.mp3"
    broken.write_text("not audio")

    with pytest.raises(CorruptAudioError):
        transcribe_file(broken, transcriber=fake_transcriber, settings=settings)

    assert fake_transcriber.calls == []


def test_engine_failure_surfaces_as_a_typed_error(make_audio, settings) -> None:
    transcriber = FakeTranscriber(error=TranscriptionFailedError("model died"))
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    with pytest.raises(TranscriptionFailedError) as caught:
        transcribe_file(source, transcriber=transcriber, settings=settings)

    # Transient, so a caller or scheduler knows this one is worth retrying.
    assert caught.value.retryable is True


def test_silence_yields_an_empty_transcript_not_an_error(make_audio, settings) -> None:
    """Pure silence is valid input with no speech in it."""
    transcriber = FakeTranscriber(segments=[])
    source = make_audio("silence.wav", extra=["-c:a", "pcm_s16le"])

    result = transcribe_file(source, transcriber=transcriber, settings=settings)

    assert result.segments == []
    assert result.text == ""


def test_display_name_overrides_temp_filename(make_audio, fake_transcriber, settings) -> None:
    source = make_audio("upload.bin", extra=["-c:a", "pcm_s16le", "-f", "wav"])

    result = transcribe_file(
        source,
        transcriber=fake_transcriber,
        settings=settings,
        display_name="meeting notes.mp3",
    )

    assert result.audio.filename == "meeting notes.mp3"


# --- Filename sanitisation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("clip.mp3", "clip.mp3"),
        ("../../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\x.wav", "x.wav"),
        ("/absolute/path/clip.mp3", "clip.mp3"),
        ("....//clip.mp3", "clip.mp3"),
        ("", "upload"),
        (None, "upload"),
        ("...", "upload"),
    ],
)
def test_safe_filename_strips_path_components(supplied, expected) -> None:
    """Client filenames are untrusted and never used to build a path."""
    assert safe_filename(supplied) == expected


def test_safe_filename_drops_control_characters() -> None:
    assert safe_filename("cl\x00ip\n.mp3") == "clip.mp3"


def test_safe_filename_is_length_bounded() -> None:
    assert len(safe_filename("a" * 500 + ".mp3")) <= 255


def test_settings_reject_nonsense_limits() -> None:
    with pytest.raises(ValueError):
        Settings(max_duration_seconds=0)


def test_engine_is_warmed_before_timing(make_audio, fake_transcriber, settings) -> None:
    """Model-load time must not be counted as transcription time."""
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    transcribe_file(source, transcriber=fake_transcriber, settings=settings)

    assert fake_transcriber.warm_calls == 1


# --- Multi-window stitching -----------------------------------------------------------


class WindowEchoTranscriber:
    """Returns one segment spanning each window it is given, labelled by call order.

    Lets a test verify stitching precisely: the pipeline must rebase each window's
    window-relative timestamps onto the whole recording's timeline.
    """

    def __init__(self) -> None:
        self.offsets_seen: list[float] = []
        self.languages_seen: list[str | None] = []

    @property
    def name(self) -> str:
        return "window-echo"

    def warm(self) -> None:
        return None

    def transcribe(self, samples, *, language=None, word_timestamps=False):
        from transcription.models import Word

        index = len(self.offsets_seen)
        duration = len(samples) / 16_000
        self.offsets_seen.append(duration)
        self.languages_seen.append(language)

        words = None
        if word_timestamps:
            words = [Word(start=0.0, end=duration, word=f"w{index}", probability=0.9)]
        segment = Segment(
            id=0,  # every window numbers from zero; the pipeline must renumber
            start=0.0,
            end=duration,
            text=f"window{index}",
            avg_logprob=-0.3,
            words=words,
        )
        meta = TranscribeMeta(
            language=language or "en", language_probability=0.9, duration=duration
        )
        return meta, iter([segment])


def test_long_audio_is_transcribed_in_multiple_windows(make_audio) -> None:
    tight = Settings(whisper_model="fake", window_seconds=2.0, window_search_seconds=0.5)
    transcriber = WindowEchoTranscriber()
    source = make_audio("long.wav", seconds=9.0, extra=["-c:a", "pcm_s16le"])

    result = transcribe_file(source, transcriber=transcriber, settings=tight)

    assert len(transcriber.offsets_seen) >= 4
    assert result.text.startswith("window0 window1")


def test_stitched_timestamps_are_absolute_and_contiguous(make_audio) -> None:
    """Segment times must describe the whole recording, not the window they came from."""
    tight = Settings(whisper_model="fake", window_seconds=2.0, window_search_seconds=0.5)
    source = make_audio("long.wav", seconds=9.0, extra=["-c:a", "pcm_s16le"])

    result = transcribe_file(source, transcriber=WindowEchoTranscriber(), settings=tight)

    assert result.segments[0].start == 0.0
    previous_end = 0.0
    for segment in result.segments:
        # Each window's segment begins exactly where the previous one ended.
        assert segment.start == pytest.approx(previous_end, abs=1e-3)
        assert segment.end > segment.start
        previous_end = segment.end
    # The final timestamp covers the whole file, so nothing was dropped at the seams.
    assert previous_end == pytest.approx(result.audio.duration_seconds, abs=0.05)


def test_segment_ids_are_contiguous_across_windows(make_audio) -> None:
    tight = Settings(whisper_model="fake", window_seconds=2.0, window_search_seconds=0.5)
    source = make_audio("long.wav", seconds=9.0, extra=["-c:a", "pcm_s16le"])

    result = transcribe_file(source, transcriber=WindowEchoTranscriber(), settings=tight)

    assert [s.id for s in result.segments] == list(range(len(result.segments)))


def test_word_timestamps_are_shifted_too(make_audio) -> None:
    tight = Settings(whisper_model="fake", window_seconds=2.0, window_search_seconds=0.5)
    source = make_audio("long.wav", seconds=9.0, extra=["-c:a", "pcm_s16le"])

    result = transcribe_file(
        source,
        transcriber=WindowEchoTranscriber(),
        settings=tight,
        word_timestamps=True,
    )

    for segment in result.segments:
        assert segment.words
        assert segment.words[0].start == pytest.approx(segment.start, abs=1e-3)
        assert segment.words[0].end == pytest.approx(segment.end, abs=1e-3)


def test_language_is_detected_once_and_pinned(make_audio) -> None:
    """Re-detecting per window lets a quiet window flip language mid-recording."""
    tight = Settings(whisper_model="fake", window_seconds=2.0, window_search_seconds=0.5)
    transcriber = WindowEchoTranscriber()
    source = make_audio("long.wav", seconds=9.0, extra=["-c:a", "pcm_s16le"])

    transcribe_file(source, transcriber=transcriber, settings=tight)

    assert transcriber.languages_seen[0] is None  # auto-detect on the first window
    assert all(lang == "en" for lang in transcriber.languages_seen[1:])


def test_progress_advances_monotonically_over_windows(make_audio) -> None:
    tight = Settings(whisper_model="fake", window_seconds=2.0, window_search_seconds=0.5)
    source = make_audio("long.wav", seconds=9.0, extra=["-c:a", "pcm_s16le"])
    updates: list[tuple[float, float]] = []

    transcribe_file(
        source,
        transcriber=WindowEchoTranscriber(),
        settings=tight,
        progress=lambda done, total: updates.append((done, total)),
    )

    done_values = [done for done, _ in updates]
    assert done_values == sorted(done_values)
    assert updates[-1][0] == updates[-1][1]
