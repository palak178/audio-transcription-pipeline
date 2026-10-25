"""Shared fixtures.

The fast suite runs the real pipeline, the real ffmpeg calls and the real HTTP stack, but
substitutes a fake speech-to-text engine. That is the point of the ``Transcriber`` protocol:
model inference is the only slow, non-deterministic part of the system, so faking just that
leaves everything else genuinely under test and keeps the suite at a couple of seconds with
no model download.

Audio fixtures are generated on demand with ffmpeg into ``tmp_path``, so tests do not depend
on anyone having run ``make fixtures`` first. The one committed binary,
``samples/speech_en.mp3``, supplies real speech for the slow integration test.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from transcription.api import create_app
from transcription.config import Settings
from transcription.models import Segment
from transcription.transcriber import TranscribeMeta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEECH_SAMPLE = PROJECT_ROOT / "samples" / "speech_en.mp3"

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


@pytest.fixture(scope="session")
def speech_sample() -> Path:
    if not SPEECH_SAMPLE.exists():
        pytest.skip(f"missing committed sample: {SPEECH_SAMPLE}")
    return SPEECH_SAMPLE


@pytest.fixture
def make_audio(tmp_path: Path) -> Callable[..., Path]:
    """Factory for synthetic audio files in arbitrary formats.

    ``make_audio("probe.wav", seconds=1.0)`` returns a path to a one-second tone. Extra
    ffmpeg arguments can be appended to vary codec, rate and channel count.
    """

    def _make(
        name: str,
        *,
        seconds: float = 1.0,
        frequency: int = 440,
        rate: int = 16_000,
        channels: int = 1,
        extra: list[str] | None = None,
    ) -> Path:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:r={rate}",
            "-t",
            str(seconds),
            "-ac",
            str(channels),
            *(extra or []),
            str(destination),
        ]
        subprocess.run(command, check=True, capture_output=True)
        return destination

    return _make


class FakeTranscriber:
    """Deterministic stand-in for a real engine.

    Also records the arguments it was called with, so tests can assert that request options
    actually reach the engine rather than being silently dropped.
    """

    def __init__(
        self,
        segments: list[Segment] | None = None,
        *,
        language: str = "en",
        error: Exception | None = None,
    ) -> None:
        self._segments = (
            segments
            if segments is not None
            else [
                Segment(id=0, start=0.0, end=1.5, text="Hello world.", avg_logprob=-0.2),
                Segment(id=1, start=1.5, end=3.0, text="Second segment.", avg_logprob=-0.4),
            ]
        )
        self._language = language
        self._error = error
        self.calls: list[dict[str, object]] = []
        self.warm_calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def warm(self) -> None:
        self.warm_calls += 1

    def transcribe(
        self,
        samples,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> tuple[TranscribeMeta, Iterator[Segment]]:
        self.calls.append(
            {
                "samples": samples,
                "sample_count": len(samples),
                "language": language,
                "word_timestamps": word_timestamps,
            }
        )
        if self._error is not None:
            raise self._error

        meta = TranscribeMeta(
            language=language or self._language,
            language_probability=0.99,
            duration=self._segments[-1].end if self._segments else 0.0,
        )
        return meta, iter(list(self._segments))


@pytest.fixture
def fake_transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def settings() -> Settings:
    # Explicit construction, not get_settings(), so a developer's .env cannot affect tests.
    return Settings(
        whisper_model="fake",
        max_duration_seconds=3600,
        max_sync_duration_seconds=60,
        max_upload_bytes=1024 * 1024,
        max_concurrent_transcriptions=2,
    )


@pytest.fixture
def client(settings: Settings, fake_transcriber: FakeTranscriber) -> Iterator[TestClient]:
    app = create_app(settings=settings, transcriber=fake_transcriber)
    with TestClient(app) as test_client:
        yield test_client
