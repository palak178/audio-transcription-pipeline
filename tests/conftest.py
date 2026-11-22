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

import os
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from transcription.api import create_app
from transcription.config import Settings
from transcription.models import Segment
from transcription.transcriber import TranscribeMeta

if TYPE_CHECKING:
    import numpy as np

    from transcription.diarisation import RawTurn

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


class FakeDiariser:
    """Deterministic stand-in for a diarisation backend.

    Exists for the same reason ``FakeTranscriber`` does, and it is doing more work than it
    looks: no real diarisation backend ships with this repository, so the fake is what puts
    the parts that *are* implemented — the overlapping window grid, the global clustering, the
    midpoint de-duplication rule and the alignment onto segments — genuinely under test.

    ``turns_per_window`` is called with the window index and the samples, and returns
    window-relative turns. Driving it per window is the point: a backend that returns the same
    turns for every window is exactly how you catch a de-duplication bug.
    """

    def __init__(
        self,
        turns_per_window: Callable[[int, object], list[RawTurn]] | None = None,
        *,
        name: str = "fake-diariser",
        error: Exception | None = None,
    ) -> None:
        self._turns_per_window = turns_per_window or (lambda index, samples: [])
        self._name = name
        self._error = error
        self.window_count = 0
        self.warm_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def warm(self) -> None:
        self.warm_calls += 1

    def segment(self, samples) -> list[RawTurn]:
        if self._error is not None:
            raise self._error
        index = self.window_count
        self.window_count += 1
        return self._turns_per_window(index, samples)


def embedding(*values: float) -> np.ndarray:
    """A tiny speaker embedding.

    Two dimensions rather than 192 because nothing in the clustering depends on the
    dimension, and a 2-D vector is one a reader can reason about: ``embedding(1, 0)`` and
    ``embedding(0, 1)`` are a cosine distance of exactly 1.0 apart.
    """
    import numpy as np

    return np.array(values, dtype=np.float32)


# The asynchronous tier needs a real Postgres. Pointed by env var so CI and a developer's
# machine can differ, and *skipped* rather than failed when there is no database — someone
# cloning this to read the pipeline should still get a green suite without installing
# Postgres. CI sets the variable, so the tests genuinely run there.
TEST_DATABASE_URL = os.environ.get(
    "TRANSCRIBE_TEST_DATABASE_URL", "postgresql:///transcription_test"
)


def _database_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(TEST_DATABASE_URL, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


postgres_required = pytest.mark.skipif(
    not _database_available(),
    reason=f"no Postgres at {TEST_DATABASE_URL} (set TRANSCRIBE_TEST_DATABASE_URL)",
)


@pytest.fixture
def job_settings(tmp_path: Path) -> Iterator[Settings]:
    """Settings wired to the test database, with the tables emptied.

    Truncating per test rather than using a transaction-per-test rollback: the queue is the
    thing under test, and ``FOR UPDATE SKIP LOCKED`` behaves differently inside one shared
    transaction than it does across the separate connections a real worker uses. Testing it
    through a rollback harness would be testing a different system.
    """
    from transcription import db

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        blob_root=str(tmp_path / "blobs"),
        whisper_model="fake",
        jobs_enabled=True,
        job_max_attempts=3,
        job_lease_seconds=30.0,
        # Effectively every progress callback, so a test does not wait to see a beat.
        job_heartbeat_seconds=0.001,
        job_backoff_base_seconds=4.0,
        max_upload_bytes=8 * 1024 * 1024,
    )
    db.migrate(settings)
    with db.transaction(settings) as conn:
        conn.execute("TRUNCATE jobs CASCADE")
    yield settings
    with db.transaction(settings) as conn:
        conn.execute("TRUNCATE jobs CASCADE")


@pytest.fixture
def tenant() -> uuid.UUID:
    import uuid

    return uuid.uuid4()


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
