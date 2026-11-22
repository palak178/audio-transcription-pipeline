"""Runtime configuration.

Every knob is settable by environment variable with the ``TRANSCRIBE_`` prefix, so the
container and the test suite can reconfigure the service without code changes. For example
``TRANSCRIBE_WHISPER_MODEL=small`` or ``TRANSCRIBE_MAX_SYNC_DURATION_SECONDS=120``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model selection ------------------------------------------------------------------
    # `base` is the default deliberately: 74 MB to download and fast on CPU, so a reviewer's
    # first run finishes instead of looking like a hang. Use `small` or `medium` for accuracy.
    whisper_model: str = "base"
    device: str = "cpu"
    # int8 quantisation roughly halves CPU latency at a negligible accuracy cost.
    compute_type: str = "int8"
    download_root: str | None = None

    # --- Decoding -------------------------------------------------------------------------
    beam_size: int = 5
    # Voice-activity detection skips silence (faster) and, more importantly, suppresses
    # Whisper's well-known hallucination loop on long silent stretches.
    vad_filter: bool = True
    # Stops one bad segment from poisoning every segment after it on long recordings.
    condition_on_previous_text: bool = False

    # --- Long audio -----------------------------------------------------------------------
    # Audio is transcribed in windows of at most this many seconds. This is the control on
    # peak memory: handing the engine a whole file costs roughly 1.45 MB per second of audio
    # (measured), so a four-hour recording in one pass would need ~21 GB. Windowing caps it
    # regardless of total duration. Files shorter than one window are a single window, so
    # short audio has no seams at all.
    window_seconds: float = Field(default=300.0, gt=0)
    # How far back from a window's end to look for a quiet point to cut at, so boundaries
    # land in silence instead of through the middle of a word.
    window_search_seconds: float = Field(default=15.0, gt=0)

    # --- Diarisation ----------------------------------------------------------------------
    # Off by default, and opt-in per job rather than always on. Diarisation is estimated at
    # 1.7-2x the compute of a plain transcript, which roughly doubles a worker fleet for a
    # feature a minority of jobs want. The precedent is `word_timestamps`, opt-in for the same
    # reason at a fifth of the cost. See docs/DESIGN.md.
    diarise: bool = False
    # Which backend to load. No backend ships with this repository yet: `pyannote`'s weights
    # are gated on Hugging Face, which would break the offline, token-free, three-command
    # setup this project guarantees. The clustering, de-duplication and alignment around the
    # backend are implemented and tested; the embedding model is the missing piece.
    diarisation_backend: str = "none"
    # Diarisation reads the audio on its own fixed, overlapping grid rather than reusing the
    # silence-aligned transcription windows. See audio.iter_wav_grid for why.
    diarisation_window_seconds: float = Field(default=120.0, gt=0)
    # Any turn shorter than this is wholly contained in at least one window, so it is never
    # split into two fragments too short to embed.
    diarisation_overlap_seconds: float = Field(default=15.0, ge=0)
    # Cosine distance at which agglomerative clustering stops merging. This is *the* tunable,
    # and it is an unvalidated placeholder: choosing it honestly needs labelled audio this
    # repository does not have, which is why the accuracy benchmark is sequenced ahead of
    # diarisation in docs/DESIGN.md. Erring low over-splits one speaker into two, which a
    # human can merge in review; erring high merges two speakers, which nobody can undo
    # without re-running the job.
    diarisation_threshold: float = Field(default=0.55, gt=0, le=2.0)
    # Turns shorter than this are excluded from clustering and labelled from their temporal
    # neighbours instead. An embedding from 600 ms of speech is close to noise, and letting it
    # vote drags a cluster; inheriting a neighbour's label is wrong only locally.
    diarisation_min_embedding_seconds: float = Field(default=1.0, gt=0)

    # --- Limits ---------------------------------------------------------------------------
    # Checked against the *probed* duration before decoding, so a 1 MB file that expands to
    # six hours of audio is rejected rather than decoded.
    max_duration_seconds: float = Field(default=4 * 60 * 60, gt=0)
    # The synchronous HTTP endpoint refuses anything longer and points at the CLI. A documented
    # limit beats a silent gateway timeout; see docs/DESIGN.md for the asynchronous design.
    max_sync_duration_seconds: float = Field(default=600, gt=0)
    max_upload_bytes: int = Field(default=200 * 1024 * 1024, gt=0)

    # --- Concurrency ----------------------------------------------------------------------
    # How many requests may be inside the pipeline at once. This is admission control, not
    # parallel inference: faster-whisper defaults to num_workers=1, so CTranslate2 serializes
    # concurrent transcribe() calls on the shared model, and cpu_threads=0 already lets one
    # request use every core. What a value above 1 buys is overlap between one request's
    # ffmpeg decode and another's inference. See docs/DESIGN.md for real concurrency.
    max_concurrent_transcriptions: int = Field(default=2, ge=1)


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
