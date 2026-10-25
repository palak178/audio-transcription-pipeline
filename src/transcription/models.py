"""The output contract.

These models *are* the "processed for downstream use" part of the brief. A consumer should be
able to depend on this shape, so the result carries:

* ``schema_version`` — so the format can evolve without breaking existing consumers;
* ``id`` — the SHA-256 of the source bytes, which makes the same audio produce the same
  identifier on every run (useful for de-duplication, caching and idempotent retries);
* provenance — which model, which decoding parameters, how long it took;
* per-segment ``avg_logprob`` / ``confidence`` / ``no_speech_prob``, so a downstream system
  can rank transcripts and route the least certain to review rather than trusting every one
  equally. See :meth:`Segment.confidence` for what that number is and is not.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field

SCHEMA_VERSION = "1.0"

# A complete result, used as the example in the OpenAPI schema so the Swagger page shows
# callers what to expect before they run anything. Copied verbatim from a real run of
# `transcribe samples/speech_en.mp3` with the default `base` model, rather than hand-written,
# so it cannot drift into describing output the service does not produce. Note the two
# segments sharing an `avg_logprob`: that is faster-whisper reporting one value per 30-second
# decode window, which is why `confidence` is a ranking signal and not a per-segment
# probability.
_RESULT_EXAMPLE = {
    "schema_version": SCHEMA_VERSION,
    "id": "33b72396207f5be621d5b762cac764144cf23436a02475dc82423b3cc3a6ffbf",
    "created_at": "2026-10-24T09:30:00Z",
    "language": "en",
    "language_probability": 0.9973,
    "audio": {
        "filename": "speech_en.mp3",
        "sha256": "33b72396207f5be621d5b762cac764144cf23436a02475dc82423b3cc3a6ffbf",
        "size_bytes": 44111,
        "container_format": "mp3",
        "codec": "mp3",
        "duration_seconds": 7.245,
        "sample_rate": 22050,
        "channels": 1,
        "bit_rate": 48000,
    },
    "model": {
        "name": "base",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 5,
        "vad_filter": True,
        "word_timestamps": False,
    },
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 3.6,
            "duration": 3.6,
            "text": "quick brown fox jumps over the lazy dog.",
            "no_speech_prob": 0.0044,
            "avg_logprob": -0.2576,
            "confidence": 0.7729,
            "words": None,
        },
        {
            "id": 1,
            "start": 3.6,
            "end": 7.2,
            "duration": 3.6,
            "text": "This recording is a test fixture for the audio transcription service.",
            "no_speech_prob": 0.0044,
            "avg_logprob": -0.2576,
            "confidence": 0.7729,
            "words": None,
        },
    ],
    "text": (
        "quick brown fox jumps over the lazy dog. This recording is a test fixture for the "
        "audio transcription service."
    ),
    "processing_seconds": 0.686,
    "real_time_factor": 0.0947,
}


class Word(BaseModel):
    """A single word with aligned timings. Only present when word timestamps are requested."""

    start: float = Field(description="Seconds from the start of the recording.", examples=[0.3])
    end: float = Field(description="Seconds from the start of the recording.", examples=[0.64])
    word: str = Field(description="The word, including any leading space.", examples=[" brown"])
    probability: float | None = Field(
        default=None, description="Model confidence for this word, 0-1.", examples=[0.9722]
    )


class Segment(BaseModel):
    """A contiguous span of speech.

    Timestamps are in seconds from the start of the audio. Note that Whisper's segment
    boundaries come from decoder token positions and can drift by a few hundred
    milliseconds; passing ``word_timestamps=True`` runs a DTW alignment over the model's
    cross-attention, which tightens both word *and* segment edges at ~10-25% extra compute.
    """

    id: int = Field(description="Zero-based index, contiguous across the whole transcript.")
    start: float = Field(description="Segment start, in seconds.", examples=[0.0])
    end: float = Field(description="Segment end, in seconds.", examples=[3.6])
    text: str = Field(
        description="The transcribed text for this segment.",
        examples=["quick brown fox jumps over the lazy dog."],
    )
    no_speech_prob: float | None = Field(
        default=None,
        description=(
            "Probability this span contains no speech at all. A high value on a segment that "
            "still has text is the classic signature of a hallucination."
        ),
        examples=[0.0044],
    )
    avg_logprob: float | None = Field(
        default=None,
        description=(
            "Mean log-probability per token. `confidence` is this value exponentiated — a "
            "relative certainty signal, not a calibrated probability."
        ),
        examples=[-0.2576],
    )
    words: list[Word] | None = Field(
        default=None,
        description="Per-word timings. Only populated when `word_timestamps` is requested.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float | None:
        """Geometric mean of the per-token probabilities, i.e. ``exp(avg_logprob)``.

        A readable 0-1 restatement of ``avg_logprob``, **not** a calibrated probability that
        this text is correct. It is monotone in the model's own certainty, so it is useful
        for *ranking* segments — routing the weakest to human review — and not for
        thresholding in absolute terms. Note also that faster-whisper reports one
        ``avg_logprob`` per 30-second decode window, so adjacent segments often share a
        value.
        """
        if self.avg_logprob is None:
            return None
        return round(math.exp(self.avg_logprob), 4)


class AudioInfo(BaseModel):
    """What the input actually was, according to ffprobe rather than its file extension."""

    filename: str = Field(
        description="The uploaded name, sanitised. Used as a label only, never as a path.",
        examples=["speech_en.mp3"],
    )
    sha256: str = Field(description="SHA-256 of the source bytes.")
    size_bytes: int = Field(description="Size of the uploaded file.", examples=[44111])
    container_format: str | None = Field(
        default=None, description="Container as reported by ffprobe.", examples=["mp3"]
    )
    codec: str | None = Field(
        default=None,
        description="Real audio codec, detected from content rather than the extension.",
        examples=["mp3"],
    )
    duration_seconds: float = Field(description="Audio duration.", examples=[7.245])
    sample_rate: int | None = Field(
        default=None, description="Source sample rate, in Hz.", examples=[22050]
    )
    channels: int | None = Field(default=None, description="Source channel count.", examples=[1])
    bit_rate: int | None = Field(default=None, description="Source bit rate.", examples=[48000])


class ModelInfo(BaseModel):
    """Provenance: enough to reproduce this exact run."""

    name: str = Field(description="Whisper model size used.", examples=["base"])
    device: str = Field(description="`cpu` or `cuda`.", examples=["cpu"])
    compute_type: str = Field(description="Quantisation used.", examples=["int8"])
    beam_size: int = Field(description="Beam search width.", examples=[5])
    vad_filter: bool = Field(description="Whether silence was filtered out.", examples=[True])
    word_timestamps: bool = Field(description="Whether word timings were computed.")


class TranscriptionResult(BaseModel):
    """A complete transcription, designed to be consumed by another system."""

    model_config = ConfigDict(json_schema_extra={"example": _RESULT_EXAMPLE})

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Output format version, so this contract can evolve safely.",
    )
    id: str = Field(
        description=(
            "SHA-256 of the audio. Identical audio always produces the same id, which makes "
            "de-duplication, caching and idempotent retries possible."
        )
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When the result was produced."
    )

    language: str | None = Field(
        default=None, description="Detected (or forced) language code.", examples=["en"]
    )
    language_probability: float | None = Field(
        default=None, description="Confidence in the detected language.", examples=[0.9973]
    )

    audio: AudioInfo = Field(description="What the input actually was.")
    model: ModelInfo = Field(description="Which model and settings produced this.")
    segments: list[Segment] = Field(
        description="Timestamped segments, in order. Empty when the audio contains no speech."
    )

    processing_seconds: float = Field(
        description=(
            "Inference seconds only: excludes ffmpeg normalisation and the one-off model "
            "load, so the figure is comparable across runs."
        ),
        examples=[0.686],
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def text(self) -> str:
        """The full transcript as one string."""
        return " ".join(segment.text.strip() for segment in self.segments).strip()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def real_time_factor(self) -> float | None:
        """Processing seconds per second of audio. Lower is faster; <1.0 beats real time."""
        if self.audio.duration_seconds <= 0:
            return None
        return round(self.processing_seconds / self.audio.duration_seconds, 4)


class ErrorDetail(BaseModel):
    """The body of an error response.

    Extra keys are allowed because some errors carry context — ``audio_too_long`` adds
    ``duration_seconds`` and ``limit_seconds``, for example.
    """

    model_config = ConfigDict(extra="allow")

    code: str = Field(
        description="Stable, machine-readable error code. Match on this, not the message.",
        examples=["corrupt_audio"],
    )
    message: str = Field(
        description="Human-readable explanation.",
        examples=["Could not read this file as audio."],
    )


class ErrorResponse(BaseModel):
    """Every failure uses this shape, whatever the status code."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Liveness plus the limits a client needs to know before uploading."""

    status: str = Field(description="`ok` when the service is ready.", examples=["ok"])
    model: str = Field(description="Whisper model currently loaded.", examples=["base"])
    device: str = Field(description="`cpu` or `cuda`.", examples=["cpu"])
    max_sync_duration_seconds: float = Field(
        description="Longest audio this endpoint will accept. Longer files are rejected with 413.",
        examples=[600.0],
    )
    max_upload_bytes: int = Field(description="Largest accepted upload.", examples=[209715200])
    max_concurrent_transcriptions: int = Field(
        description="How many transcriptions run at once in this process.", examples=[2]
    )
