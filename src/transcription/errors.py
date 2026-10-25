"""Typed domain errors.

Every failure mode a caller can reasonably act on gets its own class carrying a stable
machine-readable ``code`` and the HTTP status it maps to. That keeps the API layer free of
error-classification logic: it catches one base class and reads two attributes.

The split that matters operationally is permanent vs. transient. Everything here except
:class:`TranscriptionFailedError` is permanent — the same input will fail the same way, so
retrying is pointless. See docs/DESIGN.md for how that distinction drives retry policy.
"""

from __future__ import annotations


class TranscriptionError(Exception):
    """Base class for every error this package raises deliberately."""

    code = "transcription_error"
    http_status = 500
    retryable = False

    # Detail keys that are useful for debugging but must not be returned over HTTP. Raw
    # ffmpeg stderr names the temporary file it was given, which would leak server paths to
    # whoever uploaded a malformed file. These stay on the exception for logging.
    internal_detail_keys = frozenset({"ffprobe_error", "ffmpeg_error"})

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, object]:
        """Client-safe representation."""
        public = {
            key: value
            for key, value in self.details.items()
            if key not in self.internal_detail_keys
        }
        return {"error": {"code": self.code, "message": self.message, **public}}


class FFmpegNotFoundError(TranscriptionError):
    """ffmpeg/ffprobe is not on PATH. A deployment problem, not a user problem."""

    code = "ffmpeg_not_found"
    http_status = 500


class EmptyAudioError(TranscriptionError):
    """Zero-byte upload."""

    code = "empty_audio"
    http_status = 400


class CorruptAudioError(TranscriptionError):
    """ffprobe could not parse the file: truncated, not media, or an unknown codec."""

    code = "corrupt_audio"
    http_status = 415


class NoAudioStreamError(TranscriptionError):
    """Parsed fine but carries no audio track, e.g. a silent-video container."""

    code = "no_audio_stream"
    http_status = 415


class AudioTooLongError(TranscriptionError):
    """Duration exceeds the configured limit."""

    code = "audio_too_long"
    http_status = 413

    def __init__(self, duration_seconds: float, limit_seconds: float, hint: str = "") -> None:
        message = (
            f"Audio is {duration_seconds:.1f}s long, which exceeds the "
            f"{limit_seconds:.1f}s limit."
        )
        if hint:
            message = f"{message} {hint}"
        super().__init__(
            message,
            duration_seconds=round(duration_seconds, 3),
            limit_seconds=limit_seconds,
        )


class UploadTooLargeError(TranscriptionError):
    """Upload exceeds the configured byte limit."""

    code = "upload_too_large"
    http_status = 413


class NormalizationError(TranscriptionError):
    """ffmpeg failed while decoding to the canonical WAV format."""

    code = "normalization_failed"
    # 415, not 422: this means "we cannot decode this media", which is the same family as
    # `corrupt_audio`. Reserving 422 for request-validation failures keeps one status from
    # carrying two unrelated meanings, which a client would otherwise have to disambiguate.
    http_status = 415


class TranscriptionFailedError(TranscriptionError):
    """The model itself failed. Transient — out of memory, a killed subprocess, a bad load."""

    code = "transcription_failed"
    http_status = 500
    retryable = True
