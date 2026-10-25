"""Audio transcription pipeline: audio in, timestamped segments out."""

from transcription.models import (
    SCHEMA_VERSION,
    AudioInfo,
    ModelInfo,
    Segment,
    TranscriptionResult,
    Word,
)
from transcription.pipeline import transcribe_file

__all__ = [
    "SCHEMA_VERSION",
    "AudioInfo",
    "ModelInfo",
    "Segment",
    "TranscriptionResult",
    "Word",
    "transcribe_file",
]
