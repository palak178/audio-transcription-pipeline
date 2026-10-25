"""Speech-to-text backends.

The engine sits behind a :class:`Transcriber` protocol for two reasons. The first is
practical: the test suite swaps in a fake, so the whole fast suite runs in seconds without
downloading model weights. The second is architectural: swapping in a hosted API
(Deepgram, OpenAI, AssemblyAI) means writing one more class, not touching the pipeline.

**faster-whisper** is the default implementation. It is a CTranslate2 reimplementation of
OpenAI's Whisper; the upstream project reports roughly 4x the throughput of the reference
PyTorch implementation on CPU at equal accuracy (their benchmark, not measured here). It runs
fully offline with no API key and exposes both segment and DTW-aligned word timestamps.
Running locally also means audio never leaves the host, which matters because voice
recordings are biometric personal data.

Two interface details worth knowing:

* :meth:`Transcriber.transcribe` takes mono 16 kHz float32 *samples*, not a path. All file
  handling therefore stays in :mod:`transcription.audio` — including the windowed reading
  that is what actually bounds memory on long audio.
* It returns an *iterator* of segments, not a list. faster-whisper decodes lazily as you
  consume it, which is what lets the caller report progress per segment instead of per
  window. The pipeline does collect the shifted segments into a list, because a transcript
  is small; the laziness buys progress reporting, not the memory bound.

On concurrency: a single :class:`WhisperModel` is created per configuration and shared, which
is safe but **not parallel**. faster-whisper defaults to ``num_workers=1``, i.e. CTranslate2
``inter_threads=1``, so concurrent ``transcribe()`` calls on one model serialize. That is
close to the right behaviour on CPU anyway, since ``cpu_threads=0`` already lets a single
request use every core. See docs/DESIGN.md for how real concurrency is reached.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from transcription.config import Settings
from transcription.errors import TranscriptionFailedError
from transcription.models import Segment, Word

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class TranscribeMeta:
    """What the engine detected about the audio as a whole."""

    language: str | None
    language_probability: float | None
    duration: float | None


class Transcriber(Protocol):
    """A speech-to-text engine."""

    @property
    def name(self) -> str:
        """Identifier recorded in the result's provenance block."""
        ...

    def warm(self) -> None:
        """Make the engine ready to decode.

        Part of the interface because loading weights costs seconds, and the caller needs to
        be able to pay that cost deliberately — at process start, and outside any timing
        that is meant to measure inference.
        """
        ...

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> tuple[TranscribeMeta, Iterator[Segment]]:
        """Transcribe mono 16 kHz float32 samples.

        Takes samples rather than a path so that every form of file handling — decoding,
        and the windowed reading that bounds memory on long audio — stays in
        :mod:`transcription.audio`. Engines are then responsible for inference and nothing
        else, which is also what makes them trivial to fake in tests.

        Returns metadata immediately and segments as a lazy iterator. Timestamps are
        relative to the start of ``samples``; the pipeline shifts them by the window offset.
        """
        ...


# Loading a model costs several seconds and 0.5-1.5 GB of RAM, so it must happen once per
# process and never per request. Cached by configuration so that overriding the model in a
# single call does not evict the default.
_model_cache: dict[tuple[str, str, str, str | None], object] = {}
_model_cache_lock = threading.Lock()


def _load_model(settings: Settings, model_name: str):
    key = (model_name, settings.device, settings.compute_type, settings.download_root)
    with _model_cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            return cached

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise TranscriptionFailedError(
                "faster-whisper is not installed. Run `make setup`."
            ) from exc

        try:
            model = WhisperModel(
                model_name,
                device=settings.device,
                compute_type=settings.compute_type,
                download_root=settings.download_root,
            )
        except Exception as exc:
            raise TranscriptionFailedError(f"Could not load model `{model_name}`: {exc}") from exc

        _model_cache[key] = model
        return model


class FasterWhisperTranscriber:
    """faster-whisper backed implementation of :class:`Transcriber`."""

    def __init__(self, settings: Settings, model_name: str | None = None) -> None:
        self._settings = settings
        self._model_name = model_name or settings.whisper_model

    @property
    def name(self) -> str:
        return self._model_name

    def warm(self) -> None:
        """Load the weights eagerly, so the first real request does not pay for it."""
        _load_model(self._settings, self._model_name)

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> tuple[TranscribeMeta, Iterator[Segment]]:
        model = _load_model(self._settings, self._model_name)
        settings = self._settings

        try:
            raw_segments, info = model.transcribe(  # type: ignore[attr-defined]
                samples,
                language=language,
                beam_size=settings.beam_size,
                vad_filter=settings.vad_filter,
                word_timestamps=word_timestamps,
                condition_on_previous_text=settings.condition_on_previous_text,
            )
        except Exception as exc:
            raise TranscriptionFailedError(f"Transcription failed: {exc}") from exc

        meta = TranscribeMeta(
            language=getattr(info, "language", None),
            language_probability=_round(getattr(info, "language_probability", None)),
            duration=_round(getattr(info, "duration", None)),
        )
        return meta, _adapt(raw_segments)


def _adapt(raw_segments: Iterator) -> Iterator[Segment]:
    """Map faster-whisper segments onto our schema, lazily.

    This is a generator on purpose: wrapping the engine's lazy iterator in another lazy
    iterator preserves the laziness end to end, which is what lets the pipeline report
    progress per segment rather than per window.
    """
    try:
        for index, raw in enumerate(raw_segments):
            words = None
            if getattr(raw, "words", None):
                words = [
                    Word(
                        start=round(word.start, 3),
                        end=round(word.end, 3),
                        word=word.word,
                        probability=_round(getattr(word, "probability", None)),
                    )
                    for word in raw.words
                ]

            yield Segment(
                # Index ourselves rather than trusting the engine's ids, which are not
                # guaranteed to be contiguous once the VAD filter drops silent spans.
                id=index,
                start=round(raw.start, 3),
                end=round(raw.end, 3),
                text=raw.text.strip(),
                no_speech_prob=_round(getattr(raw, "no_speech_prob", None)),
                avg_logprob=_round(getattr(raw, "avg_logprob", None)),
                words=words,
            )
    except TranscriptionFailedError:
        raise
    except Exception as exc:
        # Decoding happens during iteration, so model failures surface here rather than at
        # the original transcribe() call.
        raise TranscriptionFailedError(f"Transcription failed while decoding: {exc}") from exc


def _round(value: float | None, places: int = 4) -> float | None:
    return None if value is None else round(float(value), places)
