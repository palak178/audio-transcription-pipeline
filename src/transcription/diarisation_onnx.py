"""An ungated speaker-diarisation backend.

This is the model half of the diarisation design — the part
:mod:`transcription.diarisation` deliberately left out until there was a way to get it without
breaking the project's offline, no-account property.

**Why this backend rather than pyannote.** ``pyannote``'s weights are gated behind a Hugging
Face licence acceptance, so requiring them would mean every person who builds the image needs
their own accepted licence and a token. The weights here come from the `sherpa-onnx
<https://github.com/k2-fsa/sherpa-onnx>`_ release assets, which are a plain download. The
accuracy is lower. The setup still works for a stranger with a fresh clone, and that trade was
made deliberately.

**How it works**, and what each step costs:

1. **Voice activity detection** finds the speech. Reuses the Silero VAD that ``faster-whisper``
   already ships, so this adds no dependency for the step — and it is the same VAD the
   transcription path uses, so the two agree about where speech is.
2. **Sub-segmentation** cuts each speech region into fixed-length pieces. This is the honest
   weakness: ``pyannote`` has a neural model that finds speaker *change points*, and this does
   not, so turn boundaries land on a grid rather than where the speaker actually changed.
   Boundary resolution is the sub-segment length, about 1.5 s by default.
3. **Embedding** runs each piece through a CAM++ speaker-verification model to get a 512-dim
   vector.

Everything after that — global clustering, boundary de-duplication, alignment onto transcript
segments — is :mod:`transcription.diarisation`, unchanged. That was the point of building the
clustering first: adding a model really was one class.

**What this backend cannot do**, stated plainly rather than discovered later:

* **Overlapped speech is never detected.** ``speakers_overlap`` is always ``False`` here. A
  span where two people talk at once gets attributed to whichever voice the embedding resembles
  more, with no flag. ``pyannote``'s segmentation model does detect overlap; this pipeline has
  nothing that could.
* **Boundaries are quantised** to the sub-segment grid, as above.
* **A speaker who only ever speaks in bursts shorter than the minimum** will not get their own
  cluster, because there is never enough speech to embed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from transcription.audio import TARGET_SAMPLE_RATE
from transcription.config import Settings
from transcription.diarisation import RawTurn
from transcription.errors import DiarisationFailedError, DiarisationUnavailableError

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# The model this backend is developed and measured against. Any sherpa-onnx speaker-embedding
# model with an (N, T, 80) fbank input will work; the extractor reads the dimensions from the
# file, so nothing here is hard-coded to 512.
DEFAULT_MODEL_FILENAME = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
MODEL_RELEASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/" + DEFAULT_MODEL_FILENAME
)
# Checked after download. A speaker model is a file fetched over the network that then decides
# who the service says was speaking, so pinning the bytes is worth the one line.
MODEL_SHA256 = "98daf9bcf1c8abf3db374fec4c27a5eda7c58705bb3f180d057b085a72a88bb5"


class OnnxDiariser:
    """Silero VAD, fixed-length sub-segmentation, and CAM++ embeddings.

    Satisfies :class:`transcription.diarisation.Diariser`. Holds the ONNX session, so build one
    per process and reuse it — loading the model costs a second or so and 30 MB.
    """

    def __init__(self, settings: Settings, *, model_path: Path | str | None = None) -> None:
        self._settings = settings
        self._model_path = Path(model_path or resolve_model_path(settings))
        self._extractor = None
        self._vad_model = None

    @property
    def name(self) -> str:
        # Provenance has to identify the weights, not just the approach: swapping the model
        # changes the answer, and the result's `diarisation.backend` is what records that.
        return f"onnx-campplus/{self._model_path.stem}"

    def warm(self) -> None:
        """Load the ONNX session and the VAD, so the first real job does not pay for it."""
        self._ensure_extractor()
        self._ensure_vad()

    def _ensure_extractor(self):
        if self._extractor is not None:
            return self._extractor

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise DiarisationUnavailableError(
                "This backend needs sherpa-onnx. Install with "
                '`pip install -e ".[diarise]"`, or set TRANSCRIBE_DIARISATION_BACKEND=none.',
            ) from exc

        if not self._model_path.is_file():
            raise DiarisationUnavailableError(
                f"No speaker-embedding model at {self._model_path}. "
                "Fetch it with `transcribe diarise-warm`.",
                model_path=str(self._model_path),
            )

        try:
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(self._model_path),
                    num_threads=self._settings.diarisation_threads,
                    debug=False,
                )
            )
        except Exception as exc:
            raise DiarisationUnavailableError(
                f"Could not load the speaker-embedding model: {exc}",
                model_path=str(self._model_path),
            ) from exc

        logger.info(
            "speaker embeddings ready: %s (dim %d)", self._model_path.name, self._extractor.dim
        )
        return self._extractor

    def _ensure_vad(self):
        if self._vad_model is None:
            from faster_whisper.vad import get_vad_model

            # The same Silero model the transcription path uses, so the two stages agree about
            # where speech is rather than disagreeing at the edges.
            self._vad_model = get_vad_model()
        return self._vad_model

    def segment(self, samples: np.ndarray) -> list[RawTurn]:
        """Find speech, cut it into embeddable pieces, and embed each one.

        Times are relative to the start of ``samples``, per the protocol. Pieces shorter than
        the configured minimum are returned with ``embedding=None`` rather than dropped —
        :mod:`transcription.diarisation` labels those from their neighbours, which is a better
        answer than discarding the speech entirely.
        """

        extractor = self._ensure_extractor()
        self._ensure_vad()

        regions = self._speech_regions(samples)
        if not regions:
            return []

        sub_seconds = self._settings.diarisation_subsegment_seconds
        minimum = self._settings.diarisation_min_embedding_seconds
        turns: list[RawTurn] = []

        for region_start, region_end in regions:
            for start, end in _split(region_start, region_end, sub_seconds, minimum):
                piece = samples[
                    int(start * TARGET_SAMPLE_RATE) : int(end * TARGET_SAMPLE_RATE)
                ]
                if piece.size == 0:
                    continue

                embedding = None
                if (end - start) >= minimum:
                    embedding = self._embed(extractor, piece)

                turns.append(
                    RawTurn(
                        start=round(start, 3),
                        end=round(end, 3),
                        embedding=embedding,
                        # This backend has no overlap detection. Claiming otherwise would be
                        # worse than admitting it — see the module docstring.
                        overlap=False,
                    )
                )

        logger.debug(
            "window produced %d sub-segment(s) from %d region(s)", len(turns), len(regions)
        )
        return turns

    def _speech_regions(self, samples: np.ndarray) -> list[tuple[float, float]]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        options = VadOptions(
            min_speech_duration_ms=int(self._settings.diarisation_min_speech_ms),
            min_silence_duration_ms=int(self._settings.diarisation_min_silence_ms),
            # No upper bound: a long monologue is one region, and sub-segmentation below is
            # what breaks it up. Capping here would invent boundaries for no reason.
            max_speech_duration_s=float("inf"),
            speech_pad_ms=0,
        )
        try:
            chunks = get_speech_timestamps(samples, options, sampling_rate=TARGET_SAMPLE_RATE)
        except Exception as exc:
            raise DiarisationFailedError(f"Voice activity detection failed: {exc}") from exc

        return [
            (chunk["start"] / TARGET_SAMPLE_RATE, chunk["end"] / TARGET_SAMPLE_RATE)
            for chunk in chunks
        ]

    def _embed(self, extractor, piece: np.ndarray) -> np.ndarray:
        import numpy as np

        try:
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=TARGET_SAMPLE_RATE, waveform=piece)
            stream.input_finished()
            return np.asarray(extractor.compute(stream), dtype=np.float32)
        except Exception as exc:
            raise DiarisationFailedError(f"Speaker embedding failed: {exc}") from exc


def _split(
    start: float, end: float, sub_seconds: float, minimum: float
) -> list[tuple[float, float]]:
    """Cut ``[start, end)`` into pieces of about ``sub_seconds``.

    Non-overlapping, because the pieces become turns and overlapping turns would have to be
    de-duplicated all over again — which the window grid already does once and is quite enough.

    A short tail is absorbed into the previous piece rather than left as its own. A 0.3 s
    fragment cannot be embedded usefully, so leaving it standing would only create work for the
    short-turn fallback; folding it in gives the preceding piece slightly more speech, which
    makes its embedding marginally better.
    """
    total = end - start
    if total <= 0:
        return []
    if total < sub_seconds * 1.5:
        # Too short to divide sensibly. One piece, even if it is under the minimum — the caller
        # marks it unembeddable and it gets a neighbour's label.
        return [(start, end)]

    pieces: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        piece_end = min(cursor + sub_seconds, end)
        remaining = end - piece_end
        if 0 < remaining < minimum:
            piece_end = end  # absorb the tail
        pieces.append((cursor, piece_end))
        cursor = piece_end
    return pieces


def resolve_model_path(settings: Settings) -> Path:
    """Where the speaker-embedding model lives.

    Defaults next to the Whisper weights when a download root is configured, so one directory
    holds every model and one volume mount makes a container self-sufficient.
    """
    if settings.diarisation_model_path:
        return Path(settings.diarisation_model_path).expanduser()
    root = Path(settings.download_root).expanduser() if settings.download_root else _cache_root()
    return root / DEFAULT_MODEL_FILENAME


def _cache_root() -> Path:
    import os

    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "transcription"


def download_model(settings: Settings, *, force: bool = False) -> Path:
    """Fetch the speaker-embedding model, verifying its digest. Returns the path.

    Separate from loading, and never automatic. A service that silently downloads 30 MB on its
    first request is a service that fails its first request on an air-gapped host, which is
    precisely the deployment this project cares about. ``transcribe diarise-warm`` is the
    deliberate step, matching ``transcribe warm`` for the Whisper weights.
    """
    import hashlib
    import tempfile
    import urllib.request

    destination = resolve_model_path(settings)
    if destination.is_file() and not force:
        logger.info("speaker model already present at %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", MODEL_RELEASE_URL)

    # Download to a temporary file and verify before moving into place, so an interrupted or
    # tampered transfer never leaves something at the path that claims to be the model.
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".download-", delete=False
    ) as handle:
        staging = Path(handle.name)
        try:
            with urllib.request.urlopen(MODEL_RELEASE_URL, timeout=120) as response:
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 256):
                    digest.update(chunk)
                    handle.write(chunk)
        except Exception as exc:
            staging.unlink(missing_ok=True)
            raise DiarisationUnavailableError(
                f"Could not download the speaker model: {exc}", url=MODEL_RELEASE_URL
            ) from exc

    actual = digest.hexdigest()
    if actual != MODEL_SHA256:
        staging.unlink(missing_ok=True)
        raise DiarisationUnavailableError(
            "The downloaded speaker model does not match its expected digest. "
            f"Expected {MODEL_SHA256}, got {actual}.",
            url=MODEL_RELEASE_URL,
        )

    staging.replace(destination)
    logger.info("speaker model ready at %s", destination)
    return destination
