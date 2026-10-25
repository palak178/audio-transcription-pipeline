"""Audio inspection and format normalisation.

This module is the whole answer to "how do you handle different audio formats". The strategy
is two steps:

1. **Inspect with ffprobe, never with the file extension.** The extension is a hint supplied
   by the client and is frequently wrong (a ``.mp3`` that is really FLAC, or really a PDF).
   ffprobe reports the real container, codec, duration, sample rate and channel count.
2. **Normalise everything to one canonical format** — 16 kHz mono PCM s16le WAV — with a
   single ffmpeg call. 16 kHz mono is the only rate Whisper's feature extractor is defined
   for, and faster-whisper's own loader resamples to it too, so this is not an extra
   conversion so much as the same conversion moved into a step we own and test. It is not
   free: it costs a full decode pass and a PCM WAV on temporary disk (~115 MB per audio
   hour). What it buys is that no code after this point knows that codecs exist, so
   supporting a new input format becomes "does ffmpeg read it?" rather than a change here.

ffmpeg is invoked directly through ``subprocess`` with explicit arguments rather than through
a wrapper library. The arguments are the part worth being able to read and audit, and
shelling out keeps the dependency surface to a binary we already need.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from transcription.errors import (
    CorruptAudioError,
    EmptyAudioError,
    FFmpegNotFoundError,
    NoAudioStreamError,
    NormalizationError,
)
from transcription.models import AudioInfo

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

# Canonical internal format. 16 kHz mono is what Whisper consumes natively.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1

_PROBE_TIMEOUT_SECONDS = 60
# ffmpeg gets a wall-clock bound too. Without one, a file that makes the decoder stall holds
# an API concurrency slot indefinitely — and enough of those wedge the service. Generous,
# because a legitimate four-hour decode is only minutes of work.
_DECODE_TIMEOUT_SECONDS = 1800
_HASH_CHUNK_BYTES = 1024 * 1024


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise FFmpegNotFoundError(
            f"`{binary}` was not found on PATH. Install ffmpeg — macOS: `brew install "
            "ffmpeg`; Debian/Ubuntu: `apt-get install ffmpeg`; Fedora: `dnf install "
            "ffmpeg`; Windows: `winget install Gyan.FFmpeg` (then reopen your terminal)."
        )
    return path


def sha256_file(path: Path) -> str:
    """Content hash of the source bytes, streamed so memory stays flat on large files."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _ffprobe(path: Path) -> dict:
    ffprobe = _require("ffprobe")
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CorruptAudioError(
            f"ffprobe timed out after {_PROBE_TIMEOUT_SECONDS}s; the file does not look like "
            "readable media."
        ) from exc

    if completed.returncode != 0:
        raise CorruptAudioError(
            "Could not read this file as audio. It may be truncated, encrypted, or not a "
            "media file at all.",
            ffprobe_error=_tail(completed.stderr),
        )

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CorruptAudioError("ffprobe returned output that could not be parsed.") from exc


def _tail(text: str, limit: int = 400) -> str:
    """Last few hundred characters of ffmpeg stderr — enough to debug, short enough to log."""
    cleaned = " ".join(text.split())
    return cleaned[-limit:] if len(cleaned) > limit else cleaned


def _first_float(*values: object) -> float | None:
    for value in values:
        if value in (None, "", "N/A"):
            continue
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _first_int(*values: object) -> int | None:
    parsed = _first_float(*values)
    return int(parsed) if parsed is not None else None


def inspect(path: Path) -> AudioInfo:
    """Probe a file and describe what it really is.

    Raises :class:`EmptyAudioError`, :class:`CorruptAudioError` or
    :class:`NoAudioStreamError` rather than returning a partially-populated result, so
    callers never have to guess whether the input was usable.

    ``duration_seconds`` is ``0.0`` when the container does not declare a duration (some
    streamed OGG and raw files). That is not an error here: :func:`normalize` always produces
    a WAV with an authoritative duration, and the pipeline re-probes it. Callers that enforce
    a duration limit must therefore check again after normalising.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise CorruptAudioError(f"No such file: {path}")

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise EmptyAudioError("The file is empty (0 bytes).")

    probe = _ffprobe(path)
    streams = probe.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        kinds = sorted({str(s.get("codec_type")) for s in streams}) or ["none"]
        raise NoAudioStreamError(
            "This file contains no audio stream "
            f"(streams found: {', '.join(kinds)}). A video file with no audio track, perhaps."
        )

    stream = audio_streams[0]
    container = probe.get("format") or {}

    # Prefer the stream's own duration, fall back to the container's. Both can be absent.
    duration = _first_float(stream.get("duration"), container.get("duration")) or 0.0

    return AudioInfo(
        filename=path.name,
        sha256=sha256_file(path),
        size_bytes=size_bytes,
        container_format=container.get("format_name"),
        codec=stream.get("codec_name"),
        duration_seconds=round(duration, 3),
        sample_rate=_first_int(stream.get("sample_rate")),
        channels=_first_int(stream.get("channels")),
        bit_rate=_first_int(stream.get("bit_rate"), container.get("bit_rate")),
    )


def normalize(
    source: Path,
    destination: Path,
    *,
    max_duration_seconds: float | None = None,
) -> Path:
    """Decode any ffmpeg-readable input to 16 kHz mono PCM s16le WAV.

    ``max_duration_seconds`` is passed to ffmpeg as ``-t`` with a one-second margin, which
    hard-bounds how much audio can be written. That matters for containers that do not declare
    a duration: without it, a small file that decodes into hours of audio would be fully
    expanded on disk before anything could reject it.
    """
    ffmpeg = _require("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-nostdin",  # never block waiting on stdin
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",  # drop any video stream
        "-map",
        "0:a:0",  # first audio stream only; ignores commentary/secondary tracks
        "-ac",
        str(TARGET_CHANNELS),  # downmix to mono
        "-ar",
        str(TARGET_SAMPLE_RATE),  # resample 8 kHz telephony or 48 kHz studio alike
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
    ]
    if max_duration_seconds is not None:
        command += ["-t", f"{max_duration_seconds + 1:.3f}"]
    command.append(str(destination))

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_DECODE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NormalizationError(
            f"ffmpeg did not finish decoding within {_DECODE_TIMEOUT_SECONDS}s; the file may "
            "be malformed in a way that stalls the decoder."
        ) from exc

    if completed.returncode != 0:
        raise NormalizationError(
            "ffmpeg could not decode this audio.",
            ffmpeg_error=_tail(completed.stderr),
        )
    if not destination.exists() or destination.stat().st_size == 0:
        raise NormalizationError("ffmpeg produced no output; the audio may contain no samples.")

    return destination


def read_wav_mono_f32(path: Path) -> np.ndarray:
    """Load a normalised WAV as float32 samples in [-1.0, 1.0], in one shot.

    Whisper wants a mono 16 kHz float array, and :func:`normalize` has already produced
    exactly that as 16-bit PCM, so this is a cheap byte-level conversion.

    The pipeline itself uses :func:`iter_wav_windows`, which bounds memory on long audio.
    This is the whole-file form, for callers (and tests) that want the entire array and know
    the file is short.
    """
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2 or channels != TARGET_CHANNELS:
        raise NormalizationError(
            "Normalised audio was not 16-bit mono as expected "
            f"(got {channels} channel(s), {sample_width * 8}-bit)."
        )
    if frame_rate != TARGET_SAMPLE_RATE:
        raise NormalizationError(
            f"Normalised audio was {frame_rate} Hz, expected {TARGET_SAMPLE_RATE} Hz."
        )

    # int16 -> float32 in [-1, 1); 32768 is the full-scale divisor for signed 16-bit.
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _open_normalized_wav(path: Path) -> wave.Wave_read:
    """Open a normalised WAV, asserting the contract :func:`normalize` guarantees."""
    # Deliberately not a context manager: the caller streams from this handle and owns
    # closing it. iter_wav_windows() does so in a finally block.
    handle = wave.open(str(path), "rb")  # noqa: SIM115
    try:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_rate = handle.getframerate()
    except Exception:
        handle.close()
        raise

    if sample_width != 2 or channels != TARGET_CHANNELS:
        handle.close()
        raise NormalizationError(
            "Normalised audio was not 16-bit mono as expected "
            f"(got {channels} channel(s), {sample_width * 8}-bit)."
        )
    if frame_rate != TARGET_SAMPLE_RATE:
        handle.close()
        raise NormalizationError(
            f"Normalised audio was {frame_rate} Hz, expected {TARGET_SAMPLE_RATE} Hz."
        )
    return handle


def _quietest_offset(samples: np.ndarray, search_frames: int, frame_length: int) -> int:
    """Index of the quietest short frame within the last ``search_frames`` samples.

    Used to pick a window boundary that falls in silence (or, failing that, in the quietest
    audio available) rather than through the middle of a word.
    """
    import numpy as np

    region_start = max(0, len(samples) - search_frames)
    region = samples[region_start:]
    usable = (len(region) // frame_length) * frame_length
    if usable == 0:
        return len(samples)

    # Mean square per frame; argmin is the quietest frame.
    frames = region[:usable].reshape(-1, frame_length)
    energies = np.einsum("ij,ij->i", frames, frames)
    quietest = int(np.argmin(energies))
    # Cut in the middle of the quiet frame, for maximum margin on either side.
    return region_start + quietest * frame_length + frame_length // 2


def iter_wav_windows(
    path: Path,
    *,
    window_seconds: float,
    search_seconds: float,
    silence_frame_seconds: float = 0.1,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(offset_seconds, samples)`` windows of a normalised WAV.

    This is what makes long audio safe, and it exists because of a measurement rather than a
    hunch. Handing the engine one array for the whole file uses memory linear in duration —
    measured at roughly 1.45 MB per second of audio, so a four-hour recording would need
    about 21 GB and be killed. Reading in windows caps peak memory at the window size,
    independently of how long the recording is.

    Windows are cut at the **quietest point** within the final ``search_seconds`` of each
    window, not at a fixed offset. Cutting in silence is what makes the seams harmless: no
    word is split, so there is nothing to overlap, de-duplicate or re-align afterwards, and
    segment timestamps stay exact because each window's results are simply shifted by its
    offset. Audio with no pause at all in the search band gets cut at its quietest moment,
    which may clip one word every ``window_seconds``; that is the documented worst case.

    Files shorter than one window yield exactly one window, so short audio takes the same
    code path with no special-casing and no seams.

    Reading the samples ourselves and handing the engine an array — rather than handing it a
    file path — is a deliberate decision. It keeps *all* decoding in one place behind ffmpeg,
    avoids a redundant second decode of the same audio, and decouples us from the engine's
    own audio stack. That last point is not hypothetical: faster-whisper decodes via PyAV,
    and PyAV 19 removed a keyword argument that faster-whisper still passes, so
    ``faster_whisper.audio.decode_audio`` raises ``TypeError`` on current dependency
    versions. Owning the decode step means that breakage cannot reach us, and no version
    pin was needed.
    """
    import numpy as np

    handle = _open_normalized_wav(path)
    try:
        rate = handle.getframerate()
        total_frames = handle.getnframes()
        # Floor of one second: a misconfigured tiny window would otherwise degenerate into
        # advancing a frame at a time, which on a long file is effectively a hang.
        window_frames = max(rate, int(window_seconds * rate))
        # Never search more than half the window, or boundaries could march backwards.
        search_frames = min(max(1, int(search_seconds * rate)), window_frames // 2)
        frame_length = max(1, int(silence_frame_seconds * rate))

        position = 0
        while position < total_frames:
            handle.setpos(position)
            raw = handle.readframes(window_frames)
            if not raw:
                break

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # Last window: everything remaining, no cut needed.
            if position + len(samples) >= total_frames:
                yield position / rate, samples
                return

            cut = _quietest_offset(samples, search_frames, frame_length)
            cut = min(max(cut, 1), len(samples))
            yield position / rate, samples[:cut]
            position += cut
    finally:
        handle.close()


def probe_duration(path: Path) -> float:
    """Authoritative duration of an already-normalised WAV, in seconds."""
    probe = _ffprobe(path)
    streams = probe.get("streams") or []
    container = probe.get("format") or {}
    stream_duration = streams[0].get("duration") if streams else None
    return round(_first_float(stream_duration, container.get("duration")) or 0.0, 3)
