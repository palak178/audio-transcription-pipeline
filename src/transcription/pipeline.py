"""Pipeline orchestration: inspect -> normalise -> window -> transcribe -> assemble.

This is also where "how do you deal with long audio files" is answered. The answer is
application-level, silence-aligned *windowing* — not streaming, and not reliance on the
engine's internal 30-second window.

The first design had no windowing, on the grounds that faster-whisper already slides a
30-second window across a file of any length. Measuring peak RSS against audio duration
showed why that was not enough: memory grows linearly at roughly 1.45 MB per second of
audio, because the engine materialises the whole waveform and its mel features before
decoding. At the four-hour limit configured at the time that extrapolates to ~21 GB.

So:

* :func:`transcription.audio.iter_wav_windows` reads the normalised WAV in windows, cutting
  each one at the quietest short frame in its final ``window_search_seconds``. Peak memory
  becomes a function of the window, not of the recording.
* Cutting in silence is what keeps the seams harmless: no word is split, so there is nothing
  to overlap, de-duplicate or re-align, and rebasing a window's timestamps onto the whole
  recording is a plain shift by its offset (see :func:`_shift`).
* Language is detected on the first window and pinned for the rest, so a mostly-silent
  window cannot flip language mid-recording.
* Duration is checked twice. Once from the ffprobe metadata *before* any decoding, which is
  what stops a small file that expands into hours of audio from being decoded at all; and
  once against the normalised WAV, which is authoritative even for containers that declare
  no duration. ffmpeg itself is also given a hard ``-t`` bound.
* Within a window, segments are consumed from a lazy iterator, which is what lets the
  ``progress`` callback fire per segment rather than per window — a caller waiting ten
  minutes needs to know the difference between "working" and "hung".

The remaining constraint is wall-clock time, which is a scheduling problem rather than a
pipeline problem: see docs/DESIGN.md.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from transcription import audio as audio_module
from transcription import diarisation as diarisation_module
from transcription.checkpoint import EMPTY_RESUME, Checkpointer
from transcription.config import Settings, get_settings
from transcription.diarisation import Diariser, SpeakerHint
from transcription.errors import AudioTooLongError, DiarisationError
from transcription.models import (
    AudioInfo,
    DiarisationInfo,
    ModelInfo,
    Segment,
    Speaker,
    TranscriptionResult,
)
from transcription.transcriber import FasterWhisperTranscriber, Transcriber

# Called as progress(seconds_transcribed, total_seconds).
ProgressCallback = Callable[[float, float], None]


def safe_filename(name: str | None, fallback: str = "upload") -> str:
    """Reduce a client-supplied filename to something safe to store and echo back.

    Client filenames are untrusted input. They are used here only as a display label — never
    to build a path — but stripping directory components and control characters keeps a
    ``../../etc/passwd`` style name from showing up in logs or output files.
    """
    if not name:
        return fallback
    base = Path(name.replace("\\", "/")).name
    cleaned = "".join(char for char in base if char.isprintable() and char not in '/:*?"<>|')
    cleaned = cleaned.strip().lstrip(".")
    return cleaned[:255] or fallback


def transcribe_file(
    source: Path,
    *,
    transcriber: Transcriber | None = None,
    settings: Settings | None = None,
    language: str | None = None,
    word_timestamps: bool = False,
    progress: ProgressCallback | None = None,
    max_duration_seconds: float | None = None,
    display_name: str | None = None,
    diarise: bool | None = None,
    diariser: Diariser | None = None,
    speakers: SpeakerHint | None = None,
    checkpointer: Checkpointer | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file of any ffmpeg-readable format.

    Args:
        source: Path to the input audio.
        transcriber: Engine to use. Defaults to faster-whisper per ``settings``.
        settings: Configuration. Defaults to the process-wide settings.
        language: ISO language code to force. ``None`` auto-detects.
        word_timestamps: Also emit DTW-aligned per-word timings (~10-25% slower). Forced on
            by ``diarise``, because speaker alignment is only as good as the timestamps it
            aligns against.
        progress: Called after each segment with (seconds_done, total_seconds).
        max_duration_seconds: Duration limit. Defaults to ``settings.max_duration_seconds``.
        display_name: Filename to record in the result, for when ``source`` is a temp file.
        diarise: Also label who spoke each segment. Defaults to ``settings.diarise``.
            Estimated at 1.7-2x the compute of a plain transcript, which is why it is opt-in.
        diariser: Diarisation backend. Defaults to the one named in ``settings``.
        speakers: What the caller knows about the speaker count. A known count is the
            cheapest accuracy available here.
        checkpointer: Records each window as it completes and skips windows a previous
            attempt already finished. Turns a worker lost three hours into a four-hour file
            from an hour of wasted CPU into a re-read. See :mod:`transcription.checkpoint`.

    Raises:
        TranscriptionError: Any subclass; see :mod:`transcription.errors`.

    A diarisation failure is **not** raised. The transcript is the expensive, primary
    artefact, so the stage degrades to unlabelled segments and records why in
    ``result.diarisation``. Check ``result.diarisation.status`` rather than assuming labels.
    """
    settings = settings or get_settings()
    transcriber = transcriber or FasterWhisperTranscriber(settings)
    limit = (
        max_duration_seconds if max_duration_seconds is not None else settings.max_duration_seconds
    )
    diarise = settings.diarise if diarise is None else diarise
    # Alignment is only as good as the timestamps underneath it. Whisper's segment edges come
    # from decoder token positions and drift by a few hundred milliseconds; word edges are DTW
    # aligned to the audio. At a turn change that drift is exactly what mislabels the words on
    # either side, so diarisation pays the ~10-25% for word timings rather than silently
    # aligning against the looser boundaries.
    word_timestamps = word_timestamps or diarise

    source = Path(source)
    info = audio_module.inspect(source)
    if display_name:
        info = info.model_copy(update={"filename": safe_filename(display_name)})

    # First duration gate, on metadata alone. Rejects before we spend CPU on decoding.
    if info.duration_seconds > limit:
        raise AudioTooLongError(info.duration_seconds, limit)

    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp_dir:
        wav_path = audio_module.normalize(
            source,
            Path(tmp_dir) / "normalized.wav",
            max_duration_seconds=limit,
        )

        # Second duration gate, on the decoded audio. Authoritative: it catches inputs whose
        # container under-reported or omitted the duration.
        actual_duration = audio_module.probe_duration(wav_path)
        if actual_duration > limit:
            raise AudioTooLongError(actual_duration, limit)
        if actual_duration > 0:
            info = info.model_copy(update={"duration_seconds": actual_duration})

        # Load weights before starting the clock. Model loading is a one-off process cost of
        # several seconds; counting it as transcription time would make the first request's
        # reported real-time factor meaningless.
        transcriber.warm()

        started = time.perf_counter()
        total = info.duration_seconds

        # Anything a previous attempt finished. Empty unless a checkpointer was passed and it
        # has usable state — see checkpoint.py for what "usable" means.
        resume = checkpointer.load() if checkpointer is not None else EMPTY_RESUME
        segments: list[Segment] = list(resume.segments)
        detected_language: str | None = resume.language
        language_probability: float | None = resume.language_probability

        windows = audio_module.iter_wav_windows(
            wav_path,
            window_seconds=settings.window_seconds,
            search_seconds=settings.window_search_seconds,
        )

        for window_index, (offset, samples) in enumerate(windows):
            # Already transcribed on an earlier attempt. The window still had to be *read* —
            # boundaries are content-dependent, so there is no way to seek to window N without
            # walking the ones before it — but the expensive part, inference, is skipped.
            if window_index in resume.completed_windows:
                if progress is not None and total:
                    progress(min(offset + len(samples) / 16_000, total), total)
                continue

            window_start = len(segments)

            # After the first window, pin the language. Detecting per window lets a window
            # that happens to be mostly silence or noise come back as a different language,
            # which then produces nonsense for that stretch of an otherwise fine recording.
            meta, segment_iterator = transcriber.transcribe(
                samples,
                language=language or detected_language,
                word_timestamps=word_timestamps,
            )
            if detected_language is None:
                detected_language = language or meta.language
                language_probability = meta.language_probability

            for segment in segment_iterator:
                # Window-relative timestamps become absolute here. Because windows are cut in
                # silence there is no overlap to reconcile: a plain shift is exact.
                segments.append(_shift(segment, offset, index=len(segments)))
                if progress is not None:
                    done = segments[-1].end
                    progress(min(done, total) if total else done, total)

            # Checkpoint *after* the window is fully consumed, never during. A partial window
            # recorded as complete would be silently truncated on resume — the one failure
            # this whole mechanism must not introduce.
            if checkpointer is not None:
                checkpointer.save(
                    window_index=window_index,
                    offset_seconds=offset,
                    segments=segments[window_start:],
                    language=detected_language,
                    language_probability=language_probability,
                )

        processing_seconds = time.perf_counter() - started

        # A completed run reports 100%, even if the last speech ended before the audio did.
        if progress is not None and total:
            progress(total, total)

        # Diarisation is a second pass over the same normalised WAV, inside the same temporary
        # directory and the same job. That is the reason the two stages stay together: the WAV
        # is deliberately never stored, so splitting them across workers would mean either
        # re-running ffmpeg or reversing that decision. See docs/DESIGN.md.
        #
        # It runs *after* transcription, not before or alongside. Alongside gains nothing —
        # both stages are CPU-bound and would contend for the same cores, which is the same
        # reason windows are not transcribed in parallel. Before would let turn boundaries cut
        # the transcription windows, which is a genuinely better cut, but it couples the two
        # passes and makes a diarisation failure cost the transcript.
        speaker_roster: list[Speaker] | None = None
        diarisation_info = DiarisationInfo(status="not_requested")

        if diarise:
            segments, speaker_roster, diarisation_info = _diarise_stage(
                wav_path,
                segments=segments,
                diariser=diariser,
                settings=settings,
                audio_duration=info.duration_seconds,
                speakers=speakers,
            )

    return _build_result(
        info=info,
        meta_language=detected_language,
        meta_language_probability=language_probability,
        segments=segments,
        settings=settings,
        model_name=transcriber.name,
        word_timestamps=word_timestamps,
        processing_seconds=processing_seconds,
        speakers=speaker_roster,
        diarisation=diarisation_info,
    )


def _diarise_stage(
    wav_path: Path,
    *,
    segments: list[Segment],
    diariser: Diariser | None,
    settings: Settings,
    audio_duration: float,
    speakers: SpeakerHint | None,
) -> tuple[list[Segment], list[Speaker] | None, DiarisationInfo]:
    """Run diarisation and align it, degrading to an unlabelled transcript on failure.

    This is the one error family the pipeline swallows. Everything else — a corrupt file, a
    failed decode, a model crash — means there is no transcript to return, so there is nothing
    to do but raise. Here there *is* a transcript, and it is the expensive part; discarding it
    because a cheaper downstream stage failed is the wrong trade. The failure is recorded with
    a stable code rather than hidden, and in production the same distinction drives the retry
    policy: a diarisation stage that exhausts its retries completes the job as a partial
    success rather than dead-lettering it.
    """
    try:
        backend = diariser or diarisation_module.load_diariser(settings)
        backend.warm()
        result = diarisation_module.diarise_wav(
            wav_path,
            diariser=backend,
            settings=settings,
            audio_duration=audio_duration,
            hint=speakers,
        )
    except DiarisationError as exc:
        return segments, None, DiarisationInfo(status="failed", error_code=exc.code)

    aligned = diarisation_module.assign_speakers(segments, result.turns)
    return aligned, result.speakers, result.info


def _shift(segment: Segment, offset: float, *, index: int) -> Segment:
    """Rebase a window-relative segment onto the whole recording's timeline."""
    if offset == 0 and segment.id == index:
        return segment

    words = segment.words
    if words:
        words = [
            word.model_copy(
                update={
                    "start": round(word.start + offset, 3),
                    "end": round(word.end + offset, 3),
                }
            )
            for word in words
        ]

    return segment.model_copy(
        update={
            # Renumber across windows so ids stay contiguous over the whole transcript.
            "id": index,
            "start": round(segment.start + offset, 3),
            "end": round(segment.end + offset, 3),
            "words": words,
        }
    )


def _build_result(
    *,
    info: AudioInfo,
    meta_language: str | None,
    meta_language_probability: float | None,
    segments: list[Segment],
    settings: Settings,
    model_name: str,
    word_timestamps: bool,
    processing_seconds: float,
    speakers: list[Speaker] | None = None,
    diarisation: DiarisationInfo | None = None,
) -> TranscriptionResult:
    return TranscriptionResult(
        id=info.sha256,
        language=meta_language,
        language_probability=meta_language_probability,
        audio=info,
        model=ModelInfo(
            name=model_name,
            device=settings.device,
            compute_type=settings.compute_type,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
            word_timestamps=word_timestamps,
        ),
        segments=segments,
        speakers=speakers,
        diarisation=diarisation,
        processing_seconds=round(processing_seconds, 3),
    )
