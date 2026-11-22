"""Resuming a long transcription from where it stopped.

A four-hour recording is roughly an hour of CPU. Losing a worker at the three-hour mark and
starting again from zero is the single most expensive failure mode this system has, and it is
also the easiest to fix — because the pipeline already divides the audio into windows whose
results are complete and absolute the moment each one finishes.

So each window's segments are persisted as it completes, and a retry skips the windows that
already have results.

**What is re-done on resume, honestly.** The audio is re-normalised by ffmpeg and re-read
window by window, because window boundaries are content-dependent: each cut lands at the
quietest point in its search band, so the only way to know where window 7 starts is to walk
windows 1 to 6. That is disk and a decode pass — seconds. What is skipped is inference, which
is the part measured in minutes. The saving is not total, it is the part that matters.

**Why that is safe.** Resuming assumes window boundaries are reproducible. They are:
``ffmpeg`` is deterministic for the same input and arguments, and ``_quietest_offset`` is a
pure function of the samples. But they are only reproducible if the inputs that decide them
have not changed, which is what ``plan_fingerprint`` checks. Change the window size and
window 7 is a different stretch of audio; the checkpoints are then discarded rather than
silently stitched into a wrong transcript.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Protocol

from transcription.config import Settings
from transcription.models import Segment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumePoint:
    """What a previous attempt got through."""

    completed_windows: frozenset[int]
    segments: list[Segment]
    language: str | None
    language_probability: float | None

    @property
    def is_empty(self) -> bool:
        return not self.completed_windows

    @property
    def resumed_seconds(self) -> float:
        """How far into the recording the previous attempt reached."""
        return max((segment.end for segment in self.segments), default=0.0)


EMPTY_RESUME = ResumePoint(frozenset(), [], None, None)


def plan_fingerprint(
    *,
    settings: Settings,
    model: str,
    language: str | None,
    word_timestamps: bool,
    diarise: bool,
) -> str:
    """Identity of the *windowing and decoding plan*, not of the request.

    Deliberately distinct from ``jobstore.fingerprint``, which identifies a request for
    de-duplication. This answers a narrower question: would a fresh attempt cut the audio in
    the same places and decode it the same way? Anything that moves a window boundary or
    changes a segment belongs here.
    """
    canonical = json.dumps(
        {
            "window_seconds": settings.window_seconds,
            "window_search_seconds": settings.window_search_seconds,
            "vad_filter": settings.vad_filter,
            "beam_size": settings.beam_size,
            "condition_on_previous_text": settings.condition_on_previous_text,
            "model": model,
            "language": language,
            "word_timestamps": word_timestamps,
            "diarise": diarise,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class Checkpointer(Protocol):
    """Somewhere to record and recover per-window progress.

    A protocol so the pipeline never learns that Postgres exists, and so the test suite can
    use an in-memory one. The pipeline's only concerns are "what has already been done" and
    "here is another window's worth of results".
    """

    def load(self) -> ResumePoint:
        """What a previous attempt completed. ``EMPTY_RESUME`` for a fresh start."""
        ...

    def save(
        self,
        *,
        window_index: int,
        offset_seconds: float,
        segments: list[Segment],
        language: str | None,
        language_probability: float | None,
    ) -> None:
        """Record one finished window. Must be idempotent on ``window_index``."""
        ...

    def clear(self) -> None:
        """Drop every checkpoint. Called once the transcript is complete."""
        ...


class InMemoryCheckpointer:
    """Checkpointer that forgets on process exit. For tests and for the CLI.

    Useless for crash recovery, which is the point: it lets the resume path be exercised
    without a database, and it means the CLI can share one code path with the worker instead
    of having a second one that does not checkpoint.
    """

    def __init__(self) -> None:
        self._windows: dict[int, tuple[float, list[Segment]]] = {}
        self._language: str | None = None
        self._language_probability: float | None = None

    def load(self) -> ResumePoint:
        if not self._windows:
            return EMPTY_RESUME
        ordered = sorted(self._windows.items())
        return ResumePoint(
            completed_windows=frozenset(self._windows),
            segments=[segment for _, (_, segments) in ordered for segment in segments],
            language=self._language,
            language_probability=self._language_probability,
        )

    def save(
        self,
        *,
        window_index: int,
        offset_seconds: float,
        segments: list[Segment],
        language: str | None,
        language_probability: float | None,
    ) -> None:
        self._windows[window_index] = (offset_seconds, list(segments))
        if self._language is None:
            self._language = language
            self._language_probability = language_probability

    def clear(self) -> None:
        self._windows.clear()


class JobCheckpointer:
    """Checkpointer backed by the ``job_checkpoints`` table.

    Constructed per attempt, because the plan fingerprint is what decides whether the previous
    attempt's work is still usable, and that is known only once the job's options are.
    """

    def __init__(
        self, settings: Settings, *, job_id: uuid.UUID, plan: str
    ) -> None:
        self._settings = settings
        self._job_id = job_id
        self._plan = plan

    def load(self) -> ResumePoint:
        from transcription import db

        with db.transaction(self._settings) as conn:
            rows = conn.execute(
                """
                SELECT window_index, offset_seconds, segments, language,
                       language_probability, plan_fingerprint
                FROM job_checkpoints
                WHERE job_id = %s
                ORDER BY window_index
                """,
                (self._job_id,),
            ).fetchall()

        if not rows:
            return EMPTY_RESUME

        stale = [row for row in rows if row[5] != self._plan]
        if stale:
            # The plan changed, so window N is no longer the same audio. Stitching these in
            # would produce a transcript that is wrong in a way nothing downstream could
            # detect, which is far worse than paying for a full re-run.
            logger.warning(
                "discarding %d checkpoints for job %s: the windowing plan changed",
                len(rows),
                self._job_id,
            )
            self.clear()
            return EMPTY_RESUME

        segments: list[Segment] = []
        for row in rows:
            segments.extend(Segment.model_validate(item) for item in row[2])

        logger.info(
            "resuming job %s from %d completed window(s), %d segment(s)",
            self._job_id,
            len(rows),
            len(segments),
        )
        return ResumePoint(
            completed_windows=frozenset(row[0] for row in rows),
            segments=segments,
            language=rows[0][3],
            language_probability=float(rows[0][4]) if rows[0][4] is not None else None,
        )

    def save(
        self,
        *,
        window_index: int,
        offset_seconds: float,
        segments: list[Segment],
        language: str | None,
        language_probability: float | None,
    ) -> None:
        from transcription import db

        payload = json.dumps([segment.model_dump(mode="json") for segment in segments])
        with db.transaction(self._settings) as conn:
            # ON CONFLICT DO UPDATE because a redelivered job may well redo a window whose
            # checkpoint it could not read — writing the same answer twice must be harmless.
            conn.execute(
                """
                INSERT INTO job_checkpoints (
                    job_id, window_index, offset_seconds, segments, language,
                    language_probability, plan_fingerprint
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id, window_index) DO UPDATE SET
                    offset_seconds       = EXCLUDED.offset_seconds,
                    segments             = EXCLUDED.segments,
                    language             = EXCLUDED.language,
                    language_probability = EXCLUDED.language_probability,
                    plan_fingerprint     = EXCLUDED.plan_fingerprint
                """,
                (
                    self._job_id,
                    window_index,
                    offset_seconds,
                    payload,
                    language,
                    language_probability,
                    self._plan,
                ),
            )

    def clear(self) -> None:
        from transcription import db

        with db.transaction(self._settings) as conn:
            conn.execute("DELETE FROM job_checkpoints WHERE job_id = %s", (self._job_id,))
