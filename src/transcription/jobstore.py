"""Job state and the queue.

This module is the whole reason the production design exists: **uploading is fast and
I/O-bound, transcribing is slow and CPU-bound, and treating them as one operation is the root
cause of almost every failure mode.** Everything here is in service of separating them.

The queue is a table. ``claim`` is one statement:

* ``SELECT ... FOR UPDATE SKIP LOCKED`` picks a row no other worker has locked, skipping
  contended rows instead of waiting behind them;
* the enclosing ``UPDATE`` marks it running, increments the attempt counter and takes a lease.

Both happen in one transaction, so there is no window in which a job is claimed but not
recorded as claimed. That is the specific bug a separate broker would reintroduce, and it is
why docs/DESIGN.md's original SQS/Redis suggestion is not what got built.

The three mechanisms that make at-least-once delivery survivable, and which the retry design
calls for:

* **Leases.** A claim is valid until ``locked_until``. A worker extends it by heartbeating. A
  worker that is OOM-killed stops extending, and :func:`reclaim_expired` returns the job to the
  queue. No job sits in ``running`` forever.
* **Backoff as scheduling, not as a second queue.** A failed-but-retryable job stays a
  ``queued`` row with ``available_at`` in the future. There is one queue, and nothing to keep
  in sync.
* **Idempotency by request fingerprint.** Identical bytes plus identical options collapse to
  one job, so a client retrying a submission does not pay twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from psycopg.rows import dict_row

from transcription import db
from transcription.config import Settings

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed", "dead_letter"]
LatencyClass = Literal["short", "long"]

TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "dead_letter"})

# Planning figures, not measurements, used only to route a job to a latency class. The
# measured RTF on this repository is 0.20-0.27 on CPU; 0.30 leaves headroom. Diarisation is
# estimated at 1.7-2x, so 1.8. Being wrong here costs a job the wrong queue, not a wrong
# result — which is why an estimate is acceptable where it would not be in the output.
_EXPECTED_RTF = 0.30
_DIARISED_RTF_MULTIPLIER = 1.8

_COLUMNS = """
    id, tenant_id, audio_uri, audio_sha256, display_name, duration_seconds, size_bytes,
    model, language, word_timestamps, diarise, diarisation_backend,
    request_fingerprint, idempotency_key,
    status, attempts, max_attempts, progress, error_code, error_message,
    available_at, locked_until, locked_by, latency_class,
    transcript_uri, created_at, updated_at, started_at, finished_at
"""


@dataclass(frozen=True)
class Job:
    """A unit of transcription work.

    Internal: it carries lease and fingerprint fields that no client should see. The API
    projects this onto a narrower resource.
    """

    id: uuid.UUID
    tenant_id: uuid.UUID
    audio_uri: str
    audio_sha256: str
    display_name: str | None
    duration_seconds: float | None
    size_bytes: int | None
    model: str
    language: str | None
    word_timestamps: bool
    diarise: bool
    diarisation_backend: str
    request_fingerprint: str
    idempotency_key: str | None
    status: JobStatus
    attempts: int
    max_attempts: int
    progress: float
    error_code: str | None
    error_message: str | None
    available_at: datetime
    locked_until: datetime | None
    locked_by: str | None
    latency_class: LatencyClass
    transcript_uri: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Job:
        return cls(
            **{
                **row,
                # numeric comes back as Decimal, which is right for money and wrong for a
                # progress fraction a JSON client has to read.
                "duration_seconds": _as_float(row["duration_seconds"]),
                "progress": _as_float(row["progress"]) or 0.0,
            }
        )


def fingerprint(
    *,
    audio_sha256: str,
    model: str,
    language: str | None,
    word_timestamps: bool,
    diarise: bool,
    diarisation_backend: str,
) -> str:
    """Identity of a transcription request: the audio plus every option that changes output.

    The design keyed de-duplication on ``(tenant, audio, model)``. That is a bug, and a
    quiet one: a ``diarise: true`` request would match an earlier plain run and the caller
    would get a ``200`` carrying a transcript with no speaker labels and no error. Anything
    that changes the result has to be part of identity.

    Hashing the options rather than widening a composite index means adding an option later is
    a code change, not a migration — and it keeps the unique index to two columns.
    """
    canonical = json.dumps(
        {
            "audio_sha256": audio_sha256,
            "model": model,
            "language": language,
            "word_timestamps": bool(word_timestamps),
            "diarise": bool(diarise),
            # Only meaningful when diarising. Including it unconditionally would make two
            # otherwise identical non-diarised requests look different purely because the
            # server's default backend changed.
            "diarisation_backend": diarisation_backend if diarise else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def latency_class_for(
    duration_seconds: float | None, *, diarise: bool, settings: Settings
) -> LatencyClass:
    """Route by estimated *compute*, not by audio duration.

    Two queues split by latency class stop a batch of three-hour recordings starving
    interactive clips. Splitting on duration alone gets this wrong once diarisation exists: a
    diarised ten-minute clip costs roughly what a plain twenty-minute one does, so it belongs
    with the long work even though its duration says otherwise.
    """
    if duration_seconds is None:
        return "short"
    rtf = _EXPECTED_RTF * (_DIARISED_RTF_MULTIPLIER if diarise else 1.0)
    return "long" if duration_seconds * rtf > settings.long_job_threshold_seconds else "short"


def submit(
    settings: Settings,
    *,
    tenant_id: uuid.UUID,
    audio_uri: str,
    audio_sha256: str,
    model: str,
    display_name: str | None = None,
    duration_seconds: float | None = None,
    size_bytes: int | None = None,
    language: str | None = None,
    word_timestamps: bool = False,
    diarise: bool = False,
    diarisation_backend: str = "none",
    idempotency_key: str | None = None,
    max_attempts: int | None = None,
) -> tuple[Job, bool]:
    """Enqueue a job, or return the existing one. Returns ``(job, created)``.

    Idempotent two ways over: an explicit ``Idempotency-Key``, and the content-and-options
    fingerprint. Both are enforced by unique indexes rather than by a read-then-write, so two
    concurrent identical submissions cannot both win.
    """
    request_fingerprint = fingerprint(
        audio_sha256=audio_sha256,
        model=model,
        language=language,
        word_timestamps=word_timestamps,
        diarise=diarise,
        diarisation_backend=diarisation_backend,
    )
    job_id = uuid.uuid4()

    with db.transaction(settings) as conn:
        cursor = conn.cursor(row_factory=dict_row)

        # ON CONFLICT DO NOTHING rather than DO UPDATE: an existing job must not be mutated by
        # a duplicate submission. A job that is already running should keep running, and one
        # that already failed should keep its error rather than silently restarting.
        cursor.execute(
            f"""
            INSERT INTO jobs (
                id, tenant_id, audio_uri, audio_sha256, display_name, duration_seconds,
                size_bytes, model, language, word_timestamps, diarise, diarisation_backend,
                request_fingerprint, idempotency_key, latency_class, max_attempts
            ) VALUES (
                %(id)s, %(tenant_id)s, %(audio_uri)s, %(audio_sha256)s, %(display_name)s,
                %(duration_seconds)s, %(size_bytes)s, %(model)s, %(language)s,
                %(word_timestamps)s, %(diarise)s, %(diarisation_backend)s,
                %(request_fingerprint)s, %(idempotency_key)s, %(latency_class)s,
                %(max_attempts)s
            )
            ON CONFLICT DO NOTHING
            RETURNING {_COLUMNS}
            """,
            {
                "id": job_id,
                "tenant_id": tenant_id,
                "audio_uri": audio_uri,
                "audio_sha256": audio_sha256,
                "display_name": display_name,
                "duration_seconds": duration_seconds,
                "size_bytes": size_bytes,
                "model": model,
                "language": language,
                "word_timestamps": word_timestamps,
                "diarise": diarise,
                "diarisation_backend": diarisation_backend,
                "request_fingerprint": request_fingerprint,
                "idempotency_key": idempotency_key,
                "latency_class": latency_class_for(
                    duration_seconds, diarise=diarise, settings=settings
                ),
                "max_attempts": max_attempts or settings.job_max_attempts,
            },
        )
        row = cursor.fetchone()
        if row is not None:
            return Job.from_row(row), True

        # Lost the race, or a genuine duplicate. Either way the caller wants the job that
        # does exist. Match on whichever key collided.
        if idempotency_key is not None:
            cursor.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE tenant_id = %s AND idempotency_key = %s",
                (tenant_id, idempotency_key),
            )
            existing = cursor.fetchone()
            if existing is not None:
                return Job.from_row(existing), False

        cursor.execute(
            f"SELECT {_COLUMNS} FROM jobs WHERE tenant_id = %s AND request_fingerprint = %s",
            (tenant_id, request_fingerprint),
        )
        existing = cursor.fetchone()
        if existing is None:  # pragma: no cover - would mean the conflict had no cause
            raise db.DatabaseError("Insert conflicted but no existing job was found")
        return Job.from_row(existing), False


def claim(
    settings: Settings,
    *,
    worker_id: str,
    latency_class: LatencyClass | None = None,
    lease_seconds: float | None = None,
) -> Job | None:
    """Lease the next due job, or return ``None`` if there is nothing to do.

    The one statement that matters in this module.

    ``SKIP LOCKED`` is what makes this safe to run from many workers at once: a worker whose
    chosen row is already locked moves to the next candidate instead of blocking. Without it,
    N workers serialise behind the same head-of-queue row.

    ``ORDER BY available_at, created_at`` keeps it approximately fair, and means a job
    deferred by backoff waits its turn rather than jumping the queue when it becomes due.
    """
    lease = lease_seconds or settings.job_lease_seconds

    with db.transaction(settings) as conn:
        cursor = conn.cursor(row_factory=dict_row)
        cursor.execute(
            f"""
            UPDATE jobs SET
                status       = 'running',
                attempts     = attempts + 1,
                locked_by    = %(worker_id)s,
                locked_until = now() + make_interval(secs => %(lease)s),
                started_at   = COALESCE(started_at, now()),
                progress     = 0,
                error_code   = NULL,
                error_message = NULL
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'queued'
                  AND available_at <= now()
                  AND (%(latency_class)s::text IS NULL OR latency_class = %(latency_class)s)
                ORDER BY available_at, created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING {_COLUMNS}
            """,
            {"worker_id": worker_id, "lease": float(lease), "latency_class": latency_class},
        )
        row = cursor.fetchone()
        return Job.from_row(row) if row else None


def heartbeat(
    settings: Settings,
    *,
    job_id: uuid.UUID,
    worker_id: str,
    progress: float | None = None,
    lease_seconds: float | None = None,
) -> bool:
    """Extend the lease and record progress. ``False`` means the lease is gone.

    A three-hour job has to say it is still alive, or the lease expires and the job is
    redelivered while the first worker is still chewing on it — and then it runs twice,
    forever.

    **A worker that gets ``False`` must stop.** It no longer owns the job; something else may
    already have claimed it. Continuing would mean two workers writing one result. That is why
    this returns a value rather than raising: losing a lease is a normal event in an
    at-least-once system, not an error.

    The ``locked_by`` check is what makes that guarantee real. Without it, any worker could
    refresh any lease.
    """
    lease = lease_seconds or settings.job_lease_seconds

    with db.transaction(settings) as conn:
        cursor = conn.execute(
            """
            UPDATE jobs SET
                locked_until = now() + make_interval(secs => %(lease)s),
                progress     = COALESCE(%(progress)s, progress)
            WHERE id = %(job_id)s
              AND status = 'running'
              AND locked_by = %(worker_id)s
            """,
            {
                "lease": float(lease),
                "progress": _clamp_progress(progress),
                "job_id": job_id,
                "worker_id": worker_id,
            },
        )
        return cursor.rowcount > 0


def succeed(
    settings: Settings,
    *,
    job_id: uuid.UUID,
    worker_id: str,
    result: dict[str, Any],
    transcript_uri: str | None = None,
) -> bool:
    """Record a finished transcript. ``False`` means the lease was lost and nothing was written.

    The result is written with ``ON CONFLICT DO UPDATE``, which is the idempotency
    at-least-once delivery requires: a redelivered job overwrites its own result rather than
    producing a second one. Because the output is derived from the content hash, the overwrite
    is with equivalent data anyway.
    """
    with db.transaction(settings) as conn:
        cursor = conn.execute(
            """
            UPDATE jobs SET
                status        = 'succeeded',
                progress      = 1,
                locked_until  = NULL,
                locked_by     = NULL,
                finished_at   = now(),
                transcript_uri = %(transcript_uri)s,
                error_code    = NULL,
                error_message = NULL
            WHERE id = %(job_id)s
              AND status = 'running'
              AND locked_by = %(worker_id)s
            """,
            {"job_id": job_id, "worker_id": worker_id, "transcript_uri": transcript_uri},
        )
        if cursor.rowcount == 0:
            # Lease lost mid-flight. Refusing to write is the point: another worker may have
            # already finished this job, and the last writer winning is not a property worth
            # having.
            logger.warning("refusing result for job %s: lease no longer held", job_id)
            return False

        conn.execute(
            """
            INSERT INTO job_results (job_id, result) VALUES (%s, %s)
            ON CONFLICT (job_id) DO UPDATE SET result = EXCLUDED.result
            """,
            (job_id, json.dumps(result)),
        )
        return True


def fail(
    settings: Settings,
    *,
    job_id: uuid.UUID,
    worker_id: str,
    error_code: str,
    error_message: str | None = None,
    retryable: bool,
) -> Job | None:
    """Record a failure and decide what happens next.

    The whole retry policy rests on one distinction, and it is the caller's to supply:
    **permanent versus transient.** Retrying a corrupt file five times burns money five times
    and delays the error the user needs to see. ``errors.py`` already carries that flag per
    error type, defaulting to non-retryable, so a new error type is permanent until someone
    decides otherwise.

    Three outcomes:

    * permanent — ``failed``, immediately, with the code surfaced;
    * transient with attempts remaining — back to ``queued``, with ``available_at`` pushed
      into the future by a jittered exponential backoff;
    * transient with attempts exhausted — ``dead_letter``, which is an alert rather than a
      dashboard: a human has to choose between replaying it and declaring the input broken.
    """
    with db.transaction(settings) as conn:
        cursor = conn.cursor(row_factory=dict_row)
        # Lock the row so the attempt count cannot move under us between deciding and writing.
        cursor.execute(
            f"SELECT {_COLUMNS} FROM jobs WHERE id = %s FOR UPDATE", (job_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None

        job = Job.from_row(row)
        if job.status != "running" or job.locked_by != worker_id:
            logger.warning("refusing failure for job %s: lease no longer held", job_id)
            return job

        if not retryable:
            status, available_at_sql, backoff = "failed", "available_at", None
        elif job.attempts >= job.max_attempts:
            status, available_at_sql, backoff = "dead_letter", "available_at", None
        else:
            status = "queued"
            available_at_sql = "now() + make_interval(secs => %(backoff)s)"
            backoff = _backoff_seconds(job.attempts, settings)

        cursor.execute(
            f"""
            UPDATE jobs SET
                status        = %(status)s,
                available_at  = {available_at_sql},
                locked_until  = NULL,
                locked_by     = NULL,
                error_code    = %(error_code)s,
                error_message = %(error_message)s,
                finished_at   = CASE WHEN %(status)s IN ('failed', 'dead_letter')
                                     THEN now() ELSE NULL END
            WHERE id = %(job_id)s
            RETURNING {_COLUMNS}
            """,
            {
                "status": status,
                "error_code": error_code,
                "error_message": error_message,
                "job_id": job_id,
                "backoff": backoff,
            },
        )
        updated = cursor.fetchone()
        if status == "dead_letter":
            # Worth a distinct log line: DLQ depth above zero is one of the two things the
            # design says to page on.
            logger.error(
                "job %s dead-lettered after %s attempts: %s", job_id, job.attempts, error_code
            )
        return Job.from_row(updated) if updated else None


def reclaim_expired(settings: Settings) -> list[tuple[uuid.UUID, str]]:
    """Return abandoned jobs to the queue. Returns ``(id, new_status)`` for each.

    This is the half of at-least-once delivery that a broker would normally provide. A worker
    that is OOM-killed, evicted or reclaimed as a spot instance stops heartbeating; its lease
    lapses; the job becomes claimable again. Nothing is lost and nothing hangs in ``running``.

    A job that has already burned its attempts is dead-lettered rather than requeued, so a
    crash loop cannot cycle the same job forever.

    Safe to call from every worker concurrently — it is a single statement, and a row another
    worker has already reclaimed simply will not match.
    """
    with db.transaction(settings) as conn:
        cursor = conn.execute(
            """
            UPDATE jobs SET
                status        = CASE WHEN attempts >= max_attempts
                                     THEN 'dead_letter' ELSE 'queued' END,
                locked_until  = NULL,
                locked_by     = NULL,
                error_code    = 'lease_expired',
                error_message = 'Worker stopped heartbeating; lease expired.',
                finished_at   = CASE WHEN attempts >= max_attempts THEN now() ELSE NULL END
            WHERE status = 'running' AND locked_until < now()
            RETURNING id, status
            """
        )
        reclaimed = [(row[0], row[1]) for row in cursor.fetchall()]

    for job_id, status in reclaimed:
        logger.warning("reclaimed job %s with expired lease -> %s", job_id, status)
    return reclaimed


def get(
    settings: Settings, *, job_id: uuid.UUID, tenant_id: uuid.UUID | None = None
) -> Job | None:
    """Fetch one job, optionally scoped to a tenant.

    ``tenant_id`` is not optional in practice — the API always passes it. One tenant reading
    another's job by guessing a UUID is the kind of hole that is trivial to leave open, so the
    scoping belongs in the query rather than in a check the caller might forget.
    """
    with db.transaction(settings) as conn:
        cursor = conn.cursor(row_factory=dict_row)
        if tenant_id is None:
            cursor.execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = %s", (job_id,))
        else:
            cursor.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE id = %s AND tenant_id = %s",
                (job_id, tenant_id),
            )
        row = cursor.fetchone()
        return Job.from_row(row) if row else None


def get_result(
    settings: Settings, *, job_id: uuid.UUID, tenant_id: uuid.UUID | None = None
) -> dict[str, Any] | None:
    """The stored transcript document, or ``None`` if the job has not produced one."""
    with db.transaction(settings) as conn:
        if tenant_id is None:
            cursor = conn.execute(
                "SELECT result FROM job_results WHERE job_id = %s", (job_id,)
            )
        else:
            cursor = conn.execute(
                """
                SELECT r.result FROM job_results r
                JOIN jobs j ON j.id = r.job_id
                WHERE r.job_id = %s AND j.tenant_id = %s
                """,
                (job_id, tenant_id),
            )
        row = cursor.fetchone()
        return row[0] if row else None


def list_jobs(
    settings: Settings,
    *,
    tenant_id: uuid.UUID,
    status: JobStatus | None = None,
    cursor: str | None = None,
    limit: int = 50,
) -> tuple[list[Job], str | None]:
    """Page through a tenant's jobs, newest first. Returns ``(jobs, next_cursor)``.

    Keyset pagination on ``(created_at, id)`` rather than ``OFFSET``. Offset pagination costs
    more the deeper you go and, worse, skips or repeats rows when the underlying set changes
    between pages — which it constantly does here, since jobs are being created and finishing
    while someone pages through them.
    """
    limit = max(1, min(limit, 200))
    params: dict[str, Any] = {"tenant_id": tenant_id, "limit": limit + 1}

    conditions = ["tenant_id = %(tenant_id)s"]
    if status is not None:
        conditions.append("status = %(status)s")
        params["status"] = status
    if cursor:
        created_at, last_id = _decode_cursor(cursor)
        conditions.append("(created_at, id) < (%(cursor_created_at)s, %(cursor_id)s)")
        params["cursor_created_at"] = created_at
        params["cursor_id"] = last_id

    where = " AND ".join(conditions)
    with db.transaction(settings) as conn:
        rows = (
            conn.cursor(row_factory=dict_row)
            .execute(
                f"""
                SELECT {_COLUMNS} FROM jobs
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT %(limit)s
                """,
                params,
            )
            .fetchall()
        )

    jobs = [Job.from_row(row) for row in rows[:limit]]
    # Fetching one extra row is how we know whether another page exists without a second
    # COUNT query over the same predicate.
    next_cursor = (
        _encode_cursor(jobs[-1].created_at, jobs[-1].id) if len(rows) > limit and jobs else None
    )
    return jobs, next_cursor


def delete_job(settings: Settings, *, job_id: uuid.UUID, tenant_id: uuid.UUID) -> Job | None:
    """Delete a job and its result. Returns the job as it was, so the caller can clean up.

    The caller gets the row back because erasure is not finished when the row goes: the audio
    is in a blob store and has to be deleted too. A voice recording is biometric personal
    data, so this path is an obligation rather than a convenience.
    """
    with db.transaction(settings) as conn:
        cursor = conn.cursor(row_factory=dict_row)
        cursor.execute(
            f"DELETE FROM jobs WHERE id = %s AND tenant_id = %s RETURNING {_COLUMNS}",
            (job_id, tenant_id),
        )
        row = cursor.fetchone()
        return Job.from_row(row) if row else None


def queue_stats(settings: Settings) -> dict[str, Any]:
    """Depth by status, and the age of the oldest waiting job.

    Oldest-queued age is the number worth autoscaling on, and the design says so. Mean depth
    hides the thing that matters: audio arrives in bursts, so a queue that is empty on average
    can still have a job that has been waiting twenty minutes.
    """
    with db.transaction(settings) as conn:
        by_status = dict(
            conn.execute("SELECT status, count(*) FROM jobs GROUP BY status").fetchall()
        )
        row = conn.execute(
            """
            SELECT
                count(*),
                COALESCE(max(EXTRACT(EPOCH FROM (now() - available_at))), 0)
            FROM jobs
            WHERE status = 'queued' AND available_at <= now()
            """
        ).fetchone()

    return {
        "by_status": {status: int(count) for status, count in by_status.items()},
        "claimable": int(row[0]) if row else 0,
        "oldest_claimable_age_seconds": round(float(row[1]), 3) if row else 0.0,
    }


def _backoff_seconds(attempts: int, settings: Settings) -> float:
    """Exponential backoff with equal jitter, capped.

    Jitter matters more than the curve. Without it, a batch of jobs that fail together — which
    is what a transient dependency outage looks like — retries in lockstep forever, hammering
    the thing that is already struggling. Equal jitter keeps half the delay fixed, so there is
    always a floor, and randomises the other half to spread the herd.
    """
    exponential = settings.job_backoff_base_seconds * (2 ** max(0, attempts - 1))
    capped = min(exponential, settings.job_backoff_max_seconds)
    return round(capped / 2 + random.uniform(0, capped / 2), 3)


def _clamp_progress(progress: float | None) -> float | None:
    if progress is None:
        return None
    return max(0.0, min(1.0, float(progress)))


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _encode_cursor(created_at: datetime, job_id: uuid.UUID) -> str:
    import base64

    raw = f"{created_at.astimezone(UTC).isoformat()}|{job_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    import base64

    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        timestamp, _, job_id = raw.partition("|")
        return datetime.fromisoformat(timestamp), uuid.UUID(job_id)
    except Exception as exc:
        # A bad cursor is a client error, not a server one, and it must not leak the encoding.
        raise ValueError("Malformed pagination cursor") from exc
