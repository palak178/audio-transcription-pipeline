"""The worker tier: claim a job, transcribe it, report what happened.

This is the process that makes the whole asynchronous design worth building. The API accepts
bytes and returns in milliseconds; this is where the minutes of CPU actually go, on a pool
sized to compute rather than to request volume.

It is deliberately a thin shell around :func:`transcription.pipeline.transcribe_file`. The
pipeline already handles format probing, windowing, long audio and diarisation; a worker that
reimplemented any of that would be a second code path to keep correct. What this module adds
is only the things the pipeline has no business knowing about: leases, heartbeats, failure
classification and graceful shutdown.

Three behaviours worth reading the code for:

* **Heartbeats ride on the progress callback.** The pipeline already reports progress per
  segment, which is exactly the hook a lease extension needs. No timer thread, no separate
  bookkeeping — the thing that proves the job is advancing is the thing that proves the worker
  is alive.
* **A lost lease aborts the job.** If a heartbeat comes back ``False`` this worker no longer
  owns the job and something else may have claimed it. Continuing would mean two workers
  racing to write one result, so the work is abandoned mid-flight instead.
* **Unexpected exceptions are treated as permanent.** That follows the policy in
  ``errors.py``: a failure nobody has classified is non-retryable until someone decides
  otherwise. Retrying an unclassified bug five times buys nothing and costs four extra runs.
  Genuine infrastructure death — OOM kill, eviction, spot reclaim — never reaches this handler
  at all, because the process is gone; that case is covered by lease expiry.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
import uuid
from pathlib import Path

from transcription import blobstore, db, jobstore, pipeline
from transcription.checkpoint import JobCheckpointer, plan_fingerprint
from transcription.config import Settings, get_settings
from transcription.errors import TranscriptionError
from transcription.jobstore import Job, LatencyClass
from transcription.transcriber import FasterWhisperTranscriber, Transcriber

logger = logging.getLogger(__name__)


class _LeaseLost(Exception):
    """Raised inside the progress callback to unwind a job this worker no longer owns."""


def default_worker_id() -> str:
    """Host, pid and a short random suffix.

    Host and pid so an operator reading ``locked_by`` can go find the process. The random
    suffix because pids are recycled, and two workers in one container would otherwise be
    indistinguishable.
    """
    return f"{socket.gethostname()}/{os.getpid()}/{uuid.uuid4().hex[:6]}"


class Worker:
    """Claims and runs transcription jobs until told to stop."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        worker_id: str | None = None,
        latency_class: LatencyClass | None = None,
        transcriber: Transcriber | None = None,
        blob_store: blobstore.BlobStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.worker_id = worker_id or default_worker_id()
        self.latency_class = latency_class
        # Injected in tests; built per job otherwise, because the model is a per-job option.
        self._transcriber = transcriber
        self._blob_store = blob_store or blobstore.load_blob_store(self.settings)
        self._stop = threading.Event()
        self.jobs_completed = 0
        self.jobs_failed = 0

    # --- lifecycle ------------------------------------------------------------------------

    def stop(self) -> None:
        """Ask the worker to exit after the job in flight finishes."""
        self._stop.set()

    def install_signal_handlers(self) -> None:
        """Make SIGTERM and SIGINT mean "finish this job, then exit".

        Kubernetes sends SIGTERM and then waits. Exiting immediately would abandon a job
        mid-transcription; it would be redelivered after its lease expired, but that is
        minutes of CPU thrown away on every routine deploy. Draining instead costs one job's
        remaining runtime and wastes nothing.
        """

        def handle(signum, _frame):
            logger.info("signal %s received; draining", signal.Signals(signum).name)
            self.stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, handle)

    def run_forever(self) -> None:
        """Claim and run jobs until stopped."""
        logger.info(
            "worker %s starting (latency_class=%s, blob_store=%s)",
            self.worker_id,
            self.latency_class or "any",
            self._blob_store.name,
        )
        while not self._stop.is_set():
            try:
                did_work = self.run_once()
            except db.DatabaseError as exc:
                # The database being briefly unreachable must not kill the worker. Back off
                # and try again; the alternative is a crash loop during a failover.
                logger.warning("database unavailable: %s", exc)
                self._stop.wait(self.settings.worker_poll_seconds * 5)
                continue

            if not did_work:
                # Nothing claimable. Waiting on the stop event rather than sleeping means
                # shutdown is immediate when the queue is empty.
                self._stop.wait(self.settings.worker_poll_seconds)

        logger.info(
            "worker %s stopped (completed=%d failed=%d)",
            self.worker_id,
            self.jobs_completed,
            self.jobs_failed,
        )

    def run_once(self) -> bool:
        """Reclaim abandoned work, then claim and run at most one job.

        Returns whether a job was run, so the caller knows whether to poll or to come
        straight back for more.
        """
        # Every worker does this, not a dedicated reaper. It is one statement against a
        # partial index, a row already reclaimed simply will not match, and it means there is
        # no separate process whose death silently strands every job in the system.
        jobstore.reclaim_expired(self.settings)

        job = jobstore.claim(
            self.settings, worker_id=self.worker_id, latency_class=self.latency_class
        )
        if job is None:
            return False

        logger.info(
            "claimed job %s (attempt %d/%d, %s, %.1fs audio)",
            job.id,
            job.attempts,
            job.max_attempts,
            job.model,
            job.duration_seconds or 0.0,
        )
        self._process(job)
        return True

    # --- one job --------------------------------------------------------------------------

    def _process(self, job: Job) -> None:
        started = time.perf_counter()
        try:
            with self._blob_store.materialise(job.audio_uri) as audio_path:
                result = self._transcribe(job, audio_path)
        except _LeaseLost:
            # Deliberately no state write. We do not own this job any more, so touching it
            # would risk clobbering whoever does.
            logger.warning("abandoned job %s: lease lost mid-flight", job.id)
            return
        except TranscriptionError as exc:
            self._fail(job, code=exc.code, message=exc.message, retryable=exc.retryable)
            return
        except Exception as exc:
            # Unclassified, therefore permanent — see the module docstring.
            logger.exception("unhandled error on job %s", job.id)
            self._fail(
                job,
                code="worker_error",
                message=f"{type(exc).__name__}: {exc}",
                retryable=False,
            )
            return

        stored = jobstore.succeed(
            self.settings,
            job_id=job.id,
            worker_id=self.worker_id,
            result=result.model_dump(mode="json"),
        )
        if not stored:
            logger.warning("job %s finished but the lease was gone; result discarded", job.id)
            return

        # Only once the result is durably stored. Clearing earlier would turn a crash in
        # between into a full re-run, which is exactly what checkpointing exists to avoid.
        JobCheckpointer(self.settings, job_id=job.id, plan="").clear()

        self.jobs_completed += 1
        logger.info(
            "job %s succeeded in %.1fs (%d segments, rtf %s)",
            job.id,
            time.perf_counter() - started,
            len(result.segments),
            result.real_time_factor,
        )

    def _transcribe(self, job: Job, audio_path: Path):
        transcriber = self._transcriber or FasterWhisperTranscriber(
            self.settings, model_name=job.model
        )

        # Keyed by the windowing-and-decoding plan rather than by the job, so a checkpoint
        # written under different settings is discarded instead of stitched into a transcript
        # that would be wrong in a way nothing downstream could detect.
        checkpointer = JobCheckpointer(
            self.settings,
            job_id=job.id,
            plan=plan_fingerprint(
                settings=self.settings,
                model=job.model,
                language=job.language,
                word_timestamps=job.word_timestamps,
                diarise=job.diarise,
            ),
        )

        last_beat = 0.0

        def progress(done_seconds: float, total_seconds: float) -> None:
            nonlocal last_beat
            now = time.monotonic()
            # Throttled: the pipeline fires this per segment, which on a long file is
            # hundreds of times a minute. One UPDATE per segment would be pointless load.
            if now - last_beat < self.settings.job_heartbeat_seconds:
                return
            last_beat = now
            fraction = (done_seconds / total_seconds) if total_seconds else 0.0
            if not jobstore.heartbeat(
                self.settings,
                job_id=job.id,
                worker_id=self.worker_id,
                progress=fraction,
            ):
                raise _LeaseLost(str(job.id))

        return pipeline.transcribe_file(
            audio_path,
            transcriber=transcriber,
            settings=self.settings,
            language=job.language,
            word_timestamps=job.word_timestamps,
            diarise=job.diarise,
            progress=progress,
            display_name=job.display_name,
            checkpointer=checkpointer,
            # The submission endpoint already enforced the tenant-facing limit. Re-checking
            # against the sync endpoint's stricter cap here would reject exactly the long
            # audio this tier exists to handle.
            max_duration_seconds=self.settings.max_duration_seconds,
        )

    def _fail(self, job: Job, *, code: str, message: str, retryable: bool) -> None:
        self.jobs_failed += 1
        outcome = jobstore.fail(
            self.settings,
            job_id=job.id,
            worker_id=self.worker_id,
            error_code=code,
            error_message=message,
            retryable=retryable,
        )
        logger.warning(
            "job %s failed (%s, retryable=%s) -> %s",
            job.id,
            code,
            retryable,
            outcome.status if outcome else "unknown",
        )


def run(
    settings: Settings | None = None,
    *,
    latency_class: LatencyClass | None = None,
    migrate: bool = True,
) -> None:
    """Entry point for the ``transcribe-worker`` command."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    if migrate:
        applied = db.migrate(settings)
        if applied:
            logger.info("applied migrations: %s", ", ".join(applied))

    worker = Worker(settings, latency_class=latency_class)
    worker.install_signal_handlers()
    try:
        worker.run_forever()
    finally:
        db.close_pools()
