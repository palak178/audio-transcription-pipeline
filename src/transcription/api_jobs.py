"""The asynchronous job API.

Separated from :mod:`transcription.api` for two reasons. It is optional — it needs Postgres,
where the synchronous endpoint needs nothing — and ``api.py`` is already long enough that
adding six endpoints to it would hurt.

**One deliberate deviation from docs/DESIGN.md.** The design put the asynchronous submission
at ``POST /v1/transcriptions`` and moved the synchronous one to ``POST
/v1/transcriptions:sync``. That is a clean URL scheme and it is not what got built, because it
would mean the documented, working, zero-infrastructure endpoint starts returning ``503``
unless a database is configured. A fresh clone must still transcribe with nothing but ffmpeg
and a model.

So jobs are their own collection. That turns out to be more honest REST anyway: a job and a
transcript are genuinely different resources. A job has a status, an attempt count and a
progress fraction; a transcript has segments. ``GET /v1/jobs/{id}`` answers "how is it going",
``GET /v1/jobs/{id}/transcript`` answers "what did it say", and neither has to pretend to be
the other.

Authentication is **not** here. Per-tenant API keys are a later phase; until then the tenant
comes from a header with a documented default. That is stated plainly rather than hidden,
because an API that looks authenticated and is not is worse than one that is openly open.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

import anyio
from fastapi import APIRouter, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from transcription import audio as audio_module
from transcription import blobstore, exporters, jobstore
from transcription.config import Settings
from transcription.errors import TranscriptionError
from transcription.exporters import ResponseFormat
from transcription.models import (
    JobError,
    JobList,
    JobResource,
    QueueStats,
    TranscriptionResult,
)
from transcription.pipeline import safe_filename

logger = logging.getLogger(__name__)

# Until per-tenant API keys exist, every unattributed request belongs to this tenant. A fixed
# UUID rather than a random one per process, so jobs submitted before a restart are still
# visible after it.
ANONYMOUS_TENANT = uuid.UUID("00000000-0000-0000-0000-000000000001")


class JobNotFoundError(TranscriptionError):
    code = "job_not_found"
    http_status = 404


class TranscriptNotReadyError(TranscriptionError):
    """The job exists but has not produced a transcript.

    ``409`` rather than ``404``: the resource is real, the request is just premature. A client
    can tell "keep polling" from "you have the wrong id", which it could not if both were 404.
    """

    code = "transcript_not_ready"
    http_status = 409


def _tenant_from(header: str | None) -> uuid.UUID:
    if not header:
        return ANONYMOUS_TENANT
    try:
        return uuid.UUID(header)
    except ValueError as exc:
        raise TranscriptionError(f"X-Tenant-Id must be a UUID, got {header!r}") from exc


def to_resource(job: jobstore.Job) -> JobResource:
    """Project an internal row onto the client-facing resource."""
    return JobResource(
        id=str(job.id),
        status=job.status,
        progress=round(job.progress, 4),
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        latency_class=job.latency_class,
        audio={
            "filename": job.display_name,
            "duration_seconds": job.duration_seconds,
            "sha256": job.audio_sha256,
            "size_bytes": job.size_bytes,
        },
        options={
            "model": job.model,
            "language": job.language,
            "word_timestamps": job.word_timestamps,
            "diarise": job.diarise,
            "diarisation_backend": job.diarisation_backend if job.diarise else None,
        },
        # Advertised only once there is something behind it, so a client cannot be led into
        # polling a URL that is guaranteed to 409.
        transcript_url=(
            f"/v1/jobs/{job.id}/transcript" if job.status == "succeeded" else None
        ),
        error=(
            JobError(code=job.error_code, message=job.error_message)
            if job.error_code and job.status in {"failed", "dead_letter"}
            else None
        ),
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def create_jobs_router(settings: Settings) -> APIRouter:
    """Build the job router. Mounted only when ``jobs_enabled`` is set."""
    router = APIRouter(prefix="/v1", tags=["jobs"])
    store = blobstore.load_blob_store(settings)

    @router.post(
        "/jobs",
        status_code=201,
        response_model=JobResource,
        summary="Submit audio for asynchronous transcription",
        description=(
            "Accepts the audio, records the intent durably, and returns immediately. This is "
            "the endpoint for long audio: nothing here is bounded by how long transcription "
            "takes.\n\n"
            "Submitting identical bytes with identical options returns the **existing** job "
            "rather than creating a second one, so a client that retries a submission it is "
            "unsure about does not pay twice."
        ),
    )
    async def submit_job(
        request: Request,
        file: Annotated[UploadFile, File(description="Audio in any ffmpeg-readable format.")],
        model: Annotated[str | None, Form(description="Whisper model size.")] = None,
        language: Annotated[str | None, Form(description="Force a language code.")] = None,
        word_timestamps: Annotated[bool, Form(description="Add per-word timings.")] = False,
        diarise: Annotated[bool, Form(description="Label who spoke.")] = False,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
        tenant: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
    ) -> Response:
        current: Settings = request.app.state.settings
        tenant_id = _tenant_from(tenant)

        with tempfile.TemporaryDirectory(prefix="submit-") as tmp_dir:
            staged = Path(tmp_dir) / "upload"
            await _spool(file, staged, current.max_upload_bytes)

            # Probing before enqueuing is the point of the first duration gate in the design:
            # a small compressed file that expands into hours of audio is rejected here,
            # rather than consuming a worker to discover the same thing.
            info = await anyio.to_thread.run_sync(audio_module.inspect, staged)

            job, created = await anyio.to_thread.run_sync(
                lambda: _store_and_submit(
                    current,
                    store=store,
                    staged=staged,
                    info=info,
                    tenant_id=tenant_id,
                    display_name=safe_filename(file.filename),
                    model=model or current.whisper_model,
                    language=language,
                    word_timestamps=word_timestamps,
                    diarise=diarise,
                    idempotency_key=idempotency_key,
                )
            )

        resource = to_resource(job)
        return JSONResponse(
            status_code=201 if created else 200,
            content=resource.model_dump(mode="json"),
            headers={"Location": f"/v1/jobs/{job.id}"},
        )

    @router.get(
        "/jobs/{job_id}",
        response_model=JobResource,
        summary="Job status and progress",
        description=(
            "Poll this. The state machine is exposed deliberately — `queued`, `running` with "
            "a progress fraction, `succeeded`, `failed`, or `dead_letter` when retries are "
            "exhausted."
        ),
    )
    async def get_job(
        request: Request,
        job_id: uuid.UUID,
        tenant: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
    ) -> JobResource:
        current: Settings = request.app.state.settings
        tenant_id = _tenant_from(tenant)
        job = await anyio.to_thread.run_sync(
            lambda: jobstore.get(current, job_id=job_id, tenant_id=tenant_id)
        )
        if job is None:
            raise JobNotFoundError(f"No job {job_id}")
        return to_resource(job)

    @router.get(
        "/jobs/{job_id}/transcript",
        summary="The finished transcript",
        description=(
            "Available once the job has succeeded; `409 transcript_not_ready` before that, so "
            "a client can tell 'keep polling' from 'wrong id'. Every format "
            "`exporters.py` produces is available here."
        ),
    )
    async def get_transcript(
        request: Request,
        job_id: uuid.UUID,
        response_format: Annotated[
            ResponseFormat, Query(alias="format", description="Output format.")
        ] = ResponseFormat.json,
        speaker_labels: Annotated[
            bool,
            Query(
                description=(
                    "Render speaker labels when the transcript has them. Set false for clean "
                    "captions out of a diarised job."
                )
            ),
        ] = True,
        tenant: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
    ) -> Response:
        current: Settings = request.app.state.settings
        tenant_id = _tenant_from(tenant)

        def load() -> tuple[jobstore.Job | None, dict | None]:
            found = jobstore.get(current, job_id=job_id, tenant_id=tenant_id)
            if found is None or found.status != "succeeded":
                return found, None
            return found, jobstore.get_result(current, job_id=job_id, tenant_id=tenant_id)

        job, payload = await anyio.to_thread.run_sync(load)

        if job is None:
            raise JobNotFoundError(f"No job {job_id}")
        if payload is None:
            raise TranscriptNotReadyError(
                f"Job {job_id} is {job.status}, so there is no transcript yet.",
                status=job.status,
                progress=round(job.progress, 4),
            )

        result = TranscriptionResult.model_validate(payload)
        fmt = response_format.value
        if fmt == "json":
            return JSONResponse(content=payload)
        return PlainTextResponse(
            content=exporters.render(result, fmt, speaker_labels=speaker_labels),
            media_type=exporters.MEDIA_TYPES[fmt],
        )

    @router.get(
        "/jobs",
        response_model=JobList,
        summary="List jobs, newest first",
        description="Keyset-paginated, so pages do not skip or repeat while jobs are created.",
    )
    async def list_jobs(
        request: Request,
        status: Annotated[str | None, Query(description="Filter by status.")] = None,
        cursor: Annotated[str | None, Query(description="From a previous page.")] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        tenant: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
    ) -> JobList:
        current: Settings = request.app.state.settings
        tenant_id = _tenant_from(tenant)

        if status is not None and status not in {
            "queued",
            "running",
            "succeeded",
            "failed",
            "dead_letter",
        }:
            raise TranscriptionError(f"Unknown status {status!r}")

        def query():
            return jobstore.list_jobs(
                current, tenant_id=tenant_id, status=status, cursor=cursor, limit=limit
            )

        try:
            jobs, next_cursor = await anyio.to_thread.run_sync(query)
        except ValueError as exc:
            raise TranscriptionError(str(exc)) from exc

        return JobList(data=[to_resource(job) for job in jobs], next_cursor=next_cursor)

    @router.delete(
        "/jobs/{job_id}",
        status_code=204,
        summary="Delete a job, its transcript and its audio",
        description=(
            "Erasure, not archiving. A voice recording is biometric personal data, so being "
            "able to actually remove it is an obligation rather than a convenience: the row, "
            "the stored result and the audio blob all go."
        ),
    )
    async def delete_job(
        request: Request,
        job_id: uuid.UUID,
        tenant: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
    ) -> Response:
        current: Settings = request.app.state.settings
        tenant_id = _tenant_from(tenant)

        def erase() -> jobstore.Job | None:
            removed = jobstore.delete_job(current, job_id=job_id, tenant_id=tenant_id)
            if removed is None:
                return None
            # Only delete the audio once no job still references those bytes. Content
            # addressing means two jobs can legitimately share one blob, and deleting it for
            # one would quietly break the other.
            if not _blob_still_referenced(current, removed.audio_sha256):
                try:
                    store.delete(removed.audio_uri)
                except blobstore.BlobStoreError:
                    # The row is gone, which is the part that must not be lost. An orphaned
                    # blob is a cleanup problem, not a correctness one.
                    logger.warning("could not delete blob %s", removed.audio_uri, exc_info=True)
            return removed

        removed = await anyio.to_thread.run_sync(erase)
        if removed is None:
            raise JobNotFoundError(f"No job {job_id}")
        return Response(status_code=204)

    @router.get(
        "/queue",
        response_model=QueueStats,
        summary="Queue depth and the age of the oldest waiting job",
        description=(
            "`oldest_claimable_age_seconds` is the autoscaling signal. Scaling on mean depth "
            "is the trap: audio arrives in bursts."
        ),
    )
    async def queue_stats(request: Request) -> QueueStats:
        current: Settings = request.app.state.settings
        stats = await anyio.to_thread.run_sync(lambda: jobstore.queue_stats(current))
        return QueueStats(**stats)

    return router


def _store_and_submit(
    settings: Settings,
    *,
    store: blobstore.BlobStore,
    staged: Path,
    info,
    tenant_id: uuid.UUID,
    display_name: str,
    model: str,
    language: str | None,
    word_timestamps: bool,
    diarise: bool,
    idempotency_key: str | None,
) -> tuple[jobstore.Job, bool]:
    """Put the bytes somewhere durable, then record the intent.

    Order matters. Storing first means a job row never points at audio that does not exist;
    the opposite order leaves a window where a worker can claim a job whose bytes are still in
    flight. The cost of this order is an orphaned blob if the insert fails, which is harmless
    and cheap to reap — content addressing means a retry writes to the same key anyway.
    """
    uri = store.put(staged, info.sha256)
    return jobstore.submit(
        settings,
        tenant_id=tenant_id,
        audio_uri=uri,
        audio_sha256=info.sha256,
        model=model,
        display_name=display_name,
        duration_seconds=info.duration_seconds,
        size_bytes=info.size_bytes,
        language=language,
        word_timestamps=word_timestamps,
        diarise=diarise,
        diarisation_backend=settings.diarisation_backend,
        idempotency_key=idempotency_key,
    )


def _blob_still_referenced(settings: Settings, audio_sha256: str) -> bool:
    from transcription import db

    with db.transaction(settings) as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE audio_sha256 = %s LIMIT 1", (audio_sha256,)
        ).fetchone()
    return row is not None


async def _spool(upload: UploadFile, destination: Path, max_bytes: int) -> None:
    """Copy an upload to disk a megabyte at a time, enforcing the byte limit as we go.

    Chunked rather than ``await upload.read()``, so a 200 MB upload does not become 200 MB of
    resident memory. The limit is re-checked here as well as in middleware because a client
    that sends no ``Content-Length`` gets past the middleware check.
    """
    from transcription.errors import UploadTooLargeError

    written = 0
    with destination.open("wb") as sink:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                raise UploadTooLargeError(
                    f"Upload exceeds the {max_bytes} byte limit.", limit_bytes=max_bytes
                )
            sink.write(chunk)

    if written == 0:
        from transcription.errors import EmptyAudioError

        raise EmptyAudioError("The uploaded file is empty.")
