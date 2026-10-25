"""HTTP API.

Deliberately synchronous, with a documented duration limit.

Transcription takes minutes, which makes a blocking HTTP request the wrong shape for long
audio. The correct fix is an asynchronous job API, and that requires durable job state,
retries and a worker tier — production infrastructure that is designed in docs/DESIGN.md
rather than built here. So rather than half-build a job system, this endpoint:

* transcribes inline and returns the full result, which is the right behaviour for the short
  clips an interactive caller actually sends;
* refuses audio longer than ``max_sync_duration_seconds`` with a 413 that names the limit and
  points at the CLI. An explicit limit is better engineering than a silent gateway timeout;
* bounds in-flight work with a semaphore. Note what that does and does not buy: a single
  ``WhisperModel`` defaults to ``num_workers=1``, so CTranslate2 serializes concurrent
  ``transcribe()`` calls anyway — and with ``cpu_threads=0`` one request already uses every
  core. The semaphore bounds queue depth and peak memory, and lets one request's ffmpeg
  decode overlap another's inference. It is *not* metering parallel inference, and real
  parallelism is the worker pool in docs/DESIGN.md;
* runs the CPU-bound work in a worker thread, so the event loop stays free to serve health
  checks and reject malformed requests while a transcription is in progress.

docs/DESIGN.md describes the asynchronous architecture this would become in production.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import anyio
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse

from transcription import exporters
from transcription.config import Settings, get_settings
from transcription.errors import (
    AudioTooLongError,
    CorruptAudioError,
    EmptyAudioError,
    NoAudioStreamError,
    NormalizationError,
    TranscriptionError,
    TranscriptionFailedError,
    UploadTooLargeError,
)
from transcription.exporters import ResponseFormat
from transcription.models import ErrorResponse, HealthResponse, TranscriptionResult
from transcription.pipeline import safe_filename, transcribe_file
from transcription.transcriber import FasterWhisperTranscriber, Transcriber

logger = logging.getLogger("transcription.api")

_UPLOAD_CHUNK_BYTES = 1024 * 1024

# Multipart framing (boundaries, part headers) sits on top of the file bytes, so a legitimate
# at-the-limit upload declares slightly more than the limit. A small allowance keeps it from
# being rejected on framing alone.
_MULTIPART_OVERHEAD_BYTES = 8 * 1024

API_DESCRIPTION = """
Converts audio into text with a timestamp for every segment.

### Quick start

Expand **POST /v1/transcriptions** below, click **Try it out**, choose an audio file and
press **Execute**. Any format ffmpeg can read works — mp3, wav, flac, m4a, ogg, or the audio
track inside an mp4.

The same thing with `curl`:

```bash
curl -F file=@samples/speech_en.mp3 http://localhost:8000/v1/transcriptions

# subtitles instead of JSON
curl -F file=@samples/speech_en.mp3 -F response_format=srt \\
     http://localhost:8000/v1/transcriptions
```

### Things worth knowing

* **Formats are detected from content, not the filename.** A FLAC file named `.mp3`
  transcribes correctly; a text file named `.mp3` is rejected with `415`.
* **This endpoint is synchronous**, so it caps audio length (see `/healthz` for the current
  limit) and returns `413` beyond it. Use the CLI for long files. The asynchronous job API
  this would become in production is described in `docs/DESIGN.md`.
* **Errors are machine-readable.** Every failure returns `{"error": {"code": ..., "message":
  ...}}`; match on `code`, never on the message text.
* **Results are content-addressed.** The `id` is the SHA-256 of the audio, so uploading the
  same file twice gives the same `id`.
* **Every response is timed.** `Server-Timing: inference;dur=..., total;dur=...` in
  milliseconds — `inference` is model time, `total` adds ffmpeg decoding and any queue wait.
  Present on subtitle and plain-text responses too, which have no body field for it, and on
  errors (`total` only).
"""

TAGS_METADATA = [
    {
        "name": "transcription",
        "description": "Turn audio into timestamped text.",
    },
    {
        "name": "ops",
        "description": "Health and configuration, for monitoring and for discovering limits.",
    },
]


def _error_responses(
    *error_types: type[TranscriptionError],
) -> dict[int | str, dict[str, object]]:
    """Build OpenAPI response docs from the error classes themselves.

    Generating these from `errors.py` rather than hand-writing them means the documented
    status codes and error codes cannot drift away from what the code actually raises.
    """
    grouped: dict[int, list[type[TranscriptionError]]] = {}
    for error_type in error_types:
        grouped.setdefault(error_type.http_status, []).append(error_type)

    responses: dict[int | str, dict[str, object]] = {}
    for status, types in sorted(grouped.items()):
        responses[status] = {
            "model": ErrorResponse,
            "description": " or ".join(t.code for t in types),
            "content": {
                "application/json": {
                    "examples": {
                        t.code: {
                            "summary": t.code,
                            "value": {
                                "error": {
                                    "code": t.code,
                                    "message": (t.__doc__ or "").strip().split("\n")[0],
                                }
                            },
                        }
                        for t in types
                    }
                }
            },
        }
    return responses


# Real output from `transcribe samples/speech_en.mp3 --format srt|vtt`, not hand-written.
_SRT_EXAMPLE = (
    "1\n00:00:00,000 --> 00:00:03,600\nquick brown fox jumps over the lazy dog.\n\n"
    "2\n00:00:03,600 --> 00:00:07,200\n"
    "This recording is a test fixture for the audio transcription service.\n"
)
_VTT_EXAMPLE = (
    "WEBVTT\n\n00:00:00.000 --> 00:00:03.600\nquick brown fox jumps over the lazy dog.\n"
)


def create_app(
    settings: Settings | None = None,
    transcriber: Transcriber | None = None,
) -> FastAPI:
    """Build the application.

    Both dependencies are injectable so the test suite can run the real HTTP stack against a
    fake engine — no model download, no GPU, whole suite in seconds.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved_settings = settings or get_settings()
        # Created inside the running loop, and shared by every request.
        app.state.capacity = anyio.Semaphore(resolved_settings.max_concurrent_transcriptions)

        if transcriber is not None:
            app.state.transcriber = transcriber
        else:
            engine = FasterWhisperTranscriber(resolved_settings)
            # Load weights once at startup rather than on the first unlucky request.
            await anyio.to_thread.run_sync(engine.warm)
            app.state.transcriber = engine
            logger.info("model %s ready", resolved_settings.whisper_model)
        yield

    app = FastAPI(
        title="Audio Transcription Service",
        version="1.0.0",
        summary="Converts audio into timestamped text segments.",
        description=API_DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def reject_oversized_bodies(request: Request, call_next):
        """Refuse an over-limit upload on its declared length, before the body is read.

        This has to be middleware rather than a check inside the handler. FastAPI resolves
        ``File(...)`` by awaiting ``request.form()`` during dependency resolution, which
        receives and parses the *entire* multipart body before the handler's first line runs
        — so a check in the handler can only ever reject after the fact. Middleware runs
        first, so a client that declares its length is turned away without sending a body.

        Two honest limits. A client that sends no ``Content-Length`` (chunked) still has its
        bytes received and spooled by Starlette — to disk past 1 MB, not memory — and is
        caught by the second check in the handler. And this bounds the whole request body,
        which for a multipart upload is the file plus framing; that is the point, since the
        body is what costs us.

        Returns the response directly rather than raising: Starlette's exception handlers sit
        inside the user middleware stack, so a raise here would bypass the envelope that
        every other error uses.
        """
        settings_for_request: Settings | None = getattr(request.app.state, "settings", None)
        declared = request.headers.get("content-length")
        if settings_for_request is not None and declared is not None and declared.isdigit():
            limit = settings_for_request.max_upload_bytes
            if int(declared) > limit + _MULTIPART_OVERHEAD_BYTES:
                error = UploadTooLargeError(
                    f"Upload exceeds the {limit // (1024 * 1024)} MB limit.",
                    limit_bytes=limit,
                )
                logger.info("rejected request: %s %s", error.code, error.message)
                return JSONResponse(status_code=error.http_status, content=error.to_dict())
        return await call_next(request)

    # Registered after the size check so it wraps it: FastAPI runs the last-added middleware
    # outermost, which means rejections get timed too.
    @app.middleware("http")
    async def add_server_timing(request: Request, call_next):
        """Report how long the request took in a ``Server-Timing`` header.

        ``processing_seconds`` in the JSON body is inference only, and the subtitle and plain
        text formats have no body to carry it at all — so without this, a caller asking for
        SRT could not see any timing, and nobody could see the cost of ffmpeg decoding.

        ``Server-Timing`` rather than an ad-hoc ``X-`` header because it is the W3C standard
        for this, carries several named metrics at once, and browsers surface it in devtools.
        Durations are milliseconds, as the spec requires. The handler adds an ``inference``
        metric; this adds ``total``, which also covers normalisation and queue wait.
        """
        started = time.perf_counter()
        response = await call_next(request)
        total = f"total;dur={(time.perf_counter() - started) * 1000:.1f}"
        inference = response.headers.get("server-timing")
        response.headers["Server-Timing"] = f"{inference}, {total}" if inference else total
        return response

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Send anyone who opens the bare host straight to the documentation."""
        return RedirectResponse(url="/docs")

    @app.exception_handler(TranscriptionError)
    async def handle_transcription_error(_: Request, exc: TranscriptionError) -> JSONResponse:
        """One handler for every domain error; status and code come off the exception."""
        # Full details (including raw ffmpeg stderr) go to the log; to_dict() strips the
        # parts that would leak server-side paths to the caller.
        if exc.http_status >= 500:
            logger.exception("request failed: %s %s %s", exc.code, exc.message, exc.details)
        else:
            logger.info("rejected request: %s %s %s", exc.code, exc.message, exc.details)
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Reshape FastAPI's validation errors into the same envelope as everything else.

        FastAPI returns ``{"detail": [...]}`` by default, which would make this the one
        failure shape a client has to special-case. The contract promises that every error
        carries a `code`, so validation errors get the same treatment, with the per-field
        detail preserved under `fields`.
        """
        problems = jsonable_encoder(exc.errors())
        summary = "; ".join(
            f"{'.'.join(str(part) for part in problem.get('loc', [])[1:]) or 'request'}: "
            f"{problem.get('msg', 'invalid')}"
            for problem in problems
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": summary or "The request could not be validated.",
                    "fields": problems,
                }
            },
        )

    @app.get(
        "/healthz",
        tags=["ops"],
        summary="Health check and current limits",
        description=(
            "Returns `ok` once the model is loaded, along with the limits a client should "
            "know before uploading. Call this first if an upload is being rejected."
        ),
        response_model=HealthResponse,
        response_description="Service is ready.",
    )
    async def healthz(request: Request) -> dict[str, object]:
        current: Settings = request.app.state.settings
        return {
            "status": "ok",
            "model": request.app.state.transcriber.name,
            "device": current.device,
            "max_sync_duration_seconds": current.max_sync_duration_seconds,
            "max_upload_bytes": current.max_upload_bytes,
            "max_concurrent_transcriptions": current.max_concurrent_transcriptions,
        }

    @app.post(
        "/v1/transcriptions",
        tags=["transcription"],
        summary="Transcribe an audio file",
        response_model=TranscriptionResult,
        response_model_exclude_none=False,
        response_description="The transcript, with a timestamp for every segment.",
        responses={
            200: {
                "description": (
                    "Transcription succeeded. The content type depends on `response_format`."
                ),
                "content": {
                    "application/json": {},
                    "text/plain": {
                        "schema": {"type": "string"},
                        "example": _SRT_EXAMPLE,
                    },
                    "text/vtt": {
                        "schema": {"type": "string"},
                        "example": _VTT_EXAMPLE,
                    },
                },
            },
            **_error_responses(
                EmptyAudioError,
                CorruptAudioError,
                NoAudioStreamError,
                NormalizationError,
                AudioTooLongError,
                UploadTooLargeError,
                TranscriptionFailedError,
            ),
            # Documented by hand because it comes from request validation rather than from
            # an error class. FastAPI would otherwise describe it with its own default shape.
            422: {
                "model": ErrorResponse,
                "description": "invalid_request",
                "content": {
                    "application/json": {
                        "examples": {
                            "invalid_request": {
                                "summary": "invalid_request",
                                "value": {
                                    "error": {
                                        "code": "invalid_request",
                                        "message": (
                                            "response_format: Input should be 'json', 'srt', "
                                            "'vtt' or 'txt'"
                                        ),
                                    }
                                },
                            }
                        }
                    }
                },
            },
        },
    )
    async def create_transcription(
        request: Request,
        file: UploadFile = File(
            ...,
            description=(
                "The audio file. Any format ffmpeg can read: mp3, wav, flac, m4a, ogg, or an "
                "mp4 with an audio track. Detected from content, so the extension may be wrong."
            ),
        ),
        language: str | None = Form(
            None,
            description=(
                "Force a two-letter language code such as `en` or `hi`. Leave blank to "
                "auto-detect, which is usually right and costs little."
            ),
            examples=["en"],
        ),
        word_timestamps: bool = Form(
            False,
            description=(
                "Also return a timestamp for every word. More accurate segment boundaries "
                "too, for roughly 10-25% more compute."
            ),
        ),
        response_format: ResponseFormat = Form(
            ResponseFormat.json,
            description=(
                "`json` for the full result, `srt` or `vtt` for subtitles, `txt` for plain "
                "prose."
            ),
        ),
    ):
        """
        Upload an audio file and get back the spoken text with timestamps.

        Pick a file, press **Execute**, and the response appears below. Try
        `samples/speech_en.mp3` from the repository if you need something to hand.

        **How it works:** the file is probed with `ffprobe` to find out what it really is,
        converted to 16 kHz mono with `ffmpeg`, then transcribed with Whisper. Long audio is
        read in windows cut at silence, so peak memory tracks the window rather than the
        recording, and no word is split at a seam.
        """
        current: Settings = request.app.state.settings
        output_format = response_format.value
        display_name = safe_filename(file.filename)

        with tempfile.TemporaryDirectory(prefix="upload-") as tmp_dir:
            # The client's filename is never used to build this path — only the fixed name
            # below. The body is copied a megabyte at a time rather than held in memory, and
            # re-checked against the limit as it goes. The re-check is what catches a chunked
            # upload that declared no length and so slipped past the middleware above.
            upload_path = Path(tmp_dir) / "upload.bin"
            written = 0
            with upload_path.open("wb") as handle:
                while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if written > current.max_upload_bytes:
                        raise UploadTooLargeError(
                            "Upload exceeds the "
                            f"{current.max_upload_bytes // (1024 * 1024)} MB limit.",
                            limit_bytes=current.max_upload_bytes,
                        )
                    handle.write(chunk)

            if written == 0:
                raise EmptyAudioError("The uploaded file is empty (0 bytes).")

            work = partial(
                transcribe_file,
                upload_path,
                transcriber=request.app.state.transcriber,
                settings=current,
                language=language,
                word_timestamps=word_timestamps,
                # The synchronous endpoint's own, stricter limit.
                max_duration_seconds=current.max_sync_duration_seconds,
                display_name=display_name,
            )

            # Admission control, then hand the CPU-bound work to a thread so the event loop
            # is never blocked by the model.
            async with request.app.state.capacity:
                try:
                    result: TranscriptionResult = await anyio.to_thread.run_sync(work)
                except AudioTooLongError as exc:
                    exc.details["hint"] = (
                        "This endpoint transcribes synchronously. Use the CLI "
                        "(`transcribe <file>`) for longer audio, or see docs/DESIGN.md for the "
                        "asynchronous job API this would become in production."
                    )
                    raise

        # Carried on every format, including the plain-text ones that have nowhere in the
        # body to put it. The middleware appends `total` alongside.
        timing = {"Server-Timing": f"inference;dur={result.processing_seconds * 1000:.1f}"}

        if output_format == "json":
            return JSONResponse(content=result.model_dump(mode="json"), headers=timing)

        return PlainTextResponse(
            content=exporters.render(result, output_format),
            media_type=exporters.MEDIA_TYPES[output_format],
            headers=timing,
        )

    return app


# Module-level instance for `uvicorn transcription.api:app`.
app = create_app()
