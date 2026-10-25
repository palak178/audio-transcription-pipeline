"""Tests for the HTTP layer, against the real ASGI stack with a fake engine."""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import FakeTranscriber, ffmpeg_required

from transcription.api import create_app
from transcription.config import Settings

pytestmark = ffmpeg_required


def upload(path: Path, name: str | None = None) -> dict:
    return {"file": (name or path.name, path.read_bytes(), "application/octet-stream")}


# --- Health ---------------------------------------------------------------------------


def test_healthz_reports_configuration(client: TestClient) -> None:
    body = client.get("/healthz").json()

    assert body["status"] == "ok"
    assert body["model"] == "fake"
    assert body["max_sync_duration_seconds"] == 60


def test_openapi_schema_is_served(client: TestClient) -> None:
    """A documented API is part of exposing one; FastAPI generates this from the handlers."""
    schema = client.get("/openapi.json").json()

    assert "/v1/transcriptions" in schema["paths"]


# --- Happy paths ----------------------------------------------------------------------


def test_returns_timestamped_segments(client: TestClient, make_audio) -> None:
    source = make_audio("clip.mp3", seconds=2.0, extra=["-c:a", "libmp3lame"])

    response = client.post("/v1/transcriptions", files=upload(source))

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.0"
    assert body["text"] == "Hello world. Second segment."
    assert body["audio"]["filename"] == "clip.mp3"
    assert body["audio"]["codec"] == "mp3"
    first = body["segments"][0]
    assert {"id", "start", "end", "text", "duration", "confidence"} <= set(first)


@pytest.mark.parametrize(
    ("response_format", "expected_fragment", "expected_media_type"),
    [
        ("srt", "00:00:00,000 --> 00:00:01,500", "text/plain"),
        ("vtt", "WEBVTT", "text/vtt"),
        ("txt", "Hello world.", "text/plain"),
    ],
)
def test_alternate_response_formats(
    client: TestClient, make_audio, response_format, expected_fragment, expected_media_type
) -> None:
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    response = client.post(
        "/v1/transcriptions",
        files=upload(source),
        data={"response_format": response_format},
    )

    assert response.status_code == 200
    assert expected_fragment in response.text
    assert response.headers["content-type"].startswith(expected_media_type)


def test_form_options_reach_the_engine(
    client: TestClient, make_audio, fake_transcriber: FakeTranscriber
) -> None:
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    client.post(
        "/v1/transcriptions",
        files=upload(source),
        data={"language": "de", "word_timestamps": "true"},
    )

    assert fake_transcriber.calls[0]["language"] == "de"
    assert fake_transcriber.calls[0]["word_timestamps"] is True


def test_format_is_detected_from_content_not_filename(client: TestClient, make_audio) -> None:
    """A FLAC payload uploaded as `.mp3` still transcribes."""
    source = make_audio("real.flac", extra=["-c:a", "flac"])

    response = client.post("/v1/transcriptions", files=upload(source, name="lying.mp3"))

    assert response.status_code == 200
    assert response.json()["audio"]["codec"] == "flac"


def test_hostile_filename_is_sanitised(client: TestClient, make_audio) -> None:
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    response = client.post(
        "/v1/transcriptions",
        files=upload(source, name="../../../etc/passwd.wav"),
    )

    assert response.status_code == 200
    assert response.json()["audio"]["filename"] == "passwd.wav"


# --- Timing -----------------------------------------------------------------------------


def _metrics(header: str) -> dict[str, float]:
    """Parse `Server-Timing: name;dur=1.5, other;dur=2` into {name: duration}."""
    out = {}
    for part in header.split(","):
        name, _, rest = part.strip().partition(";")
        out[name] = float(rest.removeprefix("dur="))
    return out


def test_json_response_reports_timing(client: TestClient, make_audio) -> None:
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    response = client.post("/v1/transcriptions", files=upload(source))

    metrics = _metrics(response.headers["server-timing"])
    assert set(metrics) == {"inference", "total"}
    # `total` covers ffmpeg and the queue wait as well, so it cannot be the smaller of the two.
    assert metrics["total"] >= metrics["inference"]
    # The header must agree with the body rather than being measured separately.
    assert metrics["inference"] == pytest.approx(
        response.json()["processing_seconds"] * 1000, abs=1.0
    )


@pytest.mark.parametrize("response_format", ["srt", "vtt", "txt"])
def test_text_responses_report_timing_too(client: TestClient, make_audio, response_format) -> None:
    """The reason this header exists: these formats have no body field to carry timing."""
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    response = client.post(
        "/v1/transcriptions",
        files=upload(source),
        data={"response_format": response_format},
    )

    assert set(_metrics(response.headers["server-timing"])) == {"inference", "total"}


def test_failed_requests_are_timed(client: TestClient) -> None:
    """A caller debugging a slow rejection needs the number too, so middleware wraps errors."""
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("notes.mp3", b"this is not audio at all", "application/octet-stream")},
    )

    assert response.status_code == 415
    # No inference ran, so only `total` is present.
    assert set(_metrics(response.headers["server-timing"])) == {"total"}


# --- Rejections -----------------------------------------------------------------------


def test_missing_file_field_is_a_validation_error(client: TestClient) -> None:
    response = client.post("/v1/transcriptions")

    assert response.status_code == 422
    # Validation failures use the same envelope as every other error, so a client never has
    # to special-case one shape.
    assert response.json()["error"]["code"] == "invalid_request"
    assert "file" in response.json()["error"]["message"]


def test_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("empty.mp3", b"", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_audio"


def test_non_audio_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("notes.mp3", b"this is not audio at all", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "corrupt_audio"


def test_errors_do_not_leak_server_paths(client: TestClient) -> None:
    """ffmpeg's stderr names the temp file it was handed; the client must not see it."""
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("notes.mp3", b"this is not audio at all", "application/octet-stream")},
    )

    body = response.text
    assert "ffprobe_error" not in body
    # Checked against the real temp directory rather than a hardcoded "/tmp", so this stays
    # meaningful on Windows, where it is somewhere under AppData\Local\Temp.
    assert tempfile.gettempdir() not in body
    assert "upload.bin" not in body


def test_unknown_response_format_is_rejected(client: TestClient, make_audio) -> None:
    """`response_format` is an enum, so validation rejects it before any work starts."""
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    response = client.post(
        "/v1/transcriptions",
        files=upload(source),
        data={"response_format": "docx"},
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    # The message names the offending field and lists what was allowed.
    assert "response_format" in error["message"]
    for allowed in ("json", "srt", "vtt", "txt"):
        assert allowed in error["message"]


def test_oversized_upload_is_rejected(client: TestClient) -> None:
    """Just over the limit: caught by the re-check while the body is copied.

    Note what this does *not* claim. FastAPI resolves `File(...)` by awaiting
    `request.form()`, which receives and parses the whole multipart body before the handler
    runs — so a handler-level check can only reject after the fact. Rejecting *before* the
    body arrives is the middleware's job, tested below.
    """
    oversized = b"\x00" * (1024 * 1024 + 1)  # the test settings cap uploads at 1 MB

    response = client.post(
        "/v1/transcriptions",
        files={"file": ("big.wav", oversized, "application/octet-stream")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_oversized_upload_is_rejected_before_the_body_is_read(
    client: TestClient, monkeypatch
) -> None:
    """A declared Content-Length over the limit is refused without reading the body.

    Proven by making any read of the upload fail: if the response is still a clean 413, the
    rejection cannot have come from the copy loop.
    """
    from starlette.datastructures import UploadFile

    async def explode(self, size: int = -1) -> bytes:
        raise AssertionError("the body must not be read when Content-Length is over the limit")

    monkeypatch.setattr(UploadFile, "read", explode)
    way_oversized = b"\x00" * (4 * 1024 * 1024)  # 4 MB against the 1 MB test limit

    response = client.post(
        "/v1/transcriptions",
        files={"file": ("huge.wav", way_oversized, "application/octet-stream")},
    )

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "upload_too_large"
    # Same envelope as every other failure, even though it came from middleware.
    assert error["limit_bytes"] == 1024 * 1024


def test_upload_at_the_limit_is_still_accepted(client: TestClient, make_audio) -> None:
    """The multipart-framing allowance must not reject a legitimate at-the-limit upload."""
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    response = client.post("/v1/transcriptions", files=upload(source))

    assert response.status_code == 200


def test_audio_over_the_sync_limit_is_rejected_with_a_pointer(make_audio) -> None:
    """The documented limit, and the 413 that explains what to do instead."""
    tight = Settings(whisper_model="fake", max_sync_duration_seconds=1.0)
    app = create_app(settings=tight, transcriber=FakeTranscriber())
    source = make_audio("long.mp3", seconds=4.0, extra=["-c:a", "libmp3lame"])

    with TestClient(app) as tight_client:
        response = tight_client.post("/v1/transcriptions", files=upload(source))

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "audio_too_long"
    assert error["limit_seconds"] == 1.0
    assert "CLI" in error["hint"]


def test_engine_failure_returns_500_with_a_code(make_audio) -> None:
    from transcription.errors import TranscriptionFailedError

    app = create_app(
        settings=Settings(whisper_model="fake"),
        transcriber=FakeTranscriber(error=TranscriptionFailedError("model died")),
    )
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])

    with TestClient(app, raise_server_exceptions=False) as failing_client:
        response = failing_client.post("/v1/transcriptions", files=upload(source))

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "transcription_failed"


# --- Concurrency ----------------------------------------------------------------------


def test_concurrent_uploads_all_succeed(client: TestClient, make_audio) -> None:
    """More simultaneous requests than the semaphore allows: they queue, none fail.

    This only proves the in-process admission control is sound. Scaling concurrency properly
    is a Part 2 topic — see docs/DESIGN.md.
    """
    source = make_audio("clip.wav", extra=["-c:a", "pcm_s16le"])
    payload = source.read_bytes()

    def send(index: int):
        return client.post(
            "/v1/transcriptions",
            files={"file": (f"clip{index}.wav", payload, "application/octet-stream")},
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(send, range(6)))

    assert [r.status_code for r in responses] == [200] * 6
    # Same bytes in, same content-addressed id out, every time.
    assert len({r.json()["id"] for r in responses}) == 1


# --- API documentation ----------------------------------------------------------------
#
# The Swagger page is how a reviewer will try this service, so the parts that make it
# usable are worth protecting with tests.


def test_docs_pages_are_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_root_redirects_to_docs(client: TestClient) -> None:
    """Opening the bare host should land somewhere useful, not on a 404."""
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_openapi_documents_the_endpoints(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    assert spec["info"]["title"] == "Audio Transcription Service"
    assert "Quick start" in spec["info"]["description"]
    assert set(spec["paths"]) >= {"/v1/transcriptions", "/healthz"}
    assert {tag["name"] for tag in spec["tags"]} == {"transcription", "ops"}


def test_openapi_documents_every_error_status(client: TestClient) -> None:
    """Documented failures must match what the code can actually raise."""
    post = client.get("/openapi.json").json()["paths"]["/v1/transcriptions"]["post"]

    assert {"200", "400", "413", "415", "422", "500"} <= set(post["responses"])

    def codes(status: str) -> set[str]:
        return set(post["responses"][status]["content"]["application/json"]["examples"])

    # 415 covers everything that means "we cannot decode this media".
    assert codes("415") == {"corrupt_audio", "no_audio_stream", "normalization_failed"}
    assert codes("413") == {"audio_too_long", "upload_too_large"}
    assert codes("400") == {"empty_audio"}
    # 422 is reserved for request validation, so one status never means two things.
    assert codes("422") == {"invalid_request"}
    assert codes("500") == {"transcription_failed"}


def test_openapi_documents_all_response_content_types(client: TestClient) -> None:
    post = client.get("/openapi.json").json()["paths"]["/v1/transcriptions"]["post"]

    assert set(post["responses"]["200"]["content"]) == {
        "application/json",
        "text/plain",
        "text/vtt",
    }


def test_openapi_offers_response_format_as_a_dropdown(client: TestClient) -> None:
    """An enum means Swagger renders a select box instead of a free-text field."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    assert schemas["ResponseFormat"]["enum"] == ["json", "srt", "vtt", "txt"]


def test_openapi_includes_a_full_result_example(client: TestClient) -> None:
    """So callers can see the shape of a result without running anything."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    example = schemas["TranscriptionResult"]["example"]

    assert example["schema_version"] == "1.0"
    assert example["segments"][0]["text"]
    assert "real_time_factor" in example


def test_openapi_describes_the_upload_fields(client: TestClient) -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    body = next(name for name in schemas if name.startswith("Body_"))
    properties = schemas[body]["properties"]

    assert set(properties) == {"file", "language", "word_timestamps", "response_format"}
    # Every field carries help text, which is what Swagger shows beside the input.
    for name in ("file", "language", "word_timestamps"):
        assert properties[name].get("description"), f"{name} has no description"


def test_health_response_is_documented_with_a_schema(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    get = spec["paths"]["/healthz"]["get"]

    ref = get["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("HealthResponse")
    assert set(spec["components"]["schemas"]["HealthResponse"]["properties"]) == {
        "status",
        "model",
        "device",
        "max_sync_duration_seconds",
        "max_upload_bytes",
        "max_concurrent_transcriptions",
    }
