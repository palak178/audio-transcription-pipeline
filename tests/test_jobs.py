"""Tests for the asynchronous job tier: the queue, leases, retry policy and the job API.

These carry the weight of the whole production design, so the ones that matter are the ones
that pin down behaviour nobody can verify by reading:

* ``test_concurrent_workers_never_claim_the_same_job`` — what ``FOR UPDATE SKIP LOCKED`` is
  for, run from real threads on real connections.
* ``test_an_expired_lease_returns_the_job_to_the_queue`` — the half of at-least-once delivery
  that a message broker would normally provide.
* ``test_retries_back_off_then_dead_letter`` — the retry policy, end to end.
* ``test_identical_options_dedupe_but_different_options_do_not`` — the idempotency bug the
  design had, now a test.

Every test here needs a real Postgres. They are skipped, not failed, when there is not one;
see ``postgres_required`` in conftest.
"""

from __future__ import annotations

import threading
import uuid

import pytest
from fastapi.testclient import TestClient

from conftest import FakeTranscriber, ffmpeg_required, postgres_required
from transcription import audio as audio_module
from transcription import blobstore, db, jobstore
from transcription.api import create_app
from transcription.worker import Worker

pytestmark = postgres_required


def submit(settings, tenant, *, sha="a" * 64, **overrides):
    payload = {
        "tenant_id": tenant,
        "audio_uri": f"file:///tmp/{sha}",
        "audio_sha256": sha,
        "model": "base",
    }
    payload.update(overrides)
    return jobstore.submit(settings, **payload)


# --- Identity and idempotency -------------------------------------------------------------


def test_identical_options_dedupe_but_different_options_do_not(job_settings, tenant) -> None:
    """The design keyed idempotency on (tenant, audio, model). That was a bug.

    A `diarise: true` request would have matched an earlier plain run, and the caller would
    have received a 200 carrying a transcript with no speaker labels and no error — a wrong
    answer wearing a success. Anything that changes the output has to be part of identity.
    """
    first, created_first = submit(job_settings, tenant)
    again, created_again = submit(job_settings, tenant)

    assert created_first is True
    assert created_again is False
    assert first.id == again.id

    # Each of these changes the output, so each must be a distinct job.
    for option in (
        {"diarise": True},
        {"word_timestamps": True},
        {"language": "fr"},
        {"model": "small"},
    ):
        variant, created = submit(job_settings, tenant, **option)
        assert created is True, f"{option} should not de-duplicate against a plain run"
        assert variant.id != first.id


def test_tenants_do_not_share_jobs(job_settings) -> None:
    one, two = uuid.uuid4(), uuid.uuid4()
    job_one, _ = submit(job_settings, one)
    job_two, created = submit(job_settings, two)

    assert created is True
    assert job_one.id != job_two.id, "identical audio for two tenants is two jobs"
    assert jobstore.get(job_settings, job_id=job_one.id, tenant_id=two) is None


def test_an_explicit_idempotency_key_also_dedupes(job_settings, tenant) -> None:
    first, created = submit(job_settings, tenant, idempotency_key="abc", sha="b" * 64)
    # Different audio entirely, same key: the caller asserted these are one request.
    again, created_again = submit(job_settings, tenant, idempotency_key="abc", sha="c" * 64)

    assert created is True
    assert created_again is False
    assert again.id == first.id


# --- The queue ----------------------------------------------------------------------------


def test_concurrent_workers_never_claim_the_same_job(job_settings, tenant) -> None:
    """What SKIP LOCKED exists for.

    Eight threads on eight separate connections race for eight jobs. Every job must go to
    exactly one worker, and nobody may come back empty-handed while work remains. Without
    SKIP LOCKED the workers serialise behind the head of the queue; without the enclosing
    transaction two of them could claim the same row.
    """
    count = 8
    for index in range(count):
        submit(job_settings, tenant, sha=f"{index:064d}")

    claimed: list[uuid.UUID] = []
    lock = threading.Lock()
    barrier = threading.Barrier(count)

    def worker(worker_index: int) -> None:
        barrier.wait()  # maximise the overlap
        job = jobstore.claim(job_settings, worker_id=f"w{worker_index}")
        if job is not None:
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(claimed) == count, "every job should have been claimed exactly once"
    assert len(set(claimed)) == count, "two workers claimed the same job"


def test_a_deferred_job_is_not_claimable_yet(job_settings, tenant) -> None:
    """Backoff is scheduling, not a second queue.

    A retryable failure leaves a `queued` row with `available_at` in the future. There is
    still only one queue, and nothing to keep in sync with it.
    """
    job, _ = submit(job_settings, tenant)
    claimed = jobstore.claim(job_settings, worker_id="w1")
    jobstore.fail(
        job_settings,
        job_id=claimed.id,
        worker_id="w1",
        error_code="transcription_failed",
        retryable=True,
    )

    requeued = jobstore.get(job_settings, job_id=job.id)
    assert requeued.status == "queued"
    assert jobstore.claim(job_settings, worker_id="w2") is None, "backoff was not respected"

    with db.transaction(job_settings) as conn:
        conn.execute("UPDATE jobs SET available_at = now() WHERE id = %s", (job.id,))
    assert jobstore.claim(job_settings, worker_id="w2") is not None


def test_latency_classes_are_claimed_separately(job_settings, tenant) -> None:
    """So a batch of long recordings cannot starve interactive clips."""
    submit(job_settings, tenant, sha="d" * 64, duration_seconds=10.0)
    submit(job_settings, tenant, sha="e" * 64, duration_seconds=100_000.0)

    short = jobstore.claim(job_settings, worker_id="w-short", latency_class="short")
    long_job = jobstore.claim(job_settings, worker_id="w-long", latency_class="long")

    assert short is not None and short.latency_class == "short"
    assert long_job is not None and long_job.latency_class == "long"


def test_diarisation_pushes_a_job_into_the_long_class(job_settings) -> None:
    """Routing on duration alone gets this wrong once diarisation exists.

    A diarised clip costs roughly twice a plain one, so it belongs with the long work even
    though its duration says otherwise.
    """
    just_under = job_settings.long_job_threshold_seconds / 0.30 * 0.9

    assert (
        jobstore.latency_class_for(just_under, diarise=False, settings=job_settings) == "short"
    )
    assert jobstore.latency_class_for(just_under, diarise=True, settings=job_settings) == "long"


# --- Leases -------------------------------------------------------------------------------


def test_only_the_lease_holder_can_heartbeat_or_finish(job_settings, tenant) -> None:
    """Otherwise any worker could refresh any lease, and the guarantee would be decorative."""
    submit(job_settings, tenant)
    job = jobstore.claim(job_settings, worker_id="owner")

    assert jobstore.heartbeat(job_settings, job_id=job.id, worker_id="owner", progress=0.5)
    assert not jobstore.heartbeat(job_settings, job_id=job.id, worker_id="impostor")
    assert not jobstore.succeed(
        job_settings, job_id=job.id, worker_id="impostor", result={"bad": True}
    )
    assert jobstore.get_result(job_settings, job_id=job.id) is None

    assert jobstore.succeed(
        job_settings, job_id=job.id, worker_id="owner", result={"text": "ok"}
    )
    assert jobstore.get_result(job_settings, job_id=job.id) == {"text": "ok"}


def test_an_expired_lease_returns_the_job_to_the_queue(job_settings, tenant) -> None:
    """The half of at-least-once delivery a broker would normally provide.

    A worker that is OOM-killed, evicted or reclaimed as a spot instance stops heartbeating.
    Its lease lapses and the job becomes claimable again, rather than sitting in `running`
    forever.
    """
    submit(job_settings, tenant)
    abandoned = jobstore.claim(job_settings, worker_id="doomed", lease_seconds=30)

    # Reach in and expire the lease, rather than sleeping: this is what a dead worker looks
    # like to the database, and a test should not take 30 seconds to say so.
    with db.transaction(job_settings) as conn:
        conn.execute(
            "UPDATE jobs SET locked_until = now() - interval '1 second' WHERE id = %s",
            (abandoned.id,),
        )

    assert jobstore.reclaim_expired(job_settings) == [(abandoned.id, "queued")]

    retried = jobstore.claim(job_settings, worker_id="fresh")
    assert retried is not None
    assert retried.id == abandoned.id
    assert retried.attempts == 2, "the redelivery should count as another attempt"


def test_a_job_that_keeps_losing_its_worker_is_dead_lettered(job_settings, tenant) -> None:
    """So a crash loop cannot cycle one job forever."""
    submit(job_settings, tenant)

    for _ in range(job_settings.job_max_attempts):
        job = jobstore.claim(job_settings, worker_id="doomed")
        assert job is not None
        with db.transaction(job_settings) as conn:
            conn.execute(
                "UPDATE jobs SET locked_until = now() - interval '1 second' WHERE id = %s",
                (job.id,),
            )
        jobstore.reclaim_expired(job_settings)

    final = jobstore.get(job_settings, job_id=job.id)
    assert final.status == "dead_letter"
    assert final.error_code == "lease_expired"
    assert jobstore.claim(job_settings, worker_id="fresh") is None


# --- Retry policy -------------------------------------------------------------------------


def test_a_permanent_failure_is_not_retried(job_settings, tenant) -> None:
    """Retrying a corrupt file five times burns money five times and delays the real error."""
    submit(job_settings, tenant)
    job = jobstore.claim(job_settings, worker_id="w1")

    outcome = jobstore.fail(
        job_settings,
        job_id=job.id,
        worker_id="w1",
        error_code="corrupt_audio",
        retryable=False,
    )

    assert outcome.status == "failed"
    assert outcome.error_code == "corrupt_audio"
    assert outcome.finished_at is not None
    assert jobstore.claim(job_settings, worker_id="w2") is None


def test_retries_back_off_then_dead_letter(job_settings, tenant) -> None:
    job, _ = submit(job_settings, tenant)
    statuses = []

    for _ in range(job_settings.job_max_attempts):
        claimed = jobstore.claim(job_settings, worker_id="w1")
        assert claimed is not None
        outcome = jobstore.fail(
            job_settings,
            job_id=claimed.id,
            worker_id="w1",
            error_code="transcription_failed",
            retryable=True,
        )
        statuses.append(outcome.status)
        with db.transaction(job_settings) as conn:
            conn.execute("UPDATE jobs SET available_at = now() WHERE id = %s", (job.id,))

    assert statuses == ["queued"] * (job_settings.job_max_attempts - 1) + ["dead_letter"]


def test_backoff_grows_and_is_jittered(job_settings) -> None:
    """Jitter matters more than the curve.

    A batch of jobs that fail together — which is what a transient dependency outage looks
    like — would otherwise retry in lockstep forever, hammering the thing already struggling.
    """
    delays = [jobstore._backoff_seconds(attempt, job_settings) for attempt in (1, 2, 3, 4)]
    assert delays == sorted(delays), f"backoff should grow: {delays}"

    repeated = {jobstore._backoff_seconds(3, job_settings) for _ in range(20)}
    assert len(repeated) > 1, "identical delays mean no jitter"

    assert jobstore._backoff_seconds(100, job_settings) <= job_settings.job_backoff_max_seconds


# --- Pagination ---------------------------------------------------------------------------


def test_listing_pages_without_skipping_or_repeating(job_settings, tenant) -> None:
    """Keyset pagination, not OFFSET.

    Offset pagination skips or repeats rows when the underlying set changes between pages,
    which here it constantly does: jobs are created and finishing while someone pages.
    """
    for index in range(7):
        submit(job_settings, tenant, sha=f"{index:064d}")

    seen: list[str] = []
    cursor = None
    for _ in range(10):
        page, cursor = jobstore.list_jobs(
            job_settings, tenant_id=tenant, cursor=cursor, limit=3
        )
        seen.extend(str(job.id) for job in page)
        if cursor is None:
            break

    assert len(seen) == 7
    assert len(set(seen)) == 7, "a job appeared on two pages"


def test_a_malformed_cursor_is_rejected(job_settings, tenant) -> None:
    with pytest.raises(ValueError, match="cursor"):
        jobstore.list_jobs(job_settings, tenant_id=tenant, cursor="not-a-cursor")


# --- Blob storage -------------------------------------------------------------------------


@ffmpeg_required
def test_filesystem_store_is_content_addressed_and_idempotent(job_settings, make_audio) -> None:
    store = blobstore.load_blob_store(job_settings)
    source = make_audio("clip.wav", seconds=1.0)
    digest = audio_module.sha256_file(source)

    uri = store.put(source, digest)
    assert store.put(source, digest) == uri, "storing the same bytes twice must be a no-op"
    assert digest in uri

    with store.materialise(uri) as path:
        assert audio_module.sha256_file(path) == digest

    store.delete(uri)
    store.delete(uri)  # idempotent
    with pytest.raises(blobstore.BlobNotFoundError), store.materialise(uri):
        pass


def test_filesystem_store_refuses_a_uri_outside_its_root(job_settings) -> None:
    """A URI arrives from a database row, and a row is only as trustworthy as its writer."""
    store = blobstore.load_blob_store(job_settings)

    with (
        pytest.raises(blobstore.BlobStoreError, match="outside the blob root"),
        store.materialise("file:///etc/passwd"),
    ):
        pass

    with (
        pytest.raises(blobstore.BlobStoreError, match="file://"),
        store.materialise("s3://bucket/key"),
    ):
        pass


@ffmpeg_required
def test_s3_store_round_trips(make_audio) -> None:
    """The S3 backend, against moto rather than a real account.

    Same key layout as the filesystem backend, which is what makes the two interchangeable.
    """
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")

    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="transcripts")
        store = blobstore.S3BlobStore("transcripts", client=client)

        source = make_audio("clip.wav", seconds=1.0)
        digest = audio_module.sha256_file(source)

        uri = store.put(source, digest)
        assert uri == f"s3://transcripts/audio/{digest[:2]}/{digest}"
        assert store.put(source, digest) == uri

        with store.materialise(uri) as path:
            assert audio_module.sha256_file(path) == digest

        store.delete(uri)
        with pytest.raises(blobstore.BlobNotFoundError), store.materialise(uri):
            pass


def test_s3_store_refuses_another_bucket() -> None:
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")

    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        store = blobstore.S3BlobStore("mine", client=client)
        with (
            pytest.raises(blobstore.BlobStoreError, match="names bucket"),
            store.materialise("s3://not-mine/audio/ab/abc"),
        ):
            pass


# --- Worker -------------------------------------------------------------------------------


@ffmpeg_required
def test_the_worker_transcribes_a_queued_job(job_settings, tenant, make_audio) -> None:
    store = blobstore.load_blob_store(job_settings)
    source = make_audio("job.wav", seconds=1.0)
    digest = audio_module.sha256_file(source)
    info = audio_module.inspect(source)

    job, _ = jobstore.submit(
        job_settings,
        tenant_id=tenant,
        audio_uri=store.put(source, digest),
        audio_sha256=digest,
        model="base",
        duration_seconds=info.duration_seconds,
    )

    worker = Worker(
        job_settings, worker_id="t", transcriber=FakeTranscriber(), blob_store=store
    )
    assert worker.run_once() is True
    assert worker.run_once() is False, "the queue should be empty now"

    finished = jobstore.get(job_settings, job_id=job.id)
    assert finished.status == "succeeded"
    assert finished.progress == 1.0
    assert jobstore.get_result(job_settings, job_id=job.id)["segments"]


@ffmpeg_required
def test_the_worker_marks_missing_audio_permanent(job_settings, tenant) -> None:
    """Audio past its retention clock does not come back on a retry."""
    jobstore.submit(
        job_settings,
        tenant_id=tenant,
        audio_uri=f"file://{job_settings.blob_root}/audio/ff/{'f' * 64}",
        audio_sha256="f" * 64,
        model="base",
    )

    worker = Worker(
        job_settings, worker_id="t", transcriber=FakeTranscriber(), blob_store=None
    )
    worker.run_once()

    jobs, _ = jobstore.list_jobs(job_settings, tenant_id=tenant)
    assert jobs[0].status == "failed"
    assert jobs[0].error_code == "blob_not_found"


# --- The HTTP API -------------------------------------------------------------------------


@pytest.fixture
def jobs_client(job_settings):
    app = create_app(settings=job_settings, transcriber=FakeTranscriber())
    with TestClient(app) as client:
        yield client


@ffmpeg_required
def test_submit_returns_immediately_and_never_blocks_on_compute(
    jobs_client, make_audio
) -> None:
    """The whole point of the tier: accept the bytes, record the intent, return."""
    source = make_audio("upload.wav", seconds=1.0)

    with source.open("rb") as handle:
        response = jobs_client.post("/v1/jobs", files={"file": ("upload.wav", handle)})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "queued"
    assert response.headers["location"] == f"/v1/jobs/{body['id']}"
    # Not advertised yet, so a client cannot be led into polling a URL guaranteed to 409.
    assert body["transcript_url"] is None


@ffmpeg_required
def test_resubmitting_the_same_audio_returns_the_same_job(jobs_client, make_audio) -> None:
    source = make_audio("dupe.wav", seconds=1.0)

    with source.open("rb") as handle:
        first = jobs_client.post("/v1/jobs", files={"file": ("dupe.wav", handle)})
    with source.open("rb") as handle:
        again = jobs_client.post("/v1/jobs", files={"file": ("dupe.wav", handle)})

    assert first.status_code == 201
    # 200 rather than 201: nothing was created this time, and saying so lets a client tell
    # "your retry was absorbed" from "a second job is now running".
    assert again.status_code == 200
    assert again.json()["id"] == first.json()["id"]


@ffmpeg_required
def test_the_transcript_is_409_until_the_job_succeeds(
    jobs_client, job_settings, make_audio
) -> None:
    """409, not 404: the resource is real, the request is premature.

    A client can tell "keep polling" from "wrong id", which it could not if both were 404.
    """
    source = make_audio("pending.wav", seconds=1.0)
    with source.open("rb") as handle:
        job_id = jobs_client.post("/v1/jobs", files={"file": ("pending.wav", handle)}).json()[
            "id"
        ]

    early = jobs_client.get(f"/v1/jobs/{job_id}/transcript")
    assert early.status_code == 409
    assert early.json()["error"]["code"] == "transcript_not_ready"
    assert early.json()["error"]["status"] == "queued"

    Worker(
        job_settings,
        worker_id="t",
        transcriber=FakeTranscriber(),
        blob_store=blobstore.load_blob_store(job_settings),
    ).run_once()

    assert jobs_client.get(f"/v1/jobs/{job_id}").json()["status"] == "succeeded"
    ready = jobs_client.get(f"/v1/jobs/{job_id}/transcript")
    assert ready.status_code == 200
    assert ready.json()["segments"]

    # Every format the exporters produce is reachable here.
    for fmt, needle in (("srt", "-->"), ("vtt", "WEBVTT"), ("txt", "Hello")):
        rendered = jobs_client.get(f"/v1/jobs/{job_id}/transcript?format={fmt}")
        assert rendered.status_code == 200
        assert needle in rendered.text


def test_one_tenant_cannot_read_another_tenants_job(jobs_client, job_settings) -> None:
    """Scoped in the query, not in a check a caller might forget."""
    # Deliberately not UUID(int=1): that is ANONYMOUS_TENANT, so an unattributed request
    # would legitimately own the job and the test would prove nothing.
    owner = uuid.UUID(int=42)
    job, _ = submit(job_settings, owner)

    assert jobs_client.get(f"/v1/jobs/{job.id}").status_code == 404
    assert (
        jobs_client.get(
            f"/v1/jobs/{job.id}", headers={"X-Tenant-Id": str(owner)}
        ).status_code
        == 200
    )


def test_unknown_and_malformed_ids_are_distinguishable(jobs_client) -> None:
    assert jobs_client.get(f"/v1/jobs/{uuid.uuid4()}").status_code == 404
    # A non-UUID is a validation failure, not a missing resource.
    assert jobs_client.get("/v1/jobs/not-a-uuid").status_code == 422


@ffmpeg_required
def test_deleting_a_job_removes_its_audio(jobs_client, job_settings, make_audio) -> None:
    """Erasure, not archiving: a voice recording is biometric personal data."""
    store = blobstore.load_blob_store(job_settings)
    source = make_audio("erase.wav", seconds=1.0)

    with source.open("rb") as handle:
        job_id = jobs_client.post("/v1/jobs", files={"file": ("erase.wav", handle)}).json()["id"]

    digest = audio_module.sha256_file(source)
    uri = store.uri_for(digest)
    with store.materialise(uri):
        pass  # present before the delete

    assert jobs_client.delete(f"/v1/jobs/{job_id}").status_code == 204
    assert jobs_client.get(f"/v1/jobs/{job_id}").status_code == 404
    assert jobs_client.delete(f"/v1/jobs/{job_id}").status_code == 404

    with pytest.raises(blobstore.BlobNotFoundError), store.materialise(uri):
        pass


@ffmpeg_required
def test_shared_audio_survives_one_jobs_deletion(
    jobs_client, job_settings, make_audio
) -> None:
    """Content addressing means two jobs can legitimately share one blob.

    Deleting the bytes for one would quietly break the other, so the blob only goes when no
    job still references it.
    """
    store = blobstore.load_blob_store(job_settings)
    source = make_audio("shared.wav", seconds=1.0)
    digest = audio_module.sha256_file(source)

    with source.open("rb") as handle:
        first = jobs_client.post("/v1/jobs", files={"file": ("shared.wav", handle)}).json()["id"]
    # Same bytes, different options, therefore a second job over one blob.
    with source.open("rb") as handle:
        second = jobs_client.post(
            "/v1/jobs", files={"file": ("shared.wav", handle)}, data={"diarise": "true"}
        ).json()["id"]

    assert first != second
    assert jobs_client.delete(f"/v1/jobs/{first}").status_code == 204

    with store.materialise(store.uri_for(digest)):
        pass  # still there, because the second job needs it


def test_queue_stats_report_the_autoscaling_signal(jobs_client, job_settings) -> None:
    submit(job_settings, uuid.uuid4())

    body = jobs_client.get("/v1/queue").json()

    assert body["by_status"]["queued"] == 1
    assert body["claimable"] == 1
    assert body["oldest_claimable_age_seconds"] >= 0


def test_the_synchronous_endpoint_still_works_with_the_job_tier_mounted(
    jobs_client, speech_sample
) -> None:
    """The reason jobs live at /v1/jobs rather than taking over /v1/transcriptions.

    A fresh clone must keep transcribing with nothing but ffmpeg and a model.
    """
    with speech_sample.open("rb") as handle:
        response = jobs_client.post("/v1/transcriptions", files={"file": ("a.mp3", handle)})

    assert response.status_code == 200
    assert response.json()["segments"]


def test_the_job_api_is_absent_when_the_tier_is_disabled(settings) -> None:
    """Off by default, and genuinely off — not mounted and returning errors."""
    app = create_app(settings=settings, transcriber=FakeTranscriber())
    with TestClient(app) as client:
        assert client.get("/v1/jobs").status_code == 404
        assert client.get("/healthz").status_code == 200


# --- Checkpointing ------------------------------------------------------------------------


@ffmpeg_required
def test_a_retry_resumes_instead_of_re_transcribing(job_settings, tenant, make_audio) -> None:
    """The payoff of the windowed long-audio path, and the design's own build-order item 5.

    A worker that dies partway through must not start from zero. The first attempt completes
    some windows and then fails; the second must skip exactly those windows and transcribe
    only the rest.

    Counted in engine calls rather than wall clock, because that is the thing being saved:
    inference is the expensive part, and re-reading the WAV is not.
    """
    from transcription import pipeline
    from transcription.checkpoint import JobCheckpointer, plan_fingerprint

    settings = job_settings.model_copy(
        update={"window_seconds": 2.0, "window_search_seconds": 0.5}
    )
    source = make_audio("long.wav", seconds=8.0)

    # Measured, not assumed: windows are cut at the quietest point in their search band, so
    # the count depends on the audio rather than on arithmetic over the duration.
    cold = FakeTranscriber()
    pipeline.transcribe_file(source, transcriber=cold, settings=settings)
    total_windows = len(cold.calls)
    assert total_windows >= 3, "need several windows for this test to mean anything"
    job, _ = submit(job_settings, tenant, sha="1" * 64)
    plan = plan_fingerprint(
        settings=settings, model="base", language=None, word_timestamps=False, diarise=False
    )
    checkpointer = JobCheckpointer(settings, job_id=job.id, plan=plan)

    # First attempt: let two windows through, then fail like a dying worker.
    class DiesAfterTwoWindows(FakeTranscriber):
        def __init__(self) -> None:
            super().__init__()
            self.windows = 0

        def transcribe(self, samples, **kwargs):
            self.windows += 1
            if self.windows > 2:
                raise RuntimeError("worker died")
            return super().transcribe(samples, **kwargs)

    dying = DiesAfterTwoWindows()
    with pytest.raises(RuntimeError, match="worker died"):
        pipeline.transcribe_file(
            source, transcriber=dying, settings=settings, checkpointer=checkpointer
        )

    resumed_state = checkpointer.load()
    assert resumed_state.completed_windows == {0, 1}, "two windows should have checkpointed"
    assert resumed_state.segments, "their segments should have been persisted"

    # Second attempt: the same plan, a healthy engine.
    healthy = FakeTranscriber()
    result = pipeline.transcribe_file(
        source, transcriber=healthy, settings=settings, checkpointer=checkpointer
    )

    assert len(healthy.calls) == total_windows - 2, (
        f"a cold run is {total_windows} windows and two were checkpointed, so the retry "
        f"should do {total_windows - 2}; it did {len(healthy.calls)}"
    )
    # Ids stay contiguous across the seam, which is what makes resuming concatenation.
    assert [segment.id for segment in result.segments] == list(range(len(result.segments)))


@ffmpeg_required
def test_checkpoints_are_discarded_when_the_plan_changes(
    job_settings, tenant, make_audio
) -> None:
    """Window boundaries are content-dependent, so they only survive an unchanged plan.

    Change the window size and window 7 is a different stretch of audio. Stitching the old
    checkpoints in would produce a transcript wrong in a way nothing downstream could detect,
    so they go.
    """
    from transcription import pipeline
    from transcription.checkpoint import JobCheckpointer, plan_fingerprint

    source = make_audio("replan.wav", seconds=6.0)
    job, _ = submit(job_settings, tenant, sha="2" * 64)

    def checkpointer_for(window_seconds: float):
        settings = job_settings.model_copy(
            update={"window_seconds": window_seconds, "window_search_seconds": 0.5}
        )
        plan = plan_fingerprint(
            settings=settings,
            model="base",
            language=None,
            word_timestamps=False,
            diarise=False,
        )
        return settings, JobCheckpointer(job_settings, job_id=job.id, plan=plan)

    settings_a, checkpointer_a = checkpointer_for(2.0)
    pipeline.transcribe_file(
        source, transcriber=FakeTranscriber(), settings=settings_a, checkpointer=checkpointer_a
    )
    assert not checkpointer_a.load().is_empty

    # Same job, different windowing. The stored checkpoints no longer describe this plan.
    _, checkpointer_b = checkpointer_for(5.0)
    assert checkpointer_b.load().is_empty, "stale-plan checkpoints should be discarded"
    assert checkpointer_a.load().is_empty, "and actually deleted, not just ignored"


@ffmpeg_required
def test_checkpoints_are_cleared_once_the_job_succeeds(
    job_settings, tenant, make_audio
) -> None:
    """They are scaffolding. Leaving them behind would grow unboundedly."""
    store = blobstore.load_blob_store(job_settings)
    source = make_audio("tidy.wav", seconds=1.0)
    digest = audio_module.sha256_file(source)

    job, _ = jobstore.submit(
        job_settings,
        tenant_id=tenant,
        audio_uri=store.put(source, digest),
        audio_sha256=digest,
        model="base",
    )

    Worker(
        job_settings, worker_id="t", transcriber=FakeTranscriber(), blob_store=store
    ).run_once()

    assert jobstore.get(job_settings, job_id=job.id).status == "succeeded"
    with db.transaction(job_settings) as conn:
        remaining = conn.execute(
            "SELECT count(*) FROM job_checkpoints WHERE job_id = %s", (job.id,)
        ).fetchone()[0]
    assert remaining == 0
