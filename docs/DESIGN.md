# Part 2 — Production System Design

> **Nothing in this document is implemented.** It describes the architecture this pipeline would
> need to run as a real service, and it is deliberately design rather than code. Where the
> repository already contains a seam the production design would plug into, I point at the file.
> The boundary is discussed in
> [README.md](../README.md#what-is-deliberately-not-built).

The single idea the rest of this document follows from:

> **Uploading is fast and I/O-bound. Transcribing is slow and CPU-bound. Treating them as one
> operation is the root cause of almost every failure mode below.**

So the production shape is an asynchronous job system: accept the bytes, durably record the
intent, acknowledge immediately, and transcribe on a pool sized to compute rather than to
request volume.

```
                                                      ┌──────────────┐
  client ──── presigned PUT ──────────────────────────▶│   object     │
     │                                                 │   storage    │
     │  POST /v1/transcriptions {key, options}          │   (S3)       │
     ▼                                                 └──────┬───────┘
┌──────────────┐  INSERT job   ┌───────────────┐              │ audio
│  API         │──────────────▶│  Postgres     │              │
│  (stateless, │               │  job state    │◀─────┐       │
│   N pods)    │──enqueue─────▶├───────────────┤      │       │
└──────────────┘               │  queue        │ state│       ▼
     ▲                         │  (SQS/Redis)  │ update  ┌──────────────┐
     │  GET /v1/transcriptions/{id}            │      │  │ Worker pool  │
     │                         └───────┬───────┘      └──│ (autoscaled) │
     └── 202 / 200 / webhook           │ lease           │ ffmpeg +     │
                                       └────────────────▶│ whisper      │
                                                         └──────────────┘
                                                            │
                                      transcript JSON ──────┘
```

---

## How would you handle concurrent uploads?

**Separate the two concurrency limits, because they are different numbers.** A pod can
accept hundreds of simultaneous uploads, since each is mostly waiting on a socket. The same
machine can run about as many transcriptions as it has cores. A synchronous design forces one
number on both and gets both wrong.

**Ingest.** Clients `PUT` straight to object storage with a presigned URL, then call the API
with the resulting key. The API never proxies the bytes, so a hundred simultaneous 200 MB
uploads cost the API tier almost nothing and no request is bounded by an upload's duration.
For small clips, keep the current multipart endpoint for convenience.

**Admission control.** Workers consume from a queue, so an upload spike becomes queue depth
rather than thrashing. On top of that:

- per-tenant rate limits and concurrent-job quotas, so one customer cannot monopolise capacity;
- `429` with `Retry-After` once a tenant's quota is exhausted;
- queue depth and oldest-message age as the autoscaling signal, with a hard worker ceiling to
  bound spend.

**Fairness.** Two queues split by latency class — short audio and long audio — so a batch of
three-hour recordings cannot starve interactive one-minute clips behind it. Within a class,
round-robin across tenants rather than strict FIFO.

**De-duplication.** Jobs are keyed by the SHA-256 of the audio, which the pipeline already
computes (`audio.sha256_file`, surfaced as `TranscriptionResult.id`). Two concurrent uploads of
identical bytes collapse to one unit of work, and a client retrying an upload it is unsure
about does not pay twice.

**Capacity, worked.** Measured on this repo (Apple Silicon, CPU only, `base`, int8): RTF ≈
0.20 on mixed speech and silence, where the VAD filter skips ahead, and ≈ 0.27 on wall-to-wall
speech. The table below uses 0.20, so one worker handles roughly 5× real time, ≈ 5 audio-hours
per wall-clock hour; at 0.27 every worker count rises by about a third, which is the number to
plan with if the corpus is dense speech. A GPU with `small` is an order of magnitude better.

| Target load | Steady workers (CPU, RTF 0.2) | Notes |
|---|---|---|
| 100 audio-hours/day | ~1 | 4.2 audio-hours/hour |
| 1,000 audio-hours/day | ~9 | plus headroom for diurnal peaks |
| 1,000/day, peak 3× | 20-25 at peak | scale on queue age, floor of 2 |

Sizing on the *mean* is the trap: audio arrives in bursts, so scale on queue age and keep a
warm floor, because a cold worker pays the model-load cost (several seconds) before its first
segment.

**In this repository.** `api.py` bounds in-flight work with an `anyio.Semaphore` and runs
inference in a worker thread via `anyio.to_thread.run_sync`, so the event loop keeps serving
while a transcription runs. Two honest limits on that. It is a single process. And it does not
give parallel inference even within that process: faster-whisper defaults to `num_workers=1`,
so CTranslate2 serializes concurrent `transcribe()` calls on the shared model — measured at
10.8 s for one call and 21.9 s for two. The semaphore is admission control, which is the right
primitive at this scale; real concurrency is the queue above.

---

## How would you store audio and transcripts?

Different data with different access patterns, so: **blobs in object storage, state in a
relational database.**

### Audio — object storage, content-addressed, immutable

```
s3://bucket/audio/{sha256[0:2]}/{sha256}        # original upload, exactly as received
```

Content addressing makes writes idempotent (a re-upload overwrites itself byte-identically),
gives free de-duplication, and removes any need to trust the client's filename — which the
pipeline already refuses to use as a path (`pipeline.safe_filename`).

I would **not** store the normalised 16 kHz WAV. It is derived, it is larger than most
compressed originals (1 hour ≈ 115 MB of PCM against ≈ 28 MB of MP3), and regenerating it costs
an ffmpeg pass measured in seconds. Storing derived artefacts is a recurring bill for something
cheap to recompute.

Blobs do not belong in the database: multi-megabyte rows wreck backup, restore and replication
times, and object storage is roughly two orders of magnitude cheaper per gigabyte.

### Transcripts — canonical artefact in S3, queryable projection in Postgres

```
s3://bucket/transcripts/{sha256}/v1/{model}.json   # the full result document, immutable
```

```sql
CREATE TABLE jobs (
    id              uuid PRIMARY KEY,
    tenant_id       uuid NOT NULL,
    audio_sha256    char(64) NOT NULL,
    status          text NOT NULL,     -- queued|running|succeeded|failed|dead_letter
    attempts        int  NOT NULL DEFAULT 0,
    model           text NOT NULL,
    language        text,
    duration_sec    numeric,
    error_code      text,              -- errors.py `code`, stable across versions
    transcript_uri  text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX ON jobs (tenant_id, audio_sha256, model);  -- idempotency
CREATE INDEX ON jobs (status, created_at);                     -- queue/ops views

CREATE TABLE segments (                -- only if per-segment search is a requirement
    job_id      uuid REFERENCES jobs(id) ON DELETE CASCADE,
    idx         int,
    start_sec   numeric, end_sec numeric,
    text        text,
    confidence  numeric,
    PRIMARY KEY (job_id, idx)
);
CREATE INDEX ON segments USING gin (to_tsvector('english', text));
```

Postgres holds the state machine, because that is what needs transactions, and the fields
people filter on. S3 holds the full document, because it is immutable and read whole. The
`segments` table is optional and I would not create it until someone actually needs
"find me the moment this phrase was said" — at which point full-text search over segments, and
later `pgvector` for semantic search, both land naturally.

**Versioning.** The result already carries `schema_version`, and the S3 key carries the model.
Re-transcribing with a better model writes a *new* object rather than overwriting, which makes
model upgrades, A/B comparisons and reproducing an old answer all possible. Transcripts are
append-only.

**Retention and privacy.** A voice recording is biometric personal data, so this is a design
constraint rather than a footnote:

- SSE-KMS at rest, TLS in transit, per-tenant keys where the contract demands it;
- asymmetric retention — audio expires on a short clock (say 30 days) while transcripts, which
  are far smaller and far less sensitive, persist;
- erasure by deleting the content-addressed object and cascading the rows, with the deletion
  recorded in an audit log;
- the model runs locally, so audio never reaches a third party. That is a genuine reason to
  prefer self-hosted Whisper over a hosted API in a regulated environment, and it is why
  `faster-whisper` is the default here.

---

## How do you retry or recover failed transcriptions?

**The whole policy rests on one distinction: permanent versus transient.** Retrying a corrupt
file five times just burns money five times and delays the error the user needs to see.

That classification already exists in code — `errors.py` gives every error a stable `code` and
a `retryable` flag, and the default is `False`, so a new error type is non-retryable until
someone decides otherwise.

| Failure | Class | Action |
|---|---|---|
| Corrupt / not media / no audio stream (`corrupt_audio`, `no_audio_stream`) | Permanent | Fail immediately, surface the code. No retry. |
| Over the duration or size limit (`audio_too_long`, `upload_too_large`) | Permanent | Reject at submission. |
| ffmpeg decode failure (`normalization_failed`) | Permanent | Fail; alert if the rate moves, since it may mean a new input format in the wild. |
| Worker OOM-killed, spot instance reclaimed, pod evicted | Transient | Lease expires, message redelivered, retry. |
| Model load failure, transient disk or S3 error | Transient | Retry with backoff. |
| Job exceeds its time budget (hung decode, repetition loop) | Transient, capped | Kill at `k × duration`; retry once, then dead-letter. |

**Mechanics.**

*At-least-once delivery with a visibility timeout.* A worker leases a message rather than
removing it; if the worker dies, the lease expires and the job is redelivered. No orphaned
"running forever" jobs.

*Idempotent workers*, which at-least-once delivery requires. Output keys are derived from the
content hash and the job row is updated by primary key, so a redelivered job overwrites its own
result rather than creating a second one.

*Lease heartbeats.* A three-hour job must extend its lease while it works, or it gets
redelivered mid-flight and runs twice forever. Progress reporting already exists in the
pipeline (the `progress` callback in `pipeline.transcribe_file`) and is exactly the hook that
heartbeat would hang off.

*Backoff with jitter*, capped at 3-5 attempts. The attempt counter lives in the `jobs` row, not
only in the queue, so it survives a queue migration and is visible to support staff.

*Dead-letter queue* on exhaustion, with `error_code` persisted on the job. DLQ depth above zero
is an alert, not a dashboard: it means a human needs to decide between replaying and declaring
the input broken.

*Checkpointing* for very long audio: persist each completed window's segments as they finish,
so a retry resumes from the last checkpoint instead of re-transcribing two hours of audio. This
is the direct payoff of the windowed long-audio path described in the README — the windows are
already the natural checkpoint boundary.

**Observability**, because none of the above is operable blind: structured logs keyed by
`job_id` and `audio_sha256`; metrics for queue depth and age, RTF distribution, failure rate
broken down by `error_code`, and attempt counts. The two alerts worth paging on are DLQ depth
and p95 RTF regression — the latter is how a model or dependency change that quietly doubles
cost gets caught.

---

## How would you expose this as an API?

An **asynchronous, versioned, resource-oriented HTTP API**, keeping the synchronous endpoint as
a documented convenience for short clips.

```http
POST /v1/transcriptions
Authorization: Bearer <token>
Idempotency-Key: 7c9e...            # or omit and let the content hash dedupe
{
  "audio_url": "s3://bucket/audio/ab/abc123...",   # or multipart for small files
  "language": null,
  "word_timestamps": false,
  "callback_url": "https://customer.example/hooks/transcription"
}

201 Created
Location: /v1/transcriptions/job_01HX...
{ "id": "job_01HX...", "status": "queued", "created_at": "..." }
```

```http
GET /v1/transcriptions/job_01HX...

200 OK
{ "id": "job_01HX...", "status": "running", "progress": 0.42,
  "audio": { "duration_seconds": 3600.0 } }

# once complete, the same resource carries the result, or a link to it:
{ "id": "...", "status": "succeeded", "transcript_url": "/v1/transcriptions/job_01HX.../transcript" }
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/transcriptions` | Submit. Returns `201` + job, never blocks on compute. |
| `GET /v1/transcriptions/{id}` | Status, progress, result pointer. |
| `GET /v1/transcriptions/{id}/transcript?format=json\|srt\|vtt\|txt` | The artefact. Formats already implemented in `exporters.py`. |
| `GET /v1/transcriptions?status=&cursor=` | List, cursor-paginated. |
| `DELETE /v1/transcriptions/{id}` | Erasure, for the privacy obligation above. |
| `POST /v1/transcriptions:sync` | The implemented synchronous path, duration-capped. |
| `GET /healthz`, `/readyz` | Liveness, and readiness gated on the model being loaded. |

**Job states.** `queued → running → succeeded | failed`, with `running → queued` on
redelivery and `failed → dead_letter` on exhaustion. Exposing the state machine — rather than
a boolean "done" — is what lets a client build a sane UI.

**Design points worth stating explicitly:**

- **Completion notification**: webhook with an HMAC-signed body and its own retry schedule, so
  clients are not forced to poll; polling stays available because webhooks need a public
  endpoint.
- **Idempotency**: an `Idempotency-Key` header, or the content hash, returns the *existing* job
  rather than creating a duplicate. This is what makes client-side retries safe.
- **Versioning twice over**: `/v1` in the path for breaking API changes, `schema_version` in
  the payload for additive result changes. Both already present.
- **Errors**: stable machine-readable codes (implemented in `errors.py`) rather than prose a
  client would have to string-match. In production I would serve them as RFC 9457
  `application/problem+json`.
- **Auth and limits**: per-tenant API keys or OAuth2 client credentials, scoped tokens,
  per-tenant quotas, `429` with `Retry-After`.
- **Discoverability**: the OpenAPI document FastAPI already generates, published alongside
  webhook payload schemas.
- **Limits in the contract**: max duration, max size and supported formats belong in the schema
  and the docs, not in a surprise `413` — which is the reasoning behind the explicit cap the
  implemented endpoint enforces today.

---

## What I would build first

Were this to become the real service, in order:

1. Move transcription behind a queue with Postgres job state — this alone turns the current
   documented duration limit into a non-issue.
2. Presigned direct-to-S3 upload, and content-hash idempotency end to end.
3. Retry policy, lease heartbeats and a dead-letter queue with alerting.
4. Webhooks, then per-tenant auth and quotas.
5. Window-level checkpointing, so a failure three hours into a four-hour file is cheap.
6. Only then accuracy work: larger models on GPU, speaker diarisation, custom vocabulary.

The ordering is deliberate. Every item above accuracy is about the service being trustworthy;
a more accurate transcript nobody can reliably obtain is worth less than a slightly worse one
that always arrives.
