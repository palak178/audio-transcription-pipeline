# The Asynchronous Job Tier

Accept the bytes, record the intent durably, acknowledge immediately, and transcribe on a pool
sized to compute rather than to request volume.

This is the production design, as built. [DESIGN.md](DESIGN.md) covers the reasoning and the
parts that are still design; it also marks three places where implementing this proved the
original design wrong.

---

## The asynchronous job tier

Transcription takes minutes, so the synchronous endpoint caps audio at 10 minutes and points at
the CLI. The job tier is the real answer: accept the bytes, record the intent durably,
acknowledge immediately, and transcribe on a pool sized to compute rather than to request
volume.

It is **opt-in**, because the synchronous path needs nothing but ffmpeg and a model, and a
fresh clone must keep working with no database.

The whole stack in one command — Postgres, the API and a worker:

```bash
docker compose up --build
curl -s -F file=@samples/speech_en.mp3 localhost:8000/v1/jobs | jq
```

Or against a local Postgres, if you have one:

```bash
make db-setup        # createdb transcription + transcription_test
make migrate         # apply migrations/*.sql
make serve-jobs      # API with /v1/jobs mounted, on :8000
make work            # in another terminal: a worker that drains the queue
```

```bash
# Submit. Returns 201 immediately — nothing here waits on compute.
curl -s -F file=@recording.m4a localhost:8000/v1/jobs | jq
# { "id": "…", "status": "queued", "progress": 0, "transcript_url": null, … }

curl -s localhost:8000/v1/jobs/<id> | jq '.status, .progress'
curl -s "localhost:8000/v1/jobs/<id>/transcript?format=srt"
curl -s localhost:8000/v1/queue | jq      # depth, and the oldest waiting job
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/jobs` | Submit. `201` and a `Location`; `200` with the existing job if it is a duplicate |
| `GET /v1/jobs/{id}` | Status, progress, attempt count, error |
| `GET /v1/jobs/{id}/transcript?format=&speaker_labels=` | The artefact. `409` until the job succeeds |
| `GET /v1/jobs?status=&cursor=&limit=` | Keyset-paginated list |
| `DELETE /v1/jobs/{id}` | Erasure: row, transcript and audio blob |
| `GET /v1/queue` | Depth by status and the oldest claimable job's age |

### Four decisions worth knowing

**Postgres is the queue, not SQS or Redis.** `SELECT … FOR UPDATE SKIP LOCKED` makes claiming a
job and recording that it was claimed the *same statement*. Two systems make that a dual write
that can disagree — a message delivered whose row says `succeeded`, or a row stuck in `running`
with no message behind it. [DESIGN.md](DESIGN.md) originally specified a broker; building
it showed the broker was buying a problem rather than solving one. At RTF 0.27 a worker finishes
about four audio-hours an hour, so queue throughput was never the constraint.

**Leases, not a reaper process.** A claim is valid until `locked_until`; the worker extends it
by heartbeating off the progress callback the pipeline already had. A worker that is OOM-killed
or evicted stops extending, and any other worker returns the job to the queue. There is no
dedicated reaper whose own death would silently strand every job.

**Backoff is scheduling, not a second queue.** A retryable failure leaves a `queued` row with
`available_at` in the future. One queue, nothing to keep in sync, and the attempt count visible
to support staff in the row rather than buried in a broker.

**Checkpoints make a long retry cheap.** Each window's segments are persisted as it completes,
so a worker lost three hours into a four-hour file re-reads the WAV — seconds — and skips
inference on what it already did. Guarded by a fingerprint of the windowing plan: change the
window size and window 7 is a different stretch of audio, so the checkpoints are discarded
rather than stitched into a transcript that would be wrong undetectably.

### Authentication

Off by default, so `docker compose up` works with no setup — and the process **logs a warning
on every start** while it is off, because an API that looks authenticated and is not is worse
than one that is openly open.

```bash
transcribe apikey create --name "my laptop"   # printed once, never again
transcribe apikey list                        # prefixes and usage, never secrets
transcribe apikey revoke <id>

TRANSCRIBE_AUTH_REQUIRED=true make serve-jobs
curl -H "Authorization: Bearer atk_..." localhost:8000/v1/jobs
```

Three decisions here are deliberate, and each is a place where the obvious choice is wrong:

- **Keys are hashed with SHA-256, not bcrypt or argon2.** A slow KDF exists to make guessing
  low-entropy *passwords* expensive. These are 256 bits from a CSPRNG, so guessing is not the
  threat model, and a slow hash would add latency to every request to defend against an attack
  nobody can mount. It would also break the lookup: per-row salting means finding a key would
  mean scanning every row and hashing against each, where a plain digest is a unique index and
  one probe.
- **`X-Tenant-Id` is ignored once a key is presented.** Honouring both would make the header a
  one-line authorisation bypass — authenticate as yourself, then name someone else's tenant.
- **A revoked key and an unknown key return the same message.** Distinguishing them turns the
  error into an oracle for which keys exist.

Revocation is a timestamp rather than a `DELETE`, so an incident leaves an audit trail.

### Storage

Audio goes to a content-addressed blob store, **filesystem by default**. Requiring an S3 account
would cost the offline, no-credentials property that makes this project distinctive. Both
backends share one key layout, so switching rewrites no URI:

```
{root}/audio/{sha256[0:2]}/{sha256}          # filesystem, the default
s3://{bucket}/audio/{sha256[0:2]}/{sha256}   # pip install -e ".[s3]"
```

The S3 backend is tested against `moto`, so it has real coverage without an account. Point
`TRANSCRIBE_S3_ENDPOINT_URL` at MinIO, Ceph or R2 and it works the same.

---

## Configuration

Every setting is an environment variable prefixed `TRANSCRIBE_`. The full list is in
[CONFIGURATION.md](CONFIGURATION.md); the job tier's own settings are the
`TRANSCRIBE_JOBS_*`, `TRANSCRIBE_DATABASE_*`, `TRANSCRIBE_BLOB_*` and `TRANSCRIBE_S3_*` groups.

---

## What is not built here

Webhooks, per-tenant quotas and rate limits. Authentication exists now, which is the part that
had to come first — there is nothing to meter without it. Presigned direct-to-storage upload is
also still design, and only means anything once the S3 backend is in use.

See [DESIGN.md](DESIGN.md) for the design of each.
