-- Job state and the queue, in one table.
--
-- docs/DESIGN.md originally proposed Postgres for state and SQS or Redis for the queue. This
-- schema deliberately does not do that. Two systems means the state machine and the queue
-- position are a dual write that can disagree: a message delivered whose row says
-- `succeeded`, or a row marked `running` with no message behind it. The design already
-- conceded the problem without naming it, by insisting the attempt counter live "in the jobs
-- row, not only in the queue".
--
-- One table with `SELECT ... FOR UPDATE SKIP LOCKED` removes the class of bug entirely:
-- claiming a job and recording that it was claimed are the same statement. The throughput
-- ceiling is irrelevant here — at the measured RTF of 0.27 a worker finishes about four
-- one-hour jobs per hour, so even a thousand workers is a few thousand claims per hour, which
-- is several orders of magnitude below what this pattern sustains.

CREATE TABLE IF NOT EXISTS jobs (
    id                   uuid PRIMARY KEY,
    tenant_id            uuid NOT NULL,

    -- What to transcribe. `audio_uri` is resolved by a BlobStore, so it is deliberately not
    -- constrained to a scheme: `file://` today, `s3://` once that backend exists.
    audio_uri            text        NOT NULL,
    audio_sha256         char(64)    NOT NULL,
    display_name         text,
    duration_seconds     numeric,
    size_bytes           bigint,

    -- Every option that changes the output. All of them are part of the job's identity, which
    -- is the point of `request_fingerprint` below.
    model                text        NOT NULL,
    language             text,
    word_timestamps      boolean     NOT NULL DEFAULT false,
    diarise              boolean     NOT NULL DEFAULT false,
    diarisation_backend  text        NOT NULL DEFAULT 'none',

    -- Identity, for de-duplication. SHA-256 over the audio hash and a canonical encoding of
    -- the options above.
    --
    -- The design keyed idempotency on (tenant_id, audio_sha256, model), which is a bug: a
    -- `diarise: true` request would de-duplicate against an earlier plain run and return a
    -- transcript with no speaker labels and no error — a wrong answer that looks like a
    -- success. Widening the index to every option would work but would need a migration each
    -- time an option is added, so the options are hashed into one column instead.
    request_fingerprint  char(64)    NOT NULL,
    -- An explicit client-supplied key, for callers who want retry safety without relying on
    -- the content hash.
    idempotency_key      text,

    -- State machine. Exposed to clients as-is, because a boolean "done" is not enough to
    -- build a sane UI on.
    status               text        NOT NULL DEFAULT 'queued',
    attempts             integer     NOT NULL DEFAULT 0,
    max_attempts         integer     NOT NULL DEFAULT 5,
    progress             numeric     NOT NULL DEFAULT 0,
    error_code           text,
    error_message        text,

    -- Queue. `available_at` is how backoff is expressed: a failed job is not hidden in a
    -- retry queue, it is a queued row that is not claimable yet. `locked_until` is the lease;
    -- a worker that dies stops extending it and the job becomes claimable again.
    available_at         timestamptz NOT NULL DEFAULT now(),
    locked_until         timestamptz,
    locked_by            text,
    -- Two latency classes so a batch of three-hour recordings cannot starve interactive
    -- clips. Assigned from duration x expected RTF rather than duration alone, since a
    -- diarised short clip costs roughly twice a plain one.
    latency_class        text        NOT NULL DEFAULT 'short',

    transcript_uri       text,

    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now(),
    started_at           timestamptz,
    finished_at          timestamptz,

    CONSTRAINT jobs_status_valid CHECK (
        status IN ('queued', 'running', 'succeeded', 'failed', 'dead_letter')
    ),
    CONSTRAINT jobs_latency_class_valid CHECK (latency_class IN ('short', 'long')),
    CONSTRAINT jobs_progress_range CHECK (progress >= 0 AND progress <= 1),
    CONSTRAINT jobs_attempts_sane CHECK (attempts >= 0 AND max_attempts >= 1),
    -- A lease only means something while the job is running. Letting it linger on a finished
    -- row would make the reclaim query's job harder to reason about.
    CONSTRAINT jobs_lease_only_while_running CHECK (
        status = 'running' OR (locked_until IS NULL AND locked_by IS NULL)
    )
);

-- Idempotency. Two concurrent submissions of identical bytes and identical options collapse
-- to one unit of work, and a client retrying a submission it is unsure about does not pay
-- twice.
CREATE UNIQUE INDEX IF NOT EXISTS jobs_identity
    ON jobs (tenant_id, request_fingerprint);

CREATE UNIQUE INDEX IF NOT EXISTS jobs_idempotency_key
    ON jobs (tenant_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- The claim query's index. Partial, because only queued rows are ever claimable, which keeps
-- it small no matter how much finished history accumulates.
CREATE INDEX IF NOT EXISTS jobs_claimable
    ON jobs (latency_class, available_at, created_at)
    WHERE status = 'queued';

-- The lease-reclaim query's index. Also partial, for the same reason.
CREATE INDEX IF NOT EXISTS jobs_expiring_leases
    ON jobs (locked_until)
    WHERE status = 'running';

-- Operator views: "what is failing", "what is in flight", listing by tenant.
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_tenant_created ON jobs (tenant_id, created_at DESC, id);

-- The transcript itself, kept out of `jobs` on purpose. The queue reads and writes `jobs`
-- constantly; a multi-hundred-kilobyte jsonb column on those rows would mean TOAST traffic on
-- every claim. A separate table keeps the hot row narrow.
--
-- This is also the provisional home for the result. The design puts the canonical artefact in
-- object storage with Postgres holding only a queryable projection, and that is still the
-- right shape; `transcript_uri` is already here for it.
CREATE TABLE IF NOT EXISTS job_results (
    job_id      uuid PRIMARY KEY REFERENCES jobs (id) ON DELETE CASCADE,
    result      jsonb       NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- `updated_at` maintained by the database rather than by every caller, because a column that
-- is only correct when the application remembers to set it is not worth having.
CREATE OR REPLACE FUNCTION jobs_touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS jobs_set_updated_at ON jobs;
CREATE TRIGGER jobs_set_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION jobs_touch_updated_at();
