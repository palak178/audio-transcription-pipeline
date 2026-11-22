-- Per-tenant API keys.
--
-- The `jobs` table has had a `tenant_id` since migration 001, and every query is scoped by it.
-- What was missing was anything that decides *which* tenant a request belongs to: until now it
-- came from a header the caller sets, which is a documented default rather than a check.
--
-- Keys are stored as a SHA-256 digest, never in plaintext, so a database dump does not hand
-- over working credentials.
--
-- **SHA-256 rather than bcrypt or argon2, deliberately.** A slow hash exists to make guessing
-- low-entropy human passwords expensive. These keys are 256 bits from a CSPRNG, so guessing is
-- not the threat model, and a slow KDF would add latency to every single request for nothing.
-- It would also make the lookup impossible: a bcrypt digest is salted per row, so finding a key
-- would mean scanning every row and hashing against each. A plain digest is a unique index and
-- an O(1) probe.
--
-- `key_prefix` is the first few characters of the plaintext, stored openly so a key can be
-- identified in a log line or a UI without the secret being recoverable.

CREATE TABLE IF NOT EXISTS api_keys (
    id            uuid PRIMARY KEY,
    tenant_id     uuid        NOT NULL,
    name          text        NOT NULL,
    key_prefix    text        NOT NULL,
    key_hash      char(64)    NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    -- Updated at most once a minute rather than on every request. An unconditional write per
    -- authenticated call would turn a read path into a write path and put row contention on the
    -- busiest key in the system.
    last_used_at  timestamptz,
    -- Revocation is a timestamp, not a DELETE, so an incident has an audit trail: which key,
    -- belonging to whom, was withdrawn and when.
    revoked_at    timestamptz
);

-- The authentication probe. Unique because two keys hashing alike would be a collision, and
-- indexed because this runs on every authenticated request.
CREATE UNIQUE INDEX IF NOT EXISTS api_keys_hash ON api_keys (key_hash);

-- "Show me this tenant's keys", for rotation and for revoking after someone leaves.
CREATE INDEX IF NOT EXISTS api_keys_tenant ON api_keys (tenant_id, created_at DESC);
