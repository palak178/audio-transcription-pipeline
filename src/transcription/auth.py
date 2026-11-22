"""Per-tenant API keys.

The job tables have carried a ``tenant_id`` and scoped every query by it from the start. This
is the piece that decides *which* tenant a request is, rather than taking the caller's word for
it.

Three decisions worth stating, because each is a place where the obvious choice is wrong.

**Keys are hashed with SHA-256, not bcrypt or argon2.** A slow KDF exists to make guessing
low-entropy human passwords expensive. A key here is 256 bits from ``secrets.token_urlsafe``,
so guessing is not the threat model, and a slow hash would add latency to every request to
defend against an attack nobody can mount. It would also break the lookup: per-row salting
means finding a key would require scanning every row and hashing against each, where a plain
digest is a unique index and a single probe.

**The tenant header is ignored once a key is presented.** Otherwise ``X-Tenant-Id`` is a
one-line authorisation bypass: authenticate as yourself, then name someone else's tenant. When
authentication is required the header is not read at all.

**Authentication is off by default, and says so loudly at startup.** A fresh clone has to keep
working, and the synchronous endpoint needs no database at all. What it must not do is *look*
authenticated while it is not, so enabling the job tier without ``auth_required`` logs a warning
every time the process starts.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime

from psycopg.rows import dict_row

from transcription import db
from transcription.config import Settings
from transcription.errors import TranscriptionError

logger = logging.getLogger(__name__)

# A recognisable, greppable prefix. Secret scanners key off patterns like this, and so does a
# human staring at a log line wondering what they are looking at.
KEY_PREFIX = "atk_"
# 32 bytes of CSPRNG output. token_urlsafe gives ~43 characters for that.
_KEY_BYTES = 32
_PREFIX_SAMPLE = 12
# How stale last_used_at may get before a request bothers to refresh it.
_LAST_USED_REFRESH_SECONDS = 60


class AuthenticationError(TranscriptionError):
    """No usable credential was presented.

    ``401`` with a ``WWW-Authenticate`` header, which the API layer adds. Deliberately says
    nothing about *why* — unknown key, revoked key and malformed key are one message, so the
    response cannot be used to enumerate which keys exist.
    """

    code = "unauthenticated"
    http_status = 401


@dataclass(frozen=True)
class ApiKey:
    """A key record. Never carries the secret — that exists once, at creation."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    key_prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.strip().encode()).hexdigest()


def generate_key() -> str:
    """A fresh secret. Returned once and never recoverable afterwards."""
    return f"{KEY_PREFIX}{secrets.token_urlsafe(_KEY_BYTES)}"


def create_api_key(
    settings: Settings, *, tenant_id: uuid.UUID, name: str
) -> tuple[str, ApiKey]:
    """Mint a key. Returns ``(plaintext, record)``.

    The plaintext is returned exactly once, here. Only its digest is stored, so a lost key is
    replaced rather than recovered — which is the property that makes a database dump useless
    to whoever takes it.
    """
    plaintext = generate_key()
    key_id = uuid.uuid4()

    with db.transaction(settings) as conn:
        cursor = conn.cursor(row_factory=dict_row)
        cursor.execute(
            """
            INSERT INTO api_keys (id, tenant_id, name, key_prefix, key_hash)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, tenant_id, name, key_prefix, created_at, last_used_at, revoked_at
            """,
            (key_id, tenant_id, name, plaintext[:_PREFIX_SAMPLE], _hash(plaintext)),
        )
        record = ApiKey(**cursor.fetchone())

    logger.info("created API key %s (%s…) for tenant %s", record.id, record.key_prefix, tenant_id)
    return plaintext, record


def verify(settings: Settings, plaintext: str) -> ApiKey:
    """Resolve a key to its record, or raise.

    The lookup is by digest against a unique index, so there is no string comparison in
    application code and therefore nothing to time. A revoked key is rejected with the same
    message as an unknown one: distinguishing them would let someone probe which keys exist.
    """
    if not plaintext or not plaintext.startswith(KEY_PREFIX):
        raise AuthenticationError("Invalid API key.")

    with db.transaction(settings) as conn:
        cursor = conn.cursor(row_factory=dict_row)
        cursor.execute(
            """
            SELECT id, tenant_id, name, key_prefix, created_at, last_used_at, revoked_at
            FROM api_keys WHERE key_hash = %s
            """,
            (_hash(plaintext),),
        )
        row = cursor.fetchone()
        if row is None:
            raise AuthenticationError("Invalid API key.")

        record = ApiKey(**row)
        if not record.is_active:
            raise AuthenticationError("Invalid API key.")

        # Conditional, so an authenticated read stays a read on all but one request a minute.
        cursor.execute(
            """
            UPDATE api_keys SET last_used_at = now()
            WHERE id = %s
              AND (last_used_at IS NULL
                   OR last_used_at < now() - make_interval(secs => %s))
            """,
            (record.id, _LAST_USED_REFRESH_SECONDS),
        )

    return record


def revoke(settings: Settings, *, key_id: uuid.UUID) -> bool:
    """Withdraw a key. Idempotent; ``False`` if there was no such active key."""
    with db.transaction(settings) as conn:
        cursor = conn.execute(
            "UPDATE api_keys SET revoked_at = now() WHERE id = %s AND revoked_at IS NULL",
            (key_id,),
        )
        return cursor.rowcount > 0


def list_api_keys(settings: Settings, *, tenant_id: uuid.UUID | None = None) -> list[ApiKey]:
    """Key records, newest first. Never includes secrets."""
    with db.transaction(settings) as conn:
        cursor = conn.cursor(row_factory=dict_row)
        if tenant_id is None:
            cursor.execute(
                """
                SELECT id, tenant_id, name, key_prefix, created_at, last_used_at, revoked_at
                FROM api_keys ORDER BY created_at DESC
                """
            )
        else:
            cursor.execute(
                """
                SELECT id, tenant_id, name, key_prefix, created_at, last_used_at, revoked_at
                FROM api_keys WHERE tenant_id = %s ORDER BY created_at DESC
                """,
                (tenant_id,),
            )
        return [ApiKey(**row) for row in cursor.fetchall()]


def bearer_token(authorization: str | None) -> str | None:
    """Pull the credential out of an ``Authorization`` header, or ``None``.

    Tolerant about the scheme's case, because clients vary, and strict about the shape.
    """
    if not authorization:
        return None
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credential.strip():
        return None
    return credential.strip()


ANONYMOUS_TENANT = uuid.UUID("00000000-0000-0000-0000-000000000001")


def resolve_tenant(
    settings: Settings,
    *,
    authorization: str | None,
    tenant_header: str | None,
) -> uuid.UUID:
    """Decide which tenant this request belongs to.

    Precedence matters, and it is the security-relevant part of this module:

    1. A presented key wins, and ``X-Tenant-Id`` is then **ignored entirely**. Honouring both
       would mean anyone with a valid key could act as any tenant by naming it.
    2. With no key and ``auth_required`` set, the request is refused.
    3. With no key and authentication off, the header is a development convenience and falls
       back to a fixed anonymous tenant.
    """
    token = bearer_token(authorization)
    if token is not None:
        return verify(settings, token).tenant_id

    if settings.auth_required:
        raise AuthenticationError("This endpoint requires an API key.")

    if tenant_header:
        try:
            return uuid.UUID(tenant_header)
        except ValueError as exc:
            raise TranscriptionError(
                f"X-Tenant-Id must be a UUID, got {tenant_header!r}"
            ) from exc

    return ANONYMOUS_TENANT


def warn_if_unauthenticated(settings: Settings) -> None:
    """Say so, every start, when the job tier is open.

    An API that looks authenticated and is not is worse than one that is openly open. This is
    the line that keeps the second thing from quietly becoming the first.
    """
    if settings.jobs_enabled and not settings.auth_required:
        logger.warning(
            "the job API is mounted WITHOUT authentication: any caller may set X-Tenant-Id "
            "and act as any tenant. Set TRANSCRIBE_AUTH_REQUIRED=true and mint a key with "
            "`transcribe apikey create` before exposing this."
        )
