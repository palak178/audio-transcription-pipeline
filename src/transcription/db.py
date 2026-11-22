"""Database connection pooling and migrations.

Two decisions worth stating, because both are departures from what a reader might expect.

**No ORM, and raw SQL.** The schema in docs/DESIGN.md is hand-written SQL, and the single most
important statement in this system — ``SELECT ... FOR UPDATE SKIP LOCKED`` — is precisely the
kind of thing an ORM hides. A reader should be able to see exactly what the queue does. The
cost is that there is no model layer; the benefit is that there is nothing between the code and
the lock semantics it depends on.

**Synchronous psycopg behind a pool, called from the API through a worker thread.** The API is
async, so an async driver looks like the obvious choice. It would mean two concurrency models
in one codebase, because the worker is a plain synchronous loop around CPU-bound inference.
``api.py`` already runs inference via ``anyio.to_thread.run_sync``; database calls go the same
way. One pattern, and queries here are sub-millisecond, so the thread hop is not the cost that
matters next to a transcription measured in minutes.

Migrations are numbered SQL files applied in one transaction each, with a ledger table. Alembic
would be the conventional answer; at this size it is more machinery than the job needs, and a
numbered file you can read top to bottom is easier to audit than a generated script.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg_pool import ConnectionPool

from transcription.config import Settings
from transcription.errors import TranscriptionError

logger = logging.getLogger(__name__)

def _default_migrations_dir() -> Path:
    """Where the migrations are, whether running from a checkout or an installed wheel.

    Both locations are real. The repository root is where the convention puts them and where
    they can be read without unpacking anything; the packaged copy is what lets an installed
    distribution migrate its own database. Checking the package first means a wheel is
    self-sufficient, and falling back to the root means a checkout works before any build.

    Getting this wrong is not theoretical: the container shipped with no migrations at all,
    because the Dockerfile copies `src/` and nothing else, and the service exited on startup.
    """
    packaged = Path(__file__).resolve().parent / "migrations"
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[2] / "migrations"


MIGRATIONS_DIR = _default_migrations_dir()


class DatabaseError(TranscriptionError):
    """The database was unreachable or rejected a statement.

    Transient by default, which is the right default for a network dependency: a worker that
    cannot reach Postgres for a moment should retry, not fail the job permanently.
    """

    code = "database_error"
    http_status = 503
    retryable = True


_pools: dict[str, ConnectionPool] = {}
_pools_lock = threading.Lock()


def get_pool(settings: Settings) -> ConnectionPool:
    """A process-wide pool per connection string.

    Pools are expensive to build and must not be created per request. Keyed by URL so a test
    pointing at a second database does not evict the first.
    """
    key = settings.database_url
    with _pools_lock:
        pool = _pools.get(key)
        if pool is not None:
            return pool
        try:
            pool = ConnectionPool(
                settings.database_url,
                min_size=settings.database_pool_min,
                max_size=settings.database_pool_max,
                # Fail fast rather than hanging a request behind an exhausted pool.
                timeout=10.0,
                open=True,
                kwargs={"autocommit": False},
            )
        except Exception as exc:
            raise DatabaseError(f"Could not connect to the database: {exc}") from exc
        _pools[key] = pool
        return pool


def close_pools() -> None:
    """Shut every pool down. For process teardown and between test sessions."""
    with _pools_lock:
        for pool in _pools.values():
            try:
                pool.close()
            except Exception:  # pragma: no cover - teardown is best-effort
                logger.debug("pool close failed", exc_info=True)
        _pools.clear()


@contextmanager
def transaction(settings: Settings) -> Iterator[psycopg.Connection]:
    """Run a block in one transaction, committing on success and rolling back on error.

    Every mutation in :mod:`transcription.jobstore` goes through here, which is what makes
    "claim the job and record that it was claimed" a single atomic act rather than two steps
    that can disagree.
    """
    pool = get_pool(settings)
    try:
        # psycopg's connection context manager already commits on clean exit and rolls back on
        # exception. Opening the transaction explicitly is for the reader, not the driver: every
        # mutation in jobstore.py depends on knowing exactly where the boundary is.
        with pool.connection() as conn, conn.transaction():
            yield conn
    except psycopg.Error as exc:
        raise DatabaseError(f"Database operation failed: {exc}") from exc


def migrate(settings: Settings, *, migrations_dir: Path | None = None) -> list[str]:
    """Apply any migrations this database has not seen. Returns the names applied.

    Safe to run on every process start, and that is the intent: a deployment should not need a
    separate migration step to be correct. Each file runs inside its own transaction, so a
    failure leaves the database at the last complete migration rather than halfway through one.
    """
    # Explicit argument, then configuration, then the discovered default.
    configured = getattr(settings, "migrations_dir", None)
    directory = migrations_dir or (Path(configured) if configured else _default_migrations_dir())
    if not directory.is_dir():
        raise DatabaseError(f"No migrations directory at {directory}")

    files = sorted(directory.glob("*.sql"))
    if not files:
        return []

    pool = get_pool(settings)
    applied: list[str] = []

    try:
        with pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        name        text PRIMARY KEY,
                        applied_at  timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            already = {
                row[0] for row in conn.execute("SELECT name FROM schema_migrations").fetchall()
            }

            for path in files:
                if path.name in already:
                    continue
                logger.info("applying migration %s", path.name)
                with conn.transaction():
                    conn.execute(path.read_text())
                    conn.execute(
                        "INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,)
                    )
                applied.append(path.name)
    except psycopg.Error as exc:
        raise DatabaseError(f"Migration failed: {exc}") from exc

    return applied
