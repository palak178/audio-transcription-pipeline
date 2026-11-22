"""Where audio bytes live.

The design assumes S3. Making that *mandatory* would cost this project the property it is
most distinctive for: it runs fully offline, with no account, no credentials and no network.
So storage sits behind a protocol with a **local filesystem implementation as the default**,
and S3 becomes one more backend rather than a prerequisite.

That is the same bet :class:`transcription.transcriber.Transcriber` and
:class:`transcription.diarisation.Diariser` already make, for the same two reasons: the test
suite swaps in something cheap, and a new backend is one class rather than an edit to every
caller.

Layout is content-addressed and identical in both backends, which is what makes them
interchangeable::

    {root}/audio/{sha256[0:2]}/{sha256}      # filesystem
    s3://{bucket}/audio/{sha256[0:2]}/{sha256}

Content addressing is not decoration. It makes writes idempotent — re-uploading the same audio
overwrites itself byte-identically — it gives de-duplication for free, and it means the
client's filename is never part of a path.

The fan-out by the first two hex characters exists because some filesystems degrade badly with
hundreds of thousands of entries in one directory, and it costs nothing.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from transcription.config import Settings
from transcription.errors import TranscriptionError

_AUDIO_PREFIX = "audio"


class BlobStoreError(TranscriptionError):
    """Storage was unreachable or rejected the operation. Transient by default."""

    code = "blob_store_error"
    http_status = 503
    retryable = True


class BlobNotFoundError(BlobStoreError):
    """The audio a job refers to is gone.

    Permanent, not transient: the bytes do not come back on a retry. This is what a job
    pointing at audio that has passed its retention clock looks like.
    """

    code = "blob_not_found"
    http_status = 404
    retryable = False


class BlobStore(Protocol):
    """Somewhere to put audio and get it back."""

    @property
    def name(self) -> str:
        """Identifier recorded in logs and job provenance."""
        ...

    def put(self, source: Path, sha256: str) -> str:
        """Store ``source`` under its content hash and return a URI for it.

        Idempotent by construction: storing the same bytes twice is a no-op, because the
        destination is derived from the hash rather than from a name or a counter.
        """
        ...

    @contextmanager
    def materialise(self, uri: str) -> Iterator[Path]:
        """Yield a local path for ``uri``, for the duration of the block.

        A path rather than a file object, because ffmpeg needs something it can open by name.
        The filesystem backend yields the stored file directly, with no copy; a remote backend
        downloads to a temporary file and cleans up on exit. Callers must not write to the
        path they are given.

        Raises:
            BlobNotFoundError: The URI does not resolve to stored bytes.
        """
        ...

    def delete(self, uri: str) -> None:
        """Remove the stored bytes. Idempotent — deleting what is not there is not an error.

        This is the mechanism behind the erasure obligation in the design: a voice recording
        is biometric personal data, so being able to actually delete it is a requirement
        rather than a nicety.
        """
        ...

    def uri_for(self, sha256: str) -> str:
        """The URI these bytes would be stored at, without storing them."""
        ...


class FileSystemBlobStore:
    """Content-addressed storage on a local directory. The default.

    Appropriate well past the point people assume. It is the right backend for a single-node
    deployment, for development, for an air-gapped install, and for the test suite. It stops
    being right when more than one machine needs to read the same audio, which is the point at
    which the S3 backend earns its credentials — not before.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, sha256: str) -> Path:
        _validate_sha256(sha256)
        return self._root / _AUDIO_PREFIX / sha256[:2] / sha256

    def put(self, source: Path, sha256: str) -> str:
        destination = self._path_for(sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            # Same hash means same bytes, so there is nothing to write. This is the
            # de-duplication the content-addressed layout buys.
            return self.uri_for(sha256)

        # Write to a sibling temporary name and rename, so a crash mid-copy cannot leave a
        # truncated file sitting at a path that claims to hold those exact bytes. rename
        # within a directory is atomic on POSIX.
        staging = destination.with_name(f".{destination.name}.partial")
        try:
            shutil.copyfile(source, staging)
            staging.replace(destination)
        except OSError as exc:
            staging.unlink(missing_ok=True)
            raise BlobStoreError(f"Could not store audio {sha256[:12]}: {exc}") from exc

        return self.uri_for(sha256)

    @contextmanager
    def materialise(self, uri: str) -> Iterator[Path]:
        path = self._resolve(uri)
        if not path.is_file():
            raise BlobNotFoundError(f"No stored audio at {uri}", uri=uri)
        # Already local, so hand back the real path. No copy, which matters: the alternative
        # is duplicating a 200 MB file on every job for no reason.
        yield path

    def delete(self, uri: str) -> None:
        path = self._resolve(uri)
        path.unlink(missing_ok=True)

    def uri_for(self, sha256: str) -> str:
        return self._path_for(sha256).as_uri()

    def _resolve(self, uri: str) -> Path:
        """Turn a stored URI back into a path, refusing anything outside the root.

        The containment check is not theatre. A URI reaches here from a database row, and a
        row is only as trustworthy as whatever wrote it; a job whose `audio_uri` said
        `file:///etc/passwd` should fail rather than be read.
        """
        if not uri.startswith("file://"):
            raise BlobStoreError(
                f"This store handles file:// URIs, got {uri.split(':', 1)[0]}://", uri=uri
            )

        from urllib.parse import unquote, urlparse

        path = Path(unquote(urlparse(uri).path))
        try:
            resolved = path.resolve()
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise BlobStoreError(f"Refusing a URI outside the blob root: {uri}") from exc
        return resolved


class S3BlobStore:
    """Content-addressed storage on S3, or anything that speaks its API.

    Same key layout as the filesystem backend, so the two are genuinely interchangeable and a
    deployment can move between them without rewriting URIs.

    ``endpoint_url`` is what makes this work against MinIO, Ceph or Cloudflare R2 as well as
    AWS. That is deliberate: tying the only networked backend to one vendor would undo the
    point of having the protocol.

    ``boto3`` is an optional dependency, imported lazily, so the default install stays small
    and fully offline. Install with ``pip install -e ".[s3]"``.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region: str | None = None,
        endpoint_url: str | None = None,
        sse: str | None = None,
        sse_kms_key_id: str | None = None,
        client: object | None = None,
    ) -> None:
        if not bucket:
            raise BlobStoreError("The S3 blob store needs a bucket name.")
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._sse = sse
        self._sse_kms_key_id = sse_kms_key_id
        self._region = region
        self._endpoint_url = endpoint_url
        # Injectable so tests can hand in a moto-backed client, which is what lets this class
        # be covered without an AWS account.
        self._injected_client = client
        self._cached_client: object | None = None

    @property
    def name(self) -> str:
        return "s3"

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def _client(self):
        if self._injected_client is not None:
            return self._injected_client
        if self._cached_client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise BlobStoreError(
                    "The S3 blob store needs boto3. Install with `pip install -e \".[s3]\"`, "
                    "or set TRANSCRIBE_BLOB_STORE=filesystem, which needs no account.",
                ) from exc
            self._cached_client = boto3.client(
                "s3", region_name=self._region, endpoint_url=self._endpoint_url
            )
        return self._cached_client

    def _key_for(self, sha256: str) -> str:
        _validate_sha256(sha256)
        key = f"{_AUDIO_PREFIX}/{sha256[:2]}/{sha256}"
        return f"{self._prefix}/{key}" if self._prefix else key

    def uri_for(self, sha256: str) -> str:
        return f"s3://{self._bucket}/{self._key_for(sha256)}"

    def put(self, source: Path, sha256: str) -> str:
        key = self._key_for(sha256)

        # Same hash means same bytes, so an existing object is already correct. Skipping the
        # upload is the de-duplication the layout buys, and on S3 it also saves the transfer.
        if self._head(key) is not None:
            return self.uri_for(sha256)

        extra: dict[str, str] = {}
        if self._sse:
            extra["ServerSideEncryption"] = self._sse
        if self._sse_kms_key_id:
            extra["SSEKMSKeyId"] = self._sse_kms_key_id

        try:
            self._client.upload_file(  # type: ignore[attr-defined]
                str(source), self._bucket, key, ExtraArgs=extra or None
            )
        except Exception as exc:
            raise BlobStoreError(f"Could not store audio {sha256[:12]}: {exc}") from exc

        return self.uri_for(sha256)

    @contextmanager
    def materialise(self, uri: str) -> Iterator[Path]:
        import tempfile

        key = self._key_from_uri(uri)
        # ffmpeg needs a path, so the object has to land on local disk. Scoped to the block
        # and deleted on exit, including on failure — a worker chewing through long audio
        # cannot be allowed to leak 200 MB per job.
        with tempfile.TemporaryDirectory(prefix="blob-") as tmp_dir:
            destination = Path(tmp_dir) / "audio"
            try:
                self._client.download_file(self._bucket, key, str(destination))  # type: ignore[attr-defined]
            except Exception as exc:
                if self._is_missing(exc):
                    raise BlobNotFoundError(f"No stored audio at {uri}", uri=uri) from exc
                raise BlobStoreError(f"Could not read {uri}: {exc}") from exc
            yield destination

    def delete(self, uri: str) -> None:
        key = self._key_from_uri(uri)
        try:
            # delete_object succeeds on a missing key, which is the idempotency the protocol
            # asks for.
            self._client.delete_object(Bucket=self._bucket, Key=key)  # type: ignore[attr-defined]
        except Exception as exc:
            raise BlobStoreError(f"Could not delete {uri}: {exc}") from exc

    def _head(self, key: str) -> dict | None:
        try:
            return self._client.head_object(Bucket=self._bucket, Key=key)  # type: ignore[attr-defined]
        except Exception as exc:
            if self._is_missing(exc):
                return None
            raise BlobStoreError(f"Could not reach s3://{self._bucket}/{key}: {exc}") from exc

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        """Whether a boto exception means "not there" rather than "could not ask".

        Worth being careful about: treating a credentials failure as a missing object would
        turn a fixable deployment problem into a permanent job failure.
        """
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            error = response.get("Error", {})
            if str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}:
                return True
        return type(exc).__name__ in {"NoSuchKey", "ClientError404"}

    def _key_from_uri(self, uri: str) -> str:
        if not uri.startswith("s3://"):
            raise BlobStoreError(
                f"This store handles s3:// URIs, got {uri.split(':', 1)[0]}://", uri=uri
            )
        remainder = uri[len("s3://") :]
        bucket, _, key = remainder.partition("/")
        if not key:
            raise BlobStoreError(f"No object key in {uri}", uri=uri)
        if bucket != self._bucket:
            # A row pointing at someone else's bucket is a configuration error or worse.
            # Refusing beats quietly reading from it.
            raise BlobStoreError(
                f"URI names bucket {bucket!r}, this store is configured for "
                f"{self._bucket!r}",
                uri=uri,
            )
        return key


def load_blob_store(settings: Settings) -> BlobStore:
    """Construct the configured backend.

    Defaults to the filesystem, so a fresh clone works with no account and no credentials.
    """
    backend = settings.blob_store.strip().lower()
    if backend in {"", "filesystem", "file", "local"}:
        return FileSystemBlobStore(settings.blob_root)
    if backend in {"s3", "minio"}:
        return S3BlobStore(
            settings.s3_bucket,
            prefix=settings.s3_prefix,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            sse=settings.s3_sse,
            sse_kms_key_id=settings.s3_sse_kms_key_id,
        )
    raise BlobStoreError(f"Unknown blob store {settings.blob_store!r}", backend=backend)


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise BlobStoreError(f"Not a SHA-256 hex digest: {value!r}")
