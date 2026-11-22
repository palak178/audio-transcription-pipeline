# Configuration

Every setting is an environment variable prefixed `TRANSCRIBE_`, and every one has a default
that works. See [`.env.example`](../.env.example) for an annotated copy.

Every setting is an environment variable prefixed `TRANSCRIBE_`; see
[`.env.example`](../.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIBE_WHISPER_MODEL` | `base` | `tiny` … `large-v3` |
| `TRANSCRIBE_DEVICE` / `_COMPUTE_TYPE` | `cpu` / `int8` | `cuda` / `float16` on GPU |
| `TRANSCRIBE_WINDOW_SECONDS` | `300` | Long-audio window; controls peak memory |
| `TRANSCRIBE_WINDOW_SEARCH_SECONDS` | `15` | How far back to hunt for a silent cut |
| `TRANSCRIBE_MAX_DURATION_SECONDS` | `14400` | Hard ceiling for any transcription |
| `TRANSCRIBE_MAX_SYNC_DURATION_SECONDS` | `600` | The synchronous endpoint's stricter ceiling |
| `TRANSCRIBE_MAX_UPLOAD_BYTES` | `209715200` | 200 MB |
| `TRANSCRIBE_MAX_CONCURRENT_TRANSCRIPTIONS` | `2` | In-flight inference limit |
| `TRANSCRIBE_VAD_FILTER` | `true` | Leave on |
| `TRANSCRIBE_DIARISE` | `false` | Label who spoke. Needs a backend; see the row above |
| `TRANSCRIBE_DIARISATION_BACKEND` | `none` | No backend ships yet, so this is the only value |
| `TRANSCRIBE_DIARISATION_WINDOW_SECONDS` | `120` | Diarisation's own window, separate from transcription's |
| `TRANSCRIBE_DIARISATION_OVERLAP_SECONDS` | `15` | Overlap, so no turn is split into two unembeddable halves |
| `TRANSCRIBE_DIARISATION_THRESHOLD` | `0.55` | Cosine distance to stop merging speakers. **An unvalidated placeholder** — picking it honestly needs labelled audio |
| `TRANSCRIBE_DIARISATION_MIN_EMBEDDING_SECONDS` | `1.0` | Shorter turns are labelled from neighbours, not clustered |

The asynchronous job tier is off by default, because the synchronous endpoint needs no database:

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIBE_JOBS_ENABLED` | `false` | Mount `/v1/jobs` and migrate on startup |
| `TRANSCRIBE_DATABASE_URL` | `postgresql:///transcription` | Job store **and** queue |
| `TRANSCRIBE_BLOB_STORE` | `filesystem` | `filesystem` or `s3` |
| `TRANSCRIBE_BLOB_ROOT` | `./var/blobs` | Where the filesystem backend keeps audio |
| `TRANSCRIBE_S3_BUCKET` / `_REGION` / `_ENDPOINT_URL` | — | Only read when the backend is `s3`. The endpoint points at MinIO or R2 |
| `TRANSCRIBE_JOB_LEASE_SECONDS` | `60` | How long a claim survives without a heartbeat |
| `TRANSCRIBE_JOB_HEARTBEAT_SECONDS` | `15` | Must be well under the lease |
| `TRANSCRIBE_JOB_MAX_ATTEMPTS` | `5` | Attempts before dead-lettering |
| `TRANSCRIBE_JOB_BACKOFF_BASE_SECONDS` | `5` | Doubles per attempt, jittered, capped |
| `TRANSCRIBE_LONG_JOB_THRESHOLD_SECONDS` | `600` | Estimated compute above which a job goes to the `long` queue |
