# Audio Transcription Service

[![CI](https://github.com/palak178/audio-transcription-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/palak178/audio-transcription-pipeline/actions/workflows/ci.yml)
[![Integration](https://github.com/palak178/audio-transcription-pipeline/actions/workflows/integration.yml/badge.svg)](https://github.com/palak178/audio-transcription-pipeline/actions/workflows/integration.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Speech in any ffmpeg-readable audio file becomes text with a timestamp for every segment.
Available as a CLI, a synchronous HTTP API, and an asynchronous job service with a worker pool.

Formats are detected from content rather than filenames. Long recordings are transcribed in
silence-aligned windows, so peak memory tracks the window rather than the length of the audio.
Results come back as JSON, SRT, WebVTT or plain text. It runs **fully offline on CPU** with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) — no API key, no account, and the
audio never leaves the host.

## Run it

```bash
brew install ffmpeg          # or: sudo apt-get install -y ffmpeg
make setup && make warm      # virtualenv, then the model (74 MB) once
make demo                    # transcribes the committed sample
```

```bash
transcribe recording.m4a --format srt      # any format, any length
make serve                                 # http://localhost:8000/docs
```

The whole service — Postgres, API and a worker pool — in one command:

```bash
docker compose up --build
curl -s -F file=@samples/speech_en.mp3 localhost:8000/v1/jobs | jq
```

Setup detail, Windows, troubleshooting and a guided tour: **[docs/RUNNING.md](docs/RUNNING.md)**

## If you read one thing

**[How long audio is handled](docs/PIPELINE.md#how-do-you-deal-with-long-audio-files).** My
first design had no chunking, on the reasonable grounds that Whisper already slides a 30-second
window across a file of any length. Then I measured peak memory against duration and found it
growing linearly at ~1.45 MB per second of audio — so a four-hour recording needed about 21 GB
and would have been killed, under a four-hour limit I had already shipped.

That section is the measurement, the redesign, and an honest account of what it did and did not
buy. Three more cases where building something proved a design wrong are recorded in
[DESIGN.md](docs/DESIGN.md).

## What it does

```
  CLI ──┐                        ┌─ POST /v1/jobs ─▶ Postgres (state + queue)
        ├─▶ pipeline.transcribe  │                        │ SKIP LOCKED
  HTTP ─┘   inspect ▸ normalise  │                        ▼
            ▸ window ▸ transcribe│                   worker pool
            ▸ assemble ▸ export  │                   ffmpeg + whisper
                                 └─ GET  /v1/jobs/{id}/transcript
```

| Capability | Where | Tested by |
|---|---|---|
| Any audio format, detected by content | `audio.py` | `test_audio.py` |
| Transcription with per-segment timestamps | `transcriber.py`, `models.py` | `test_integration.py` |
| Long audio without unbounded memory | `audio.iter_wav_windows` | `test_audio.py`, `test_pipeline.py` |
| JSON / SRT / WebVTT / text output | `exporters.py` | `test_exporters.py` |
| Job queue, worker tier, leases, retries | `jobstore.py`, `worker.py` | `test_jobs.py` |
| Resume a long job after a crash | `checkpoint.py` | `test_jobs.py` |
| Audio storage: filesystem or S3 | `blobstore.py` | `test_jobs.py` (S3 via `moto`) |
| Per-tenant API keys | `auth.py` | `test_jobs.py` |
| Speaker diarisation, minus the model | `diarisation.py` | `test_diarisation.py` |

**206 fast tests** with no model download, 10 more against a real model, and CI that runs both
plus a real Postgres and a `docker compose` stack end to end.

## Documentation

| Document | What is in it |
|---|---|
| **[docs/PIPELINE.md](docs/PIPELINE.md)** | How the pipeline works: formats, long audio, the output contract, the eight key design decisions, measured performance, edge cases, testing strategy |
| **[docs/JOBS.md](docs/JOBS.md)** | The asynchronous tier as built: the Postgres queue, leases and retries, checkpointing, storage backends, API keys |
| **[docs/DESIGN.md](docs/DESIGN.md)** | The production architecture. Marks each section as built or still design, and records where implementing it corrected the design |
| **[docs/RUNNING.md](docs/RUNNING.md)** | Step-by-step setup, every command worth trying, Docker, troubleshooting |
| **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** | Every environment variable |

---

## What is deliberately not built

These are deliberately design rather than code, and DESIGN.md covers each one:

| Not built | Why |
|---|---|
| Webhooks | Designed, not built. Polling works; a webhook needs its own signed delivery and retry schedule |
| Per-tenant quotas and rate limits | Designed, not built. Authentication now exists to hang them off, which is the part that had to come first |
| Presigned direct-to-storage upload | Designed. Content-hash idempotency and the storage abstraction are built; presigning only means anything for the S3 backend |
| Alerting on DLQ depth and RTF regression | The metrics are exposed at `GET /v1/queue`; wiring them to a pager is deployment, not code |
| Speaker diarisation — **the model backend only** | Everything around the model is built and tested: the overlapping window grid, the global clustering that makes labels mean the same thing across windows, boundary de-duplication, alignment onto segments, the schema and all four exporters (`diarisation.py`). The segmentation and embedding model is not, because `pyannote`'s weights are gated behind a Hugging Face licence and requiring one would break the offline, token-free setup above. Asking for diarisation returns `diarisation.status: failed` with `diarisation_unavailable` — a clear omission rather than a shaky implementation. [DESIGN.md](docs/DESIGN.md#how-would-you-add-speaker-diarisation) has the full design and the part-by-part state |
| Parallel transcription of windows | Windows contend for the same cores, and CTranslate2 already parallelises a single request across them. Parallelism belongs *across* files, in the worker pool |
| Streaming / live transcription | A genuinely different problem (partial hypotheses, revision) and out of scope |

Everything else from the production design **is** built: the job queue and worker tier, the job
state machine, both storage backends, the retry policy with leases and dead-lettering, API keys,
and window-level checkpointing. [DESIGN.md](docs/DESIGN.md) marks each section as code or design,
and records three places where implementing it proved the design wrong rather than quietly
editing the design to match.

---

## If I had more time

1. **A real accuracy benchmark** — WER against a labelled set such as LibriSpeech, per model
   size, so the default model is an evidence-based choice rather than a reasoned one. This is
   first because the next item depends on it.
2. **A speaker diarisation backend.** Everything around the model is built and tested in
   `diarisation.py`; what is missing is the segmentation and embedding model. It sits behind
   item 1 deliberately — its one load-bearing tunable, the clustering threshold, cannot be
   chosen honestly without labelled audio.
3. **Webhooks, then per-tenant quotas and rate limits.** Authentication exists to hang them
   off, which was the part that had to come first.
4. **GPU path** — `float16` on CUDA is roughly an order of magnitude faster and changes the
   capacity arithmetic in [DESIGN.md](docs/DESIGN.md) considerably.
5. **A shorter path to "it works on my machine"** — the image is ~1.5 GB, dominated by
   `onnxruntime` and `ctranslate2` rather than by weights. A multi-stage build and a mounted
   weights volume would cut it substantially.

---

## Licence

[MIT](LICENSE). The model stack is MIT too — faster-whisper, CTranslate2 and the Whisper
weights — so commercial use is unencumbered.
