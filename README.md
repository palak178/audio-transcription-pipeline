# Audio Transcription Service

An audio transcription pipeline that converts speech in any ffmpeg-readable audio file into
text, with a timestamp for every segment — available as a CLI and as an HTTP API.

Formats are detected from content rather than the filename, long recordings are transcribed in
silence-aligned windows so peak memory tracks the window size rather than the length of the
audio, and results come back as JSON, SRT, WebVTT or plain text. It runs fully offline on CPU
with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — no API key, and the audio
never leaves the host.

**Scope.** Part 1 — the pipeline — is implemented and tested. Part 2 — the production
architecture it would need at scale, covering concurrent uploads, durable storage, retry and
recovery, and an asynchronous job API — is design only, written up in
[DESIGN.md](docs/DESIGN.md) rather than built.
[What is deliberately not built](#what-is-deliberately-not-built) says exactly where that line
falls and why.

## Quick start

Needs **Python 3.11+** and **ffmpeg** — ffmpeg is the entire format-handling layer here, not an
optional extra.

```bash
brew install ffmpeg                    # macOS
# sudo apt-get install -y ffmpeg       # Debian / Ubuntu
```

```bash
make setup      # virtualenv + install (needs uv — plain-pip path below)
make warm       # downloads the model once (74 MB) — do it now, not mid-demo
make demo       # transcribes the committed sample, prints JSON and SRT
make test       # 130 tests, ~3 s, no model weights needed
```

<details>
<summary>No <code>uv</code>? Three lines with plain <code>venv</code> and <code>pip</code></summary>

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/transcribe samples/speech_en.mp3 --format srt
```
</details>

What that looks like:

```
$ transcribe samples/speech_en.mp3 --format srt
  2 segments | 7.2s audio | 0.7s compute | RTF 0.09 | language en (100%)

1
00:00:00,000 --> 00:00:03,600
quick brown fox jumps over the lazy dog.

2
00:00:03,600 --> 00:00:07,200
This recording is a test fixture for the audio transcription service.
```

> That transcript is pasted from a real run, dropped leading "The" included — both `base` and
> `small` miss it on this fixture. Every sample output and schema example in this repository
> comes from actual output rather than being written by hand, so what you see is what you get.

> Windows, troubleshooting, Docker, and a guided tour of every feature:
> **[docs/RUNNING.md](docs/RUNNING.md)**

---

## Where each requirement is answered

| Requirement | Implemented in | Covered by |
|---|---|---|
| Accepts an audio file (e.g. WAV or MP3) | `cli.py`, `api.py`, `audio.inspect` | `test_audio.py`, `test_api.py` |
| Transcribes spoken language into text | `transcriber.py` | `test_integration.py` |
| Returns the transcription with timestamps per segment | `models.Segment`, `exporters.py` | `test_exporters.py`, `test_integration.py` |
| How do you handle different audio formats? | `audio.py` — [below](#how-do-you-handle-different-audio-formats) | `test_audio.py` (format matrix) |
| How do you deal with long audio files? | `audio.iter_wav_windows`, `pipeline.py` — [below](#how-do-you-deal-with-long-audio-files) | `test_audio.py`, `test_pipeline.py`, `test_integration.py` |
| Concurrent uploads | *Design* — [DESIGN.md](docs/DESIGN.md#how-would-you-handle-concurrent-uploads) (plus an in-process semaphore in `api.py`) | `test_api.py` |
| Storing audio and transcripts | *Design* — [DESIGN.md](docs/DESIGN.md#how-would-you-store-audio-and-transcripts) | — |
| Retry / recovery | *Design* — [DESIGN.md](docs/DESIGN.md#how-do-you-retry-or-recover-failed-transcriptions) (classification exists in `errors.py`) | — |
| Exposing this as an API | *Design* — [DESIGN.md](docs/DESIGN.md#how-would-you-expose-this-as-an-api) (plus the implemented sync API) | `test_api.py` |

---

## Architecture

One library, two thin entry points, no background processing.

```
  CLI (typer) ──┐
                ├──► pipeline.transcribe_file()
  HTTP (FastAPI)┘           │
            ┌───────────┬───┴────────┬────────────┐
            ▼           ▼            ▼            ▼
         audio.py   transcriber.py  models.py  exporters.py
       ffprobe +    faster-whisper  pydantic   json / srt
       ffmpeg +     (int8, VAD)     versioned  vtt / txt
       windowing                    schema
```

The pipeline is five stages:

1. **Inspect** — `ffprobe` reports the real container, codec, duration, sample rate, channels.
2. **Normalise** — one `ffmpeg` call to 16 kHz mono PCM s16le WAV.
3. **Window** — read the WAV in silence-aligned windows, bounding memory on long audio.
4. **Transcribe** — `faster-whisper` per window, segments consumed from a lazy iterator.
5. **Assemble & export** — rebase timestamps, then render JSON / SRT / VTT / text.

---

## Part 1 — The pipeline

### Accepting an audio file

Two interfaces, because they suit different jobs.

```bash
transcribe samples/speech_en.mp3                     # JSON to stdout
transcribe recording.m4a --format all --out out/     # all four formats to files
transcribe interview.wav --word-timestamps           # add per-word timings
transcribe meeting.mp3 --language en --model small   # pin language, bigger model
transcribe inspect mystery.bin                       # what is this file, really?
```

```bash
make serve
# then open http://localhost:8000/docs and click "Try it out" — no curl needed
curl -F file=@samples/speech_en.mp3 localhost:8000/v1/transcriptions | jq
curl -F file=@clip.m4a -F response_format=srt localhost:8000/v1/transcriptions
```

The service documents itself: **<http://localhost:8000/docs>** is an interactive Swagger page
where you can upload a file and see the response, with every field described, a full example
result, and each error code listed. `/` redirects there, and `/redoc` is the same spec in a
reference layout.

The CLI is the primary entry point for long files — no request timeout, progress on stderr so
`transcribe x.mp3 > out.json` stays clean. The API suits short interactive clips; see
[the synchronous API decision](#5-a-synchronous-api-with-a-documented-limit).

Uploads are copied to disk a megabyte at a time rather than held in memory, and the byte limit
is checked twice: against `Content-Length` in middleware *before* the body is read, and again
while copying. The second check matters because of an honest limit worth naming — FastAPI
resolves `File(...)` by awaiting `request.form()`, which receives and parses the whole
multipart body before the handler runs, so a client that sends no `Content-Length` still has
its bytes spooled by Starlette (to disk past 1 MB, not memory) before anything can refuse it.
Closing that completely means a streaming multipart parser, which is more machinery than this
endpoint warrants. The client's filename is never used to build a path — only to label the
result, after `pipeline.safe_filename` strips directory components and control characters.

### Transcribing speech to text

[**faster-whisper**](https://github.com/SYSTRAN/faster-whisper), a CTranslate2 reimplementation
of OpenAI's Whisper. The upstream project reports roughly 4× the throughput of the reference
PyTorch implementation on CPU at equal accuracy — their benchmark, not one I reproduced; the
numbers I did measure are [below](#measured-performance). It runs int8-quantised and fully
offline with no API key.

Running locally is also a privacy decision: a voice recording is biometric personal data, and
with a local model the audio never leaves the host. In a regulated environment that alone can
decide this choice.

The engine sits behind a `Transcriber` protocol (`transcriber.py`). That exists for two
reasons: the test suite swaps in a fake, so the fast suite runs in ~3 s with no model weights;
and adding a hosted backend (Deepgram, OpenAI) means writing one class, not editing the
pipeline.

### Timestamps per segment

Every segment carries `start`, `end`, `duration`, `text`, plus `confidence`, `avg_logprob` and
`no_speech_prob`:

```json
{
  "id": 0, "start": 0.0, "end": 3.6, "duration": 3.6,
  "text": "quick brown fox jumps over the lazy dog.",
  "no_speech_prob": 0.0044, "avg_logprob": -0.2576, "confidence": 0.7729
},
{
  "id": 1, "start": 3.6, "end": 7.2, "duration": 3.6,
  "text": "This recording is a test fixture for the audio transcription service.",
  "no_speech_prob": 0.0044, "avg_logprob": -0.2576, "confidence": 0.7729
}
```

Two details worth knowing. Whisper's segment timestamps are derived from decoder token
positions and drift by a few hundred milliseconds; `--word-timestamps` runs a DTW alignment
over the model's cross-attention, which tightens both word *and* segment edges for about
10-25% more compute. It is off by default because segment timestamps are what most
consumers need, so the extra cost should be opt-in.

`confidence` is `exp(avg_logprob)` — the geometric mean of the per-token probabilities,
restated as a readable 0-1 value. It is deliberately **not** billed as a calibrated
probability that the text is right: it is monotone in the model's own certainty, so it is
useful for *ranking* segments and routing the weakest to human review, not for absolute
thresholding. One caveat that falls out of how Whisper works: faster-whisper reports a single
`avg_logprob` per 30-second decode window, so adjacent segments often share a value — in the
example above both segments score 0.7729. For actual hallucination detection the better
signal is a segment carrying text *and* a high `no_speech_prob`.

### How do you handle different audio formats?

**Probe the content, then normalise everything to one canonical format.**

The file extension is a hint supplied by the client and is frequently wrong, so nothing
dispatches on it. `ffprobe` reports what the bytes actually are, and then a single `ffmpeg`
call converts to **16 kHz mono PCM s16le WAV**:

```python
ffmpeg -nostdin -i <input> -vn -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le -f wav <output>
#              │           │        │       │        │
#              │           │        │       │        └─ resample 8 kHz telephony or 48 kHz studio
#              │           │        │       └────────── downmix any channel layout to mono
#              │           │        └────────────────── first audio stream only
#              │           └─────────────────────────── drop video
#              └─────────────────────────────────────── never block on stdin
```

16 kHz mono is the only rate Whisper's feature extractor is defined for, and faster-whisper's
own loader resamples to it too — so this is the same conversion moved into a step I own and
test, not an extra one. It is not free, and I would rather name the cost than call it nothing:
a full decode pass, plus a PCM WAV on temporary disk at roughly 115 MB per audio hour (see
[decision #2](#2-normalise-once-up-front)). What it buys is that **no code after this point
knows that codecs exist**. Supporting a new input format becomes "can ffmpeg read it?" rather
than a change to this repository.

Verified end to end, all producing an identical transcript:

| Input | Probed as | Result |
|---|---|---|
| `speech_en.mp3` | mp3, 22 kHz mono | ✅ |
| `speech.wav` | pcm_s16le, 16 kHz | ✅ |
| `speech.flac` | flac, 44.1 kHz | ✅ |
| `speech.m4a` | aac, 44.1 kHz | ✅ |
| `speech.ogg` | opus, 48 kHz | ✅ |
| `speech_8k_telephony.wav` | 8 kHz mono | ✅ upsampled |
| `speech_48k_stereo.wav` | 48 kHz stereo | ✅ downmixed |
| `video_with_audio.mp4` | aac in MP4 | ✅ audio extracted |
| `lying_extension.mp3` | **flac** (not mp3) | ✅ content wins over extension |
| `no_audio_track.mp4` | video only | ❌ `415 no_audio_stream` |
| `not_audio.mp3` | not media | ❌ `415 corrupt_audio` |
| `truncated.mp3` | header fragment | ❌ `415 corrupt_audio` |
| `empty.mp3` | 0 bytes | ❌ `400 empty_audio` |

Reproduce with `make fixtures` — one small real recording is committed and every other fixture
is derived from it by `scripts/make_fixtures.sh`.

### How do you deal with long audio files?

**Transcribe in silence-aligned windows, and let ffmpeg and ffprobe bound the work before any
decoding happens.** This is the part of the build where I had to change my mind, so the
reasoning is worth setting out.

My initial design had no chunking at all, on the grounds that faster-whisper already slides a
30-second window across a file of any length — so custom chunking looked like re-solving a
solved problem while adding seam artefacts and timestamp drift. Then I measured peak memory
against audio duration:

| Audio duration | Peak RSS (single pass) |
|---|---|
| 7 s | 522 MB |
| 58 s | 708 MB |
| 181 s | 819 MB |
| 362 s | 1093 MB |
| 724 s | 1634 MB |

That is linear at roughly **1.45 MB per second of audio** — the engine materialises the whole
waveform and its mel features up front. Extrapolated, a four-hour recording needs about
**21 GB** and gets OOM-killed. My default limit at the time was four hours, so the premise was
comfortable and wrong.

So audio is now read in windows (`audio.iter_wav_windows`), which caps peak memory at the
window size no matter how long the file is:

| Window | Peak RSS on the same 724 s file |
|---|---|
| 60 s | 733 MB |
| 300 s (default) | 976 MB |
| 900 s (exceeds the file, so effectively single-pass) | 1660 MB |

The end-to-end check: **a one-hour recording (3612 s) transcribes in 999 MB of peak memory**,
producing 865 segments. That is the same memory as the twelve-minute file above, against the
~21 GB the single-pass path would have needed — memory is now a function of the window, not of
the recording.

Reproduce it yourself with the commands in
[RUNNING.md Step 10](docs/RUNNING.md#step-10-prove-the-long-audio-handling-works) — one
`TRANSCRIBE_WINDOW_SECONDS` override and `/usr/bin/time`. The absolute megabytes move with the
machine and the BLAS build; the shape should not.

One thing I deliberately am *not* claiming: that windowing is faster. Across repeated runs on
a 363 s file the single-pass path (a 900 s window) came in at 72 s, 72 s, 184 s and 192 s of
wall clock, against a steady 73-78 s at the 300 s default. Peak memory was stable every time;
wall clock at the large window was not. That bimodality is what memory pressure tends to look
like, but four samples and a plausible story are not a result, so the honest summary is that
windowing buys a hard memory bound and costs nothing measurable in throughput.

**Windows are cut at the quietest point** within the final 15 s of each window, not at a fixed
offset. That one choice removes most of the complexity usually associated with chunking: a cut
in silence splits no word, so there is **nothing to overlap, de-duplicate or re-align**, and
making timestamps absolute is a plain shift by the window offset — exact, not approximate.
Windows tile the audio with no gaps and no overlaps (`test_windows_tile_the_audio_exactly`),
and segment ids are renumbered contiguously across the whole transcript.

Audio shorter than one window yields exactly one window, so **short files take the same code
path with no seams and no special-casing**.

The rest of the long-audio handling:

- **Duration is checked twice.** Once from ffprobe metadata *before* any decoding — which is
  what stops a small compressed file that expands into hours of audio from being decoded at
  all — and once against the normalised WAV, which is authoritative for containers that
  declare no duration. ffmpeg is additionally given a hard `-t` bound so it cannot write more
  than the limit.
- **Segments stream.** `transcribe()` returns a lazy iterator, wrapped in another lazy
  iterator; nothing calls `list()` on it until the final consumer.
- **Progress is reported per segment**, because a caller waiting ten minutes needs to tell
  "working" from "hung".
- **Language is detected once** and pinned for later windows. Re-detecting per window lets a
  window that happens to be mostly silence come back as a different language and produce
  nonsense for that stretch.
- **Known worst case**: audio with no pause at all in the search band is cut at its quietest
  moment, which may clip one word per window. Documented rather than hidden, and tested
  (`test_windowing_handles_audio_with_no_silence`).

What remains is wall-clock time, which is a scheduling problem rather than a pipeline problem:
an hour of audio is ~16 minutes of CPU at the RTF 0.27 measured on wall-to-wall speech, no
matter how neatly memory is bounded. That is what [DESIGN.md](docs/DESIGN.md) addresses.

---

## Output contract

The JSON is the contract for downstream consumers — designed to be depended on:

```json
{
  "schema_version": "1.0",
  "id": "33b72396207f5be6...",           // SHA-256 of the source bytes
  "created_at": "2026-10-24T16:55:06.273840Z",
  "language": "en", "language_probability": 0.9973,
  "audio":  { "filename": "speech_en.mp3", "codec": "mp3", "duration_seconds": 7.245,
              "sample_rate": 22050, "channels": 1, "size_bytes": 44111, "sha256": "33b7..." },
  "model":  { "name": "base", "device": "cpu", "compute_type": "int8",
              "beam_size": 5, "vad_filter": true, "word_timestamps": false },
  "segments": [ /* ... */ ],
  "text": "quick brown fox jumps over the lazy dog. This recording is a test fixture ...",
  "processing_seconds": 0.686,
  "real_time_factor": 0.0947
}
```

- `schema_version` lets the format evolve without breaking consumers.
- `id` is the **content hash**, so the same audio always yields the same identifier. One line
  of code that buys de-duplication, caching and safe retries — and it is the hook the
  production design in docs/DESIGN.md hangs idempotency on.
- `model` is provenance: enough to reproduce the run, and to tell *which* model produced a
  transcript when you later upgrade.
- `real_time_factor` is inference seconds per audio second. It excludes ffmpeg normalisation
  and the one-off model load, so it is comparable across runs.

Over HTTP the same timing also comes back as a header, because `srt`/`vtt`/`txt` have no body
field to carry it and because a caller debugging a *slow rejection* needs the number too:

```
Server-Timing: inference;dur=686.0, total;dur=842.3
```

`inference` is model time (the body's `processing_seconds`); `total` adds ffmpeg decoding and
any wait on the concurrency semaphore. Milliseconds, per the W3C spec — standard rather than an
ad-hoc `X-` header, so it carries both metrics at once and browsers show it in devtools.
Error responses carry `total` only, since no inference ran.

SRT and WebVTT are included because captioning is the overwhelmingly common downstream use of
a timestamped transcript, and both fall out of the same segment list.

---

## Key design decisions

### 1. Probe content, never trust the extension
Dispatching on a client-supplied extension is the most common way this kind of pipeline breaks.
*Trade-off:* an ffprobe subprocess per request (~15 ms). Easily worth it.

### 2. Normalise once, up front
Collapses all format handling into one tested step and keeps codecs out of the rest of the
system. *Trade-off:* a full decode pass before transcription, and temporary disk for the WAV.

### 3. Own the decode step; hand the engine samples, not a path
All file handling lives in `audio.py`; the engine only does inference. This avoids decoding the
same audio twice and decouples us from the engine's audio stack — not hypothetically: *PyAV 19
removed a keyword argument that faster-whisper still passes*, so faster-whisper's own
path-based API is broken on current dependency versions. On the versions installed here
(`av` 19.0.1, `faster-whisper` 1.2.1), `faster_whisper.audio.decode_audio(path)` raises
`TypeError: open() got an unexpected keyword argument 'metadata_errors'`. Because the WAV is
read here instead — `audio.iter_wav_windows` for the pipeline, `audio.read_wav_mono_f32` for
whole-file callers — that breakage cannot reach us, and no version pin was needed.

### 4. Windowed reading for long audio
Covered [above](#how-do-you-deal-with-long-audio-files). *Trade-off:* a possible clipped word
per window on gapless audio, against not being OOM-killed.

### 5. A synchronous API with a documented limit
Transcription takes minutes, so a long blocking HTTP request is the wrong shape. Fixing that
*properly* needs durable job state, retries and a worker tier — all production concerns. Rather than
half-build a job system, the endpoint transcribes inline and **refuses audio over
`max_sync_duration_seconds` (default 600 s) with a `413` that names the limit and points at the
CLI**:

```json
{"error": {"code": "audio_too_long", "message": "Audio is 3612.7s long, which exceeds the 600.0s limit.",
           "duration_seconds": 3612.7, "limit_seconds": 600.0,
           "hint": "This endpoint transcribes synchronously. Use the CLI (`transcribe <file>`) for
                    longer audio, or see docs/DESIGN.md for the asynchronous job API this would
                    become in production."}}
```

An explicit limit is better engineering than a silent gateway timeout, and it keeps the
implemented/designed boundary honest.

### 6. Bound concurrency, and never block the event loop
Inference is CPU-bound C++, so it runs via `anyio.to_thread.run_sync` behind an
`anyio.Semaphore`. The event loop keeps serving health checks and rejecting bad requests while
a transcription runs, and concurrent requests queue rather than fail.

Worth being precise about what that semaphore does **not** buy, because it is the obvious
thing to assume: it is not metering parallel inference. `WhisperModel` defaults to
`num_workers=1`, which is CTranslate2's `inter_threads`, so concurrent `transcribe()` calls on
the shared model serialize — measured at 10.8 s for one call and 21.9 s for two on two
threads. That is close to correct on CPU anyway, since `cpu_threads=0` already lets a single
request use every core; two in parallel would just split them. What a limit above 1 actually
buys is overlap between one request's ffmpeg decode and another's inference, plus a bound on
queue depth and peak memory. Raising `num_workers` and cutting `cpu_threads` per worker would
trade latency for throughput — deliberately not done here, because real concurrency is the
worker pool in docs/DESIGN.md.

### 7. VAD filtering on by default
One flag, two payoffs: silence is skipped, and it suppresses Whisper's well-documented
hallucination loop on long silence (the endlessly repeated "Thank you for watching"). Paired
with `condition_on_previous_text=False` so one bad segment cannot poison everything after it.
Measured: 7.2 s of speech followed by 40 s of silence transcribes at **RTF 0.01** and produces
no spurious segments.

### 8. `base` as the default model
`small` is noticeably more accurate, but it is a 250 MB download against 74 MB. A reviewer who
hits a silent quarter-gigabyte download may never see any output at all, and that costs more
than a point of word error rate. Override with `--model small` or
`TRANSCRIBE_WHISPER_MODEL=small`.

---

## Measured performance

Apple Silicon, CPU only, int8, `vad_filter` on. RTF = inference seconds ÷ audio seconds, so
lower is faster and 0.1 means ten times faster than real time.

| Model | Download | RTF (7.2 s clip) | Notes |
|---|---|---|---|
| `tiny` | 40 MB | 0.05 | used in CI |
| `base` | 74 MB | 0.09 | **default** |
| `small` | 250 MB | 0.27 | best accuracy/cost trade here |

| Workload | Result |
|---|---|
| 362 s file, `base` | 97 segments, RTF 0.20 |
| 724 s file, `base`, 300 s windows | peak RSS 976 MB |
| **3612 s (1 h) file, `base`, default windows** | **865 segments, RTF 0.27, peak RSS 999 MB** |
| 7 s speech + 40 s silence | RTF 0.01, no hallucinated segments |

Longer files show a slightly worse RTF than short clips because these fixtures are wall-to-wall
speech, where a short clip is partly silence and the VAD filter makes silence nearly free. At
RTF 0.27, an hour of audio costs about 16 minutes of CPU — which is precisely why the
production design in [DESIGN.md](docs/DESIGN.md) puts this behind a queue rather than an HTTP
request.

---

## Edge cases handled

| Case | Behaviour |
|---|---|
| Extension lies about the codec | Probed content wins; transcribes correctly |
| 0-byte / truncated / not-media file | `400 empty_audio` / `415 corrupt_audio` |
| Video container, no audio track | `415 no_audio_stream` |
| Video container with audio | Audio extracted and transcribed |
| 8 kHz telephony / 48 kHz / multichannel | Resampled and downmixed |
| Pure silence | Empty transcript, not an error and not a hallucination |
| Non-speech (music, tones) | No segments at all — nothing is invented. Where a segment *does* appear over near-silence, its high `no_speech_prob` is surfaced |
| Long trailing silence | No repetition loop (VAD + no prior-text conditioning) |
| Small file that decodes to hours | Rejected on probed duration, before decoding; ffmpeg `-t` bounded |
| Oversized upload | `413 upload_too_large` — refused on `Content-Length` before the body is read; spooled to disk, never to memory |
| `../../etc/passwd` as a filename | Stripped to a label; never used as a path |
| Container declaring no duration | Re-probed after normalisation |
| Timestamps past one hour | `01:00:00,000` — hours always emitted |
| Zero-length or blank caption cue | Padded / `[inaudible]`, so players don't choke |
| Unicode transcripts | Preserved in all four formats |
| Error details leaking server paths | ffmpeg stderr stays in logs, stripped from HTTP bodies |

---

## Testing

```bash
make test        # 130 tests, ~3 s, no model weights required
make test-all    # adds 10 real-model integration tests
make lint
```

The strategy follows from the `Transcriber` protocol: **inference is the only slow,
non-deterministic part of the system, so that is the only thing faked.** The fast suite runs
the real pipeline, real ffmpeg subprocesses and the real ASGI stack against a `FakeTranscriber`
— so format handling, windowing, stitching, exporting, error mapping and HTTP behaviour are all
genuinely under test, with no model download.

Two tests are worth singling out because they pin down claims that are easy to assert and hard
to keep true: `test_windows_tile_the_audio_exactly` checks that windows cover the audio with no
gaps and no overlaps, which is the invariant that makes rebasing timestamps by a plain offset
*provably* correct rather than approximately correct; and
`test_oversized_upload_is_rejected_before_the_body_is_read` makes any read of the upload fail,
so a clean `413` proves the rejection really did happen in middleware.

The `slow` suite then covers what a fake cannot: a real model, real word alignment, real
behaviour on silence, and a long file crossing real window boundaries.

Audio fixtures are generated by ffmpeg into `tmp_path` at test time, so the suite does not
depend on anyone having run `make fixtures`. The one committed binary is a 43 KB synthesised
recording used by the integration tests.

---

## What is deliberately not built

These are deliberately design rather than code, and DESIGN.md covers each one:

| Not built | Why |
|---|---|
| Job queue, worker tier, Redis/SQS | Production concurrency and recovery — designed, not built |
| Postgres / S3 / durable job state | Audio and transcript storage — designed, not built |
| Retries, leases, dead-letter queue | Designed, not built — though the *classification* that drives the policy is in `errors.py` (`code`, `retryable`) |
| Asynchronous job API, webhooks | Designed, not built — the implemented API is synchronous and capped |
| Auth, quotas, rate limits, metrics | Out of scope here; designed in docs/DESIGN.md |
| Speaker diarisation | Out of scope. `pyannote` adds a gated model and real latency; a shaky implementation would be worse than a clear omission |
| Parallel transcription of windows | Windows contend for the same cores, and CTranslate2 already parallelises a single request across them. Parallelism belongs *across* files, in the worker pool |
| Streaming / live transcription | A genuinely different problem (partial hypotheses, revision) and out of scope |

Two things on that list are partly present, because the pipeline needs them to be correct
rather than to demonstrate anything: the in-process concurrency semaphore, and the content
hash. Both are called out above.

---

## Configuration

Every setting is an environment variable prefixed `TRANSCRIBE_`; see
[`.env.example`](.env.example).

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

### Docker

ffmpeg and the model weights are the two things most likely to make "works on my machine" true
and nowhere else, so both are baked into the image at build time, letting it run without
network access:

```bash
make docker-build
make docker-demo
```

Verified end to end with networking disabled, which is the real proof that nothing is fetched
at runtime:

```bash
docker run --rm --network none -v "$PWD/samples:/data:ro" audio-transcription-service \
  transcribe /data/speech_en.mp3 --format srt
```

The entrypoint is the CLI, since that is what `docker run` is usually wanted for. To serve the
API instead:

```bash
docker run -p 8000:8000 --entrypoint uvicorn audio-transcription-service \
  transcription.api:app --host 0.0.0.0 --port 8000
```

The image is ~1.5 GB with the `tiny` model — `onnxruntime` and `ctranslate2` dominate it, not
the weights. It is already built on `python:3.12-slim`; the reductions still available are a
multi-stage build that discards the pip toolchain, installing no dev extras, and shipping the
weights in a mounted volume or an init container so one image serves every model size.

---

## Project layout

```
src/transcription/
  config.py        settings, every knob an env var
  errors.py        typed errors with stable codes, HTTP status, retryable flag
  audio.py         ffprobe inspection, ffmpeg normalisation, windowed WAV reading
  transcriber.py   Transcriber protocol + faster-whisper backend
  models.py        the output contract (versioned pydantic schema)
  exporters.py     json / srt / vtt / txt
  pipeline.py      orchestration, windowing, stitching, progress
  cli.py           transcribe / inspect / warm
  api.py           POST /v1/transcriptions, GET /healthz
docs/
  DESIGN.md        production architecture (design only, nothing implemented)
  RUNNING.md       step-by-step setup, every command to try, troubleshooting
tests/             130 fast + 10 slow
scripts/           make_fixtures.sh
samples/           one committed 43 KB recording
README.md          this file — architecture and design decisions
```

---

## If I had more time

1. **Move transcription behind a queue** ([DESIGN.md](docs/DESIGN.md)) — this turns the documented
   synchronous duration limit into a non-issue, and is the single highest-value change.
2. **Window-level checkpointing**, so a failure three hours into a four-hour file is cheap.
   The windows are already the natural checkpoint boundary.
3. **A real accuracy benchmark** — WER against a labelled set such as LibriSpeech, per model
   size, so the model default is an evidence-based choice rather than a reasoned one.
4. **Speaker diarisation**, the most commonly requested feature this does not have.
5. **GPU path** — `float16` on CUDA is roughly an order of magnitude faster and changes the
   capacity maths in docs/DESIGN.md considerably.
