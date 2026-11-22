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

## How would you add speaker diarisation?

**Diarisation runs as a second pass over the same normalised WAV, after transcription, and its
labels are attached to the transcript's existing segments rather than resegmenting them.**
Nothing here is implemented, and one part of it — which backend — is blocked on a licensing
question rather than an engineering one.

The constraint that shapes the rest: the pipeline reads audio in windows to bound memory
(`audio.iter_wav_windows`), so diarisation has to work window by window too. Diarising each
window on its own gives labels that are only locally meaningful. `SPEAKER_00` in window 3 is a
different person from `SPEAKER_00` in window 1, and nothing in a per-window result says so.

### Label continuity: cluster embeddings globally, do not reconcile labels

**Discard the per-window labels. Keep the embeddings, and cluster once over the whole
recording.** Two stages:

*Local, per window.* Segmentation gives speech activity and local turn boundaries. For each
turn, extract one speaker embedding — a 192-dimension ECAPA-TDNN or x-vector. The window's own
speaker ids are then thrown away. They are never used downstream.

*Global, once.* Pool every embedding from every window and cluster them in a single pass:
agglomerative hierarchical clustering on cosine distance, cut at a threshold. The cluster index
is the speaker id for the whole recording.

The alternative is reconciling labels pairwise between adjacent windows, and it is worse for a
structural reason: it has an ordering dependency, so error accumulates. One bad match at window
7 shifts every label after it, and nothing later can detect that it happened. Global clustering
has no ordering at all.

What it costs:

| Cost | Size |
|---|---|
| Global state carried across windows | One 192-float vector per turn ≈ 768 B. A four-hour recording with a turn every 5 s is ~2,900 turns ≈ 2 MB. |
| Clustering compute | O(n²) cosine distances — ~8M on 192 dimensions, sub-second |
| Streaming | Lost. Clustering is a barrier; see the latency section. |
| Accuracy | Short turns give noisy embeddings, and that is where diarisation error comes from |

That first row is the reason this design is compatible with the memory bound at all. The
embeddings are tiny, so holding every one of them for a four-hour file costs less than a
hundredth of the window buffer the pipeline already carries. Windowing bounds the audio;
nothing forces the *derived* state to be bounded too.

**Diarisation uses its own window grid**, fixed and overlapping, not the transcription windows.
A fixed 120 s grid with 15 s of overlap decouples the two passes and, more importantly, stops
turns being fragmented at boundaries: any turn shorter than 15 s is wholly contained in at
least one window, and a turn longer than 15 s has enough speech in either fragment to embed
reliably.

Overlap is a genuine cost, and it is the one place diarisation gives up something the
transcription path deliberately bought. The README's silence-aligned, non-overlapping windows
mean there is "nothing to overlap, de-duplicate or re-align". Diarisation reintroduces exactly
that: the same speech is seen twice and produces two turns. The de-duplication rule is a turn
is attributed to the window whose interior contains its midpoint — deterministic, and no
pairwise matching.

Turns under about 1 s are excluded from clustering entirely and labelled afterwards from their
temporal neighbours. An embedding from 600 ms of speech is close to noise, and letting it vote
drags a cluster. A back-channel "mm-hm" therefore inherits the label of the speech around it,
which is sometimes wrong — and is the right kind of wrong, because the error is local and
cannot propagate.

### Pipeline position: after transcription, sequential, in the same job

| Option | Why not |
|---|---|
| Interleaved — diarise each window as it is transcribed | The continuity problem above, and it welds two independent models into one loop for no gain |
| Parallel — diarise and transcribe at the same time | Both are CPU-bound and contend for the same cores. This is the same reason the README gives for not transcribing windows in parallel, and it applies unchanged. |
| Before transcription | The one real argument is using turn boundaries to cut transcription windows, which is a better cut than silence. Rejected: it couples the passes, and a diarisation failure then costs you the transcript. |

So: transcribe, then diarise, then align. The normalised WAV is what keeps the two stages
together — this document deliberately does not store it, so splitting the stages across workers
means either re-running ffmpeg or reversing that decision.

Transcription first also gives the better failure story. The transcript is the expensive,
primary artefact; if the second stage dies, there is still something to return.

### Alignment: the transcript's segmentation wins

**Speaker labels are attached to Whisper's segments. Segments are never re-cut at turn
boundaries.**

Diarisation and transcription are two independent segmentations of one timeline, produced by
different models with different boundary logic, so they will disagree. The transcript wins
because it is the published contract: `models.Segment` is what consumers depend on, SRT and
WebVTT cues are derived straight from it, ids are contiguous across the whole transcript, and
`test_windows_tile_the_audio_exactly` pins the tiling invariant. An additive feature should not
change the text or the count of a segment.

Assignment is a majority vote by time: for each segment, measure its overlap against every
turn, and take the speaker with the most. `speaker_confidence` is then that winner's share of
the segment's duration. A low value means the segment straddles a turn change. That is the same
spirit as `confidence` today — a ranking signal for routing the worst cases to review, not a
calibrated probability.

**The coupling worth naming: this is only as good as the timestamps are.** The README already
records that Whisper's segment edges come from decoder token positions and drift by a few
hundred milliseconds, while `--word-timestamps` DTW-aligns against the model's cross-attention
and tightens them. A 300 ms drift at a turn change mislabels the words on either side of it. So
diarisation should force `word_timestamps=True`, assign per word, and majority-vote the segment
from its words. That costs the 10-25% the README measures, before any diarisation compute at
all.

What I would not do in version 1: split a segment when a turn boundary falls inside it. It is
tempting and it is more correct. It also changes segment ids, segment text and cue counts for a
feature meant to be purely additive, so it belongs behind a `schema_version` 2.0, not here.

### Gated model access

**`pyannote` is the accurate option and it cannot be the default here.**

The facts. `pyannote/speaker-diarization-3.1`, and the segmentation and embedding checkpoints
it loads, are gated on Hugging Face: a human accepts conditions in a browser, then downloads
with a token. The weights are around 60 MB in total, comparable to `base`, so image size is not
the issue.

Runtime is not the issue either. The Docker image already bakes ffmpeg and model weights in and
is verified with `--network none`; gated weights would be fetched during the build with an
`HF_TOKEN` build secret, and the running container would stay offline and token-free. The
"fully offline, no API key" property survives intact.

What does not survive is the three-command quick start. `make warm` downloads Whisper weights
with no token today. With `pyannote` it needs an accepted licence and a token for every person
who builds the image, which is a real onboarding regression for a repository whose setup is
currently four `make` targets.

And there is a question I will not answer by guessing. The 3.1 pipeline is MIT-licensed, but
the gate exists so the author can see who is using it, and whether baking those weights into an
image you then redistribute is within the terms is a licensing question, not an engineering one.
I would get it answered before shipping rather than assume it is fine.

So: two backends behind a `Diariser` protocol — the same shape and the same two reasons as the
existing `Transcriber` protocol. The test suite swaps in a fake, and a new backend is one class
rather than an edit to the pipeline.

| Backend | Gated | Quality | Note |
|---|---|---|---|
| **Default** — ONNX segmentation + ECAPA embeddings, with the clustering above | No | Good. No neural overlap detection. | `onnxruntime` is already in the image — the README notes it is one of the two things dominating the 1.5 GB |
| Opt-in build flag — `pyannote` 3.1 | Yes | Best available, and it does detect overlap | For deployments that have accepted the licence and want the accuracy |

The ungated backend is the default because "runs fully offline, no API key" is a stated property
of this project, and an accuracy gain does not outweigh quietly breaking it.

### Latency and cost: opt-in per job

**Opt-in, `diarise: false` by default.**

First, honesty about the numbers. I have not measured any of this. Every performance figure in
this repository comes from a real run, so what follows is a planning budget and is flagged as
one. Measuring the real RTF is a prerequisite to building this, not a follow-up to it.

| Stage | RTF (CPU, int8) | Basis |
|---|---|---|
| Whisper today | 0.27 | Measured, wall-to-wall speech |
| + word timestamps, which alignment needs | ~0.30-0.34 | The measured 10-25% |
| + segmentation | ~0.02-0.05 | Estimate. A small net over a sliding window. |
| + embedding extraction | ~0.08-0.20 | Estimate. Dominates, and scales with speech, not file length. |
| + global clustering | ~0 | Sub-second |
| **Total** | **~0.45-0.55** | **Estimate** |

So roughly **1.7-2× the compute of a plain transcript**. An audio-hour goes from about 16
minutes of CPU to about 27-33. Against the capacity table above: 1,000 audio-hours/day needs
~9 steady workers at RTF 0.2, and ~20-25 if every job is diarised.

That is the case for per-job rather than always-on:

- it roughly doubles the fleet for a feature a minority of jobs want;
- the precedent already exists — `word_timestamps` is opt-in for exactly this reason, at a
  fifth of the cost;
- captioning is the overwhelmingly common downstream use, and it does not need speaker labels.

One scheduling consequence. The two latency classes split on audio duration. They should split
on `duration × expected RTF` instead, or a diarised three-minute clip lands in the fast queue
and blows the latency it exists to protect.

And the barrier. Global clustering cannot begin until the last window is embedded, so a diarised
job has no partial speaker labels — progress still reports, but there is no useful intermediate
result. Online diarisation exists and is materially less accurate. At this scale there is no
good answer, so accept the barrier and say so in the docs.

### Overlapping speech and unknown speaker count

**Overlap is detected and flagged, not resolved. Speaker count comes from a distance threshold,
with an optional caller hint.**

Overlap has no good answer here, and the reason is worth stating rather than designing around.
`pyannote` can tell you two people are talking. Whisper cannot help: it transcribes a mixed
waveform into one token stream and does not separate sources. So even with perfect overlap
detection there is one piece of text and no principled way to divide it between two speakers.
Resolving it needs source separation or per-speaker channels, and neither is in scope. The
segment therefore takes the majority speaker, carries `speakers_overlap: true`, and the
documentation says plainly that attribution inside those spans is unreliable.

Unknown count: cut the dendrogram at a cosine threshold rather than at a fixed *k*, so a
monologue stays one speaker and a six-person meeting is not forced into two. The threshold is
the tunable, and it is corpus-dependent — which means it cannot be chosen honestly without
labelled audio this repository does not have. That is why the accuracy benchmark moves ahead of
diarisation in the build order below.

Callers may pass `num_speakers`, or `min_speakers` and `max_speakers`. Given a count, the
dendrogram is cut at *k* instead of at the threshold, and accuracy improves materially. Most
real callers know how many people were in the room, so this is the cheapest accuracy on offer.

Without a hint, the threshold will sometimes split one speaker across a channel or emotion
change, and sometimes merge two similar voices. **Tune it to over-split.** A human can merge
`SPEAKER_03` into `SPEAKER_01` in a review UI; nobody can un-merge two speakers without
re-running the job.

Explicitly not handled:

| Not handled | Why |
|---|---|
| Attributing words inside overlapped speech | Needs source separation. See above — this is a limit, not an omission. |
| Speaker identity across recordings. `SPEAKER_00` in two jobs is unrelated. | Needs an enrolment store of voiceprints. That is biometric data under a much heavier obligation than a transcript, which the retention section above already treats as a design constraint. Not built, and privacy is the main reason rather than effort. |
| Naming speakers — "this is Alice" | Same answer, same store, same obligation |
| Per-channel speakers | Normalisation downmixes to mono (`-ac 1`), which destroys the strongest diarisation signal that exists when it exists: a two-channel interview with one speaker per side could be split almost perfectly on channel energy alone. Recovering it means a second ffmpeg output preserving channels, on the diarisation path only. Worth doing, after version 1. |

### Worker tier, retry and checkpointing

**Not a separate tier. A second stage inside the existing transcription job.** Three reasons:

1. Both stages need the normalised WAV, which this document deliberately does not store. A
   separate tier either re-runs ffmpeg for nothing or reverses that decision.
2. Both stages are CPU-bound and want the same worker shape and the same autoscaling signal.
   Independent scaling is not a problem anyone has here.
3. Splitting doubles the state machine — two queues, two lease timeouts, a join — for a stage
   that is about a third of the job's cost.

A separate tier *would* be right for one case: diarising transcripts that already exist.
"Diarise the 10,000 recordings I already have" is a real ask, and it is possible precisely
because the original audio is stored content-addressed and immutable — a backfill worker
re-normalises from the hash and runs the diarisation stage alone. That is a later job type, not
a reason to split the live path.

Additions to the retry taxonomy:

| Failure | Class | Action |
|---|---|---|
| Diarisation weights missing or unreadable (`diarisation_unavailable`) | Permanent, deployment | The same family as `ffmpeg_not_found` — not the caller's fault. Gate `/readyz` on the diarisation models loading when the feature is enabled, so a misconfigured pod never leases a job rather than failing them one at a time. |
| Worker killed during embedding extraction | Transient | Lease expires, job redelivered, and the retry resumes at the diarisation stage. Transcript windows are already checkpointed; per-window embeddings checkpoint alongside them, at ~768 B per turn. |
| Clustering returns an implausible speaker count | Not a failure at all | Return it. The computation is deterministic, so a retry produces the same answer — this is the third category the permanent/transient split does not cover. Surface it as a speaker-count distribution and alert on the shape, not on the job. |
| Diarisation exhausts its retries after transcription succeeded | Partial success | The job **succeeds**, carrying `diarisation.status = "failed"` and an `error_code`. A transcript without labels is useful, and discarding the expensive stage because the cheap one failed is the wrong trade. Not a dead-letter entry: the DLQ means a human must choose between replaying and declaring the input broken, and here there is nothing to choose. Count it in its own metric. |

Note what that last row does *not* do: it adds no job state. `queued → running → succeeded |
failed` stays exactly as it is, and the per-stage outcome lives in the payload. A new state
would break the client-side state machine that exposing the state machine was meant to enable.

Checkpointing has two boundaries: the stage boundary, and each diarisation window's embeddings.
Global clustering is deliberately not checkpointed — it is seconds of work, and re-running it is
cheaper than persisting a dendrogram.

### Output schema

Additive, so `schema_version` becomes **1.1** and `/v1` does not move. That is exactly the split
the API section below describes. Every new field is optional with a `None` default, so existing
consumers are untouched, and `speaker: None` honestly means "not requested, or requested and
failed" — which `diarisation.status` disambiguates.

```python
class Speaker(BaseModel):
    id: str                        # "SPEAKER_00" — opaque, meaningful only within this result
    total_speech_seconds: float
    turn_count: int

class DiarisationInfo(BaseModel):  # provenance, for the same reason ModelInfo exists
    status: Literal["not_requested", "succeeded", "failed"]
    backend: str | None            # "onnx-ecapa-ahc" | "pyannote-3.1"
    speaker_count: int | None
    speakers_hint: int | None      # what the caller asked for, if anything
    clustering_threshold: float | None
    overlap_seconds: float | None
    processing_seconds: float | None
    error_code: str | None

# added to Segment
    speaker: str | None = None
    speaker_confidence: float | None = None  # the winning speaker's share of this segment
    speakers_overlap: bool | None = None

# added to Word
    speaker: str | None = None

# added to TranscriptionResult
    speakers: list[Speaker] | None = None
    diarisation: DiarisationInfo | None = None
```

`speaker` is a string rather than an int on purpose. `"SPEAKER_00"` reads as an opaque label;
`0` invites a consumer to sort or do arithmetic on something that carries no order at all. It
also leaves room for a real name later without a type change.

**One change this forces elsewhere, and it is a bug if missed.** Idempotency currently keys on
`(tenant_id, audio_sha256, model)`, and transcripts land at
`transcripts/{sha256}/v1/{model}.json`. Diarisation is a second axis over the same audio and
the same model. Without adding it to both, a `diarise: true` request de-duplicates against an
earlier plain run and the caller gets a `200` with no speaker labels and no error — the worst
possible failure, because it looks like success. So the index becomes
`(tenant_id, audio_sha256, model, diarisation_backend)` and the key becomes
`transcripts/{sha256}/v1/{model}__{diarisation_backend|none}.json`. Both variants coexist and
transcripts stay append-only, which is the reasoning the model axis already uses.

Exporters. JSON is the only lossless one, and the caption formats each lose something different:

| Format | How speakers appear | Lossless? |
|---|---|---|
| JSON | `speaker` per segment, a `speakers` roster, a `diarisation` provenance block | Yes — the canonical artefact |
| WebVTT | `<v SPEAKER_00>text</v>`, the format's own voice span | Yes. The only caption format with a real mechanism: conformant players strip or style the tag. |
| SRT | `SPEAKER_00: text`, prefixed into the cue | No. SRT has no speaker field, so the label becomes part of the text and cannot be separated out again. The prefix is the de facto convention, not the spec. |
| TXT | `SPEAKER_00:` heading a block, consecutive same-speaker segments merged | No — timings were already gone. This is the format diarisation improves most: a readable meeting transcript. |

Three details fall out of that:

- **SRT and WebVTT have no speaker field** in the sense a consumer wants — SRT has nothing at
  all, and WebVTT's `<v>` is text markup rather than metadata. Neither round-trips back to
  labelled segments. JSON is the artefact to keep; the captions are renderings of it.
- Output stays **byte-identical to today** whenever no labels are present. That is worth an
  explicit test, because the alternative is silently changing the captions every existing
  consumer already depends on.
- A label spends about 11 characters of a caption line that convention keeps to 37-42.
  `GET /v1/transcriptions/{id}/transcript?speaker_labels=false` suppresses them, for a caller
  who diarised for the JSON and wants clean captions out of the same job.

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
6. A labelled accuracy benchmark — WER per model size, and a diarisation error rate. Nothing
   below this line can be tuned honestly without one; the clustering threshold above is the
   clearest example.
7. Speaker diarisation, as designed above: opt-in per job, the ungated backend by default, and
   gated `pyannote` behind a build flag once the licence question is answered. It needs item 5
   to retry cheaply and item 6 to be tuned at all, which is why it sits here rather than
   earlier.
8. The rest of the accuracy work: larger models on GPU, custom vocabulary.

The ordering is deliberate. Everything above item 6 is about the service being trustworthy;
a more accurate transcript nobody can reliably obtain is worth less than a slightly worse one
that always arrives. Diarisation in particular is easy to ship badly and hard to ship well,
which is the argument for building it behind a benchmark rather than ahead of one.
