"""Speaker diarisation: who spoke when.

Designed in docs/DESIGN.md#how-would-you-add-speaker-diarisation. This module implements that
design with one piece deliberately missing.

**What is here.** The diarisation window grid, the global clustering that gives speaker labels
meaning across windows, the de-duplication rule for overlapping windows, short-turn handling,
and the alignment of speaker turns onto transcript segments. All of it real, all of it tested.

**What is not here.** The model backend — segmentation and speaker embeddings. It sits behind
the :class:`Diariser` protocol and no implementation ships yet, for the reason set out in the
design: the accurate option (``pyannote``) has weights gated behind a Hugging Face licence
acceptance, and making that a requirement would break the offline, token-free, three-command
setup this project guarantees. :func:`load_diariser` therefore raises
:class:`DiarisationUnavailableError` with a message that says so, rather than silently
producing labels nobody should trust.

The ordering is the point. Everything above is the part that is easy to get subtly wrong and
hard to notice: label continuity, boundary de-duplication, alignment against timestamps that
drift. Getting it right and tested *first*, with a fake backend, means adding a real model is
one class rather than a rewrite — the same bet the ``Transcriber`` protocol already makes.

The one idea the rest of this module follows from:

    Per-window speaker labels are meaningless across windows. ``SPEAKER_00`` in window 3 is a
    different person from ``SPEAKER_00`` in window 1. So the per-window labels are discarded,
    and the per-turn *embeddings* are clustered once over the whole recording.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from transcription import audio as audio_module
from transcription.config import Settings
from transcription.errors import DiarisationFailedError, DiarisationUnavailableError
from transcription.models import DiarisationInfo, Segment, Speaker

if TYPE_CHECKING:
    import numpy as np

# Labels are generated, never supplied, so the format is ours to fix. Zero-padded to two
# digits because a recording with more than a hundred distinct speakers is not a meeting.
_LABEL_FORMAT = "SPEAKER_{:02d}"


@dataclass(frozen=True)
class RawTurn:
    """A span of speech from one backend window, before any global reconciliation.

    Times are **window-relative**, matching the ``Transcriber`` convention — the backend sees
    samples and knows nothing about where in the recording they came from.

    ``embedding`` is a 1-D float32 vector of whatever dimension the backend produces (192 for
    ECAPA-TDNN, 256 for some x-vector variants); nothing here depends on the size. ``None``
    means the backend judged the span too short to embed usefully, which is a legitimate
    answer and is handled by :func:`_label_short_turns` rather than treated as an error.
    """

    start: float
    end: float
    embedding: np.ndarray | None = None
    overlap: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SpeakerTurn:
    """A span of speech attributed to a speaker, on the recording's own timeline."""

    start: float
    end: float
    speaker: str
    overlap: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SpeakerHint:
    """What the caller claims to know about the speaker count.

    Worth accepting because it is the cheapest accuracy available: most real callers know how
    many people were in the room, and a known count turns an unsupervised threshold decision
    into a constrained one.
    """

    exact: int | None = None
    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("exact", self.exact),
            ("minimum", self.minimum),
            ("maximum", self.maximum),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} speaker count must be at least 1, got {value}")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"minimum speakers ({self.minimum}) exceeds maximum ({self.maximum})")
        if self.exact is not None and (self.minimum is not None or self.maximum is not None):
            raise ValueError("Pass an exact speaker count or a range, not both")

    @property
    def is_empty(self) -> bool:
        return self.exact is None and self.minimum is None and self.maximum is None


@dataclass(frozen=True)
class Diarisation:
    """The diarisation stage's output: labelled turns, a roster, and provenance."""

    turns: list[SpeakerTurn]
    speakers: list[Speaker]
    info: DiarisationInfo


class Diariser(Protocol):
    """A speaker segmentation and embedding backend.

    The same shape as :class:`transcription.transcriber.Transcriber`, for the same two
    reasons: tests swap in a fake so the fast suite needs no weights, and a new backend is one
    class rather than an edit to the pipeline.

    Note what this interface does *not* include: clustering. A backend reports local turns and
    their embeddings; deciding which turns belong to the same person over the whole recording
    is this module's job, not the backend's. That split is deliberate — it is what lets the
    globally-correct part be tested without a model, and it keeps a backend from having to
    know about windowing at all.
    """

    @property
    def name(self) -> str:
        """Identifier recorded in the result's provenance block."""
        ...

    def warm(self) -> None:
        """Make the backend ready, paying the weight-loading cost deliberately."""
        ...

    def segment(self, samples: np.ndarray) -> list[RawTurn]:
        """Find speech turns in mono 16 kHz float32 samples and embed each one.

        Takes samples rather than a path for the same reason the transcriber does: all file
        handling stays in :mod:`transcription.audio`. Returned times are relative to the start
        of ``samples``.
        """
        ...


def load_diariser(settings: Settings) -> Diariser:
    """Construct the configured diarisation backend.

    Raises :class:`DiarisationUnavailableError` for ``backend="none"``, which is the default
    and currently the only value, because no backend ships with this repository. That is a
    deliberate omission rather than an oversight — see the module docstring and
    docs/DESIGN.md. The error is explicit so the failure is visible, instead of a transcript
    quietly coming back with every ``speaker`` field null.
    """
    backend = settings.diarisation_backend.strip().lower()
    if backend in {"", "none"}:
        raise DiarisationUnavailableError(
            "No speaker diarisation backend is installed. The clustering and alignment are "
            "implemented; the embedding model is not. See docs/DESIGN.md "
            "(#how-would-you-add-speaker-diarisation) for the design and why the accurate "
            "backend is gated.",
            backend=backend or "none",
        )
    raise DiarisationUnavailableError(
        f"Unknown diarisation backend {settings.diarisation_backend!r}.",
        backend=settings.diarisation_backend,
    )


def window_authority(
    offsets: list[float],
    index: int,
    *,
    overlap_seconds: float,
    audio_end: float,
) -> tuple[float, float]:
    """The time range window ``index`` is responsible for.

    Overlapping windows see the same speech twice, so the same turn is reported twice and one
    copy has to go. The rule: **a turn belongs to the window whose authority range contains
    its midpoint.** Authority ranges tile the recording with no gaps and no overlaps, so every
    turn survives exactly once, and the decision needs no comparison between windows — which
    is what keeps it from having the ordering dependency that global clustering was chosen to
    avoid in the first place.

    Each range sits strictly inside its window's coverage, so a turn is only ever kept by a
    window that actually saw all of it.
    """
    half = overlap_seconds / 2.0
    start = 0.0 if index == 0 else offsets[index] + half
    end = audio_end if index == len(offsets) - 1 else offsets[index + 1] + half
    return start, end


def diarise_wav(
    wav_path: Path,
    *,
    diariser: Diariser,
    settings: Settings,
    audio_duration: float,
    hint: SpeakerHint | None = None,
) -> Diarisation:
    """Diarise a normalised WAV end to end.

    Two stages, as designed:

    1. *Local, per window.* The backend finds turns and embeds each one. The window's own
       speaker numbering, if it has any, is never read.
    2. *Global, once.* Every embedding from every window is clustered in a single pass, and
       the cluster is the speaker.

    Why not reconcile labels between adjacent windows instead: that has an ordering
    dependency, so error accumulates. One bad match at window 7 shifts every label after it
    and nothing downstream can tell it happened. Global clustering has no ordering at all.

    Raises:
        DiarisationFailedError: The backend failed. The caller is expected to degrade to an
            unlabelled transcript rather than propagate this.
    """
    hint = hint or SpeakerHint()
    started = time.perf_counter()

    offsets: list[float] = []
    # (window_index, absolute turn) for every turn any window reported, duplicates included.
    # Duplicates are cheap to carry — an embedding is under a kilobyte and the overlap is
    # ~12% of turns — and carrying them means the de-duplication rule needs no lookahead.
    collected: list[tuple[int, RawTurn]] = []

    windows = audio_module.iter_wav_grid(
        wav_path,
        window_seconds=settings.diarisation_window_seconds,
        overlap_seconds=settings.diarisation_overlap_seconds,
    )

    try:
        for index, (offset, samples) in enumerate(windows):
            offsets.append(offset)
            for turn in diariser.segment(samples):
                collected.append(
                    (
                        index,
                        RawTurn(
                            start=turn.start + offset,
                            end=turn.end + offset,
                            embedding=turn.embedding,
                            overlap=turn.overlap,
                        ),
                    )
                )
    except DiarisationFailedError:
        raise
    except Exception as exc:
        raise DiarisationFailedError(f"Diarisation failed: {exc}") from exc

    turns = _deduplicate(
        collected,
        offsets=offsets,
        overlap_seconds=settings.diarisation_overlap_seconds,
        audio_end=audio_duration,
    )

    labelled = _label(
        turns,
        threshold=settings.diarisation_threshold,
        min_embedding_seconds=settings.diarisation_min_embedding_seconds,
        hint=hint,
    )

    speakers = _roster(labelled)
    overlap_seconds = round(sum(turn.duration for turn in labelled if turn.overlap), 3)

    return Diarisation(
        turns=labelled,
        speakers=speakers,
        info=DiarisationInfo(
            status="succeeded",
            backend=diariser.name,
            speaker_count=len(speakers),
            speakers_hint=hint.exact,
            clustering_threshold=settings.diarisation_threshold,
            overlap_seconds=overlap_seconds,
            processing_seconds=round(time.perf_counter() - started, 3),
        ),
    )


def _deduplicate(
    collected: list[tuple[int, RawTurn]],
    *,
    offsets: list[float],
    overlap_seconds: float,
    audio_end: float,
) -> list[RawTurn]:
    """Keep one copy of each turn, using the midpoint-authority rule."""
    if not offsets:
        return []

    kept: list[RawTurn] = []
    for index, turn in collected:
        midpoint = (turn.start + turn.end) / 2.0
        start, end = window_authority(
            offsets, index, overlap_seconds=overlap_seconds, audio_end=audio_end
        )
        if start <= midpoint < end:
            kept.append(turn)

    kept.sort(key=lambda turn: (turn.start, turn.end))
    return kept


def _label(
    turns: list[RawTurn],
    *,
    threshold: float,
    min_embedding_seconds: float,
    hint: SpeakerHint,
) -> list[SpeakerTurn]:
    """Cluster embeddings globally and turn clusters into speaker labels."""
    if not turns:
        return []

    embeddable = [
        index
        for index, turn in enumerate(turns)
        if turn.embedding is not None and turn.duration >= min_embedding_seconds
    ]

    if not embeddable:
        # Nothing long enough to embed, so there is nothing honest to say about who spoke.
        # Returning no turns is the same stance the pipeline already takes on silence: an
        # empty answer, not an error and not an invention.
        return []

    import numpy as np

    matrix = np.vstack([np.asarray(turns[i].embedding, dtype=np.float32) for i in embeddable])
    clusters = _cluster(matrix, threshold=threshold, hint=hint)

    # Number speakers by first appearance rather than by merge order, which is arbitrary. This
    # makes SPEAKER_00 the first voice heard, which is both intuitive and deterministic.
    ordering: dict[int, int] = {}
    for position in sorted(range(len(embeddable)), key=lambda p: turns[embeddable[p]].start):
        cluster = clusters[position]
        if cluster not in ordering:
            ordering[cluster] = len(ordering)

    labels: dict[int, str] = {
        embeddable[position]: _LABEL_FORMAT.format(ordering[cluster])
        for position, cluster in enumerate(clusters)
    }

    _label_short_turns(turns, labels)

    return [
        SpeakerTurn(
            start=round(turn.start, 3),
            end=round(turn.end, 3),
            speaker=labels[index],
            overlap=turn.overlap,
        )
        for index, turn in enumerate(turns)
        if index in labels
    ]


def _label_short_turns(turns: list[RawTurn], labels: dict[int, str]) -> None:
    """Give every un-clustered turn the label of its nearest neighbour in time.

    Turns too short to embed were kept out of the clustering because an embedding from a few
    hundred milliseconds of speech is close to noise, and letting it vote drags a whole
    cluster off. They still need a label, so they inherit one.

    This is wrong sometimes — a back-channel "mm-hm" from the listener gets attributed to
    whoever was talking around it. It is the right kind of wrong: the error is local, bounded
    to one short turn, and cannot propagate the way a corrupted cluster would.
    """
    if not labels:
        return

    clustered = sorted(labels)
    for index in range(len(turns)):
        if index in labels:
            continue
        nearest = min(clustered, key=lambda other: _gap(turns[index], turns[other]))
        labels[index] = labels[nearest]


def _gap(a: RawTurn, b: RawTurn) -> float:
    """Seconds between two turns; zero if they touch or overlap."""
    return max(0.0, a.start - b.end, b.start - a.end)


def _cluster(matrix: np.ndarray, *, threshold: float, hint: SpeakerHint) -> list[int]:
    """Agglomerative clustering on cosine distance, cut at a threshold.

    Average linkage (UPGMA), updated by the Lance-Williams recurrence. A threshold rather than
    a fixed *k* is what lets an unknown speaker count work at all: a monologue stays one
    speaker and a six-person meeting is not forced into two.

    Complexity is O(n^2) memory for the distance matrix and O(n^3) worst-case time, where n is
    the number of embeddable turns. For scale: a four-hour recording with a turn every five
    seconds is ~2,900 turns, so ~34 MB of float32 matrix and a few seconds of CPU — against a
    job that already costs over an hour of CPU at that length. The matrix is transient; the
    state actually carried across windows is just the embeddings, at well under a kilobyte
    each.
    """
    import numpy as np

    count = len(matrix)
    if count == 1:
        return [0]

    # Cosine distance in [0, 2]. Unit-normalising first turns the cosine into a dot product,
    # so the whole pairwise matrix is one matmul.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction, so it has no meaningful cosine distance to anything.
    # Guard rather than divide by zero and propagate NaN into every comparison.
    norms[norms == 0] = 1.0
    unit = matrix / norms
    distances = (1.0 - unit @ unit.T).astype(np.float32)
    np.fill_diagonal(distances, np.inf)

    members: list[list[int]] = [[i] for i in range(count)]
    sizes = np.ones(count, dtype=np.float32)

    while len(members) > 1:
        flat = int(np.argmin(distances))
        left, right = divmod(flat, distances.shape[0])
        closest = float(distances[left, right])

        if not _should_merge(len(members), closest, threshold=threshold, hint=hint):
            break

        # Lance-Williams for average linkage. The right-hand side is evaluated before the
        # assignment, so reading the old row while writing the new one is safe.
        merged = (sizes[left] * distances[left, :] + sizes[right] * distances[right, :]) / (
            sizes[left] + sizes[right]
        )
        distances[left, :] = merged
        distances[:, left] = merged
        distances[left, left] = np.inf

        distances = np.delete(np.delete(distances, right, axis=0), right, axis=1)
        sizes[left] += sizes[right]
        sizes = np.delete(sizes, right)
        members[left] = members[left] + members[right]
        del members[right]

    assignments = [0] * count
    for cluster, member_indices in enumerate(members):
        for member in member_indices:
            assignments[member] = cluster
    return assignments


def _should_merge(
    remaining: int,
    closest: float,
    *,
    threshold: float,
    hint: SpeakerHint,
) -> bool:
    """Whether to merge the closest pair, given the caller's hints.

    An exact count overrides the threshold entirely — the caller knows something the audio
    does not say. A range brackets it: keep merging past the threshold while there are still
    too many speakers, and stop short of it once there are few enough.
    """
    if hint.exact is not None:
        return remaining > hint.exact
    if hint.minimum is not None and remaining <= hint.minimum:
        return False
    if hint.maximum is not None and remaining > hint.maximum:
        return True
    return closest <= threshold


def _roster(turns: list[SpeakerTurn]) -> list[Speaker]:
    """Per-speaker totals, ordered by label so the roster is stable."""
    totals: dict[str, list[float]] = {}
    for turn in turns:
        entry = totals.setdefault(turn.speaker, [0.0, 0.0])
        entry[0] += turn.duration
        entry[1] += 1

    return [
        Speaker(
            id=speaker,
            total_speech_seconds=round(seconds, 3),
            turn_count=int(count),
        )
        for speaker, (seconds, count) in sorted(totals.items())
    ]


# --- Alignment ----------------------------------------------------------------------------


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    """Attach speaker labels to transcript segments.

    **The transcript's segmentation wins.** Segments are never re-cut at turn boundaries, and
    their text, count and ids are untouched. That is because the segment list is the published
    contract: consumers depend on its shape, SRT and WebVTT cues are derived straight from it,
    and ``test_windows_tile_the_audio_exactly`` pins the tiling invariant. An additive feature
    has no business changing any of that. Splitting segments at turn changes would be more
    correct and is deliberately deferred behind a major schema version.

    Assignment is a majority vote by time: the speaker holding the most of a segment gets it,
    and ``speaker_confidence`` records that winner's share. A low share means the segment
    straddles a turn change.

    When word timings are present the vote runs per word and is then summed, which is
    materially sharper: Whisper's segment edges come from decoder token positions and drift by
    a few hundred milliseconds, while word edges are DTW-aligned against the model's
    cross-attention. At a turn change that drift is exactly what mislabels the words either
    side of it.
    """
    if not turns:
        return segments

    return [_assign_one(segment, turns) for segment in segments]


def _assign_one(segment: Segment, turns: list[SpeakerTurn]) -> Segment:
    overlapping = _overlapping(segment.start, segment.end, turns)
    # `speakers_overlap` reports only what the backend detected as simultaneous speech. Two
    # speakers in sequence inside one segment is a different thing and is already carried by a
    # low `speaker_confidence`; collapsing the two would make the flag mean nothing.
    simultaneous = any(turn.overlap for turn in overlapping) or None

    if segment.words:
        words, totals = _assign_words(segment.words, turns)
        span = sum(totals.values())
    else:
        words = segment.words
        totals = _shares(segment.start, segment.end, overlapping)
        span = sum(totals.values())

    if not totals or span <= 0:
        # No turn covers this segment at all — the diariser and the transcriber disagree about
        # where speech is. Say nothing rather than guess.
        return segment.model_copy(update={"words": words, "speakers_overlap": simultaneous})

    winner = max(totals, key=lambda speaker: (totals[speaker], speaker))
    return segment.model_copy(
        update={
            "words": words,
            "speaker": winner,
            "speaker_confidence": round(totals[winner] / span, 4),
            "speakers_overlap": simultaneous,
        }
    )


def _assign_words(words, turns: list[SpeakerTurn]):
    """Label each word, and total the labelled duration per speaker."""
    labelled = []
    totals: dict[str, float] = {}
    for word in words:
        shares = _shares(word.start, word.end, _overlapping(word.start, word.end, turns))
        if shares:
            speaker = max(shares, key=lambda name: (shares[name], name))
            labelled.append(word.model_copy(update={"speaker": speaker}))
            totals[speaker] = totals.get(speaker, 0.0) + max(word.end - word.start, 0.0)
        else:
            labelled.append(word)
    return labelled, totals


def _overlapping(start: float, end: float, turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    return [turn for turn in turns if turn.start < end and turn.end > start]


def _shares(start: float, end: float, turns: list[SpeakerTurn]) -> dict[str, float]:
    """Seconds of ``[start, end)`` held by each speaker."""
    shares: dict[str, float] = {}
    for turn in turns:
        held = min(end, turn.end) - max(start, turn.start)
        if held > 0:
            shares[turn.speaker] = shares.get(turn.speaker, 0.0) + held
    return shares


def iter_turn_blocks(
    segments: list[Segment],
) -> Iterator[tuple[str | None, list[Segment]]]:
    """Group consecutive segments by speaker.

    Used by the plain-text exporter, where merging a speaker's consecutive segments into one
    block is the difference between a wall of lines and a readable transcript.
    """
    block: list[Segment] = []
    current: str | None = None

    for segment in segments:
        if block and segment.speaker != current:
            yield current, block
            block = []
        current = segment.speaker
        block.append(segment)

    if block:
        yield current, block
