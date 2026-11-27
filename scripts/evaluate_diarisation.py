#!/usr/bin/env python3
"""Score diarisation against RTTM ground truth, reporting diarisation error rate.

DER is the standard metric, and it is the sum of three different mistakes over the total
reference speech:

* **missed speech** — the reference has more speakers active than the system found;
* **false alarm** — the system found more speakers active than the reference has;
* **confusion** — both agree on how many, and disagree about who.

Two things in here are easy to get wrong, and both were wrong in the first version.

**Frames hold a set of speakers, not one.** Real annotations overlap: 230 of 300 VoxConverse
files contain simultaneous speech. Assigning one speaker per frame lets the last turn silently
win, which understates missed speech and misattributes the rest — a real corpus would have been
scored wrongly and quietly.

**The label mapping is solved, not searched.** Labels are arbitrary, so `SPEAKER_00` and
`Daniel` are the same person if that is the mapping that fits best; choosing it is part of the
metric. The first version tried every permutation, which caps out around seven speakers — and
61 of 300 VoxConverse files have more than that, one reaching twenty. This uses the Hungarian
algorithm, implemented here because `scipy` is not a dependency of this project.

A **collar** is applied by convention: frames within the collar of a reference boundary are not
scored, because human annotators do not agree to within a few hundred milliseconds either. The
standard is 250 ms. Always report the collar with a DER, since the number means something
different without one.

**This is not a benchmark tool.** It is a dependency-free check that runs in CI against
synthetic fixtures. For a published, comparable number on a real corpus, use `dscore` (the
NIST-derived reference implementation) or `pyannote.metrics`.

Usage::

    python scripts/evaluate_diarisation.py out/two.wav out/two.rttm
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

RESOLUTION = 0.01  # 10 ms scoring grid, finer than any boundary this system can defend.


def read_rttm(path: Path) -> list[tuple[float, float, str]]:
    """Parse the turns out of an RTTM file. Overlapping turns are kept as they are."""
    turns = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) > 7 and fields[0] == "SPEAKER":
            start, duration, speaker = float(fields[3]), float(fields[4]), fields[7]
            if duration > 0:
                turns.append((start, start + duration, speaker))
    return sorted(turns)


def to_frames(turns, total_frames: int) -> list[set[str]]:
    """The set of speakers active in each 10 ms frame."""
    frames: list[set[str]] = [set() for _ in range(total_frames)]
    for start, end, speaker in turns:
        first = max(0, int(start / RESOLUTION))
        last = min(int(end / RESOLUTION), total_frames)
        for index in range(first, last):
            frames[index].add(speaker)
    return frames


def collar_mask(turns, total_frames: int, collar: float) -> list[bool]:
    """Frames to exclude from scoring: those within ``collar`` of a reference boundary."""
    mask = [False] * total_frames
    if collar <= 0:
        return mask
    width = int(collar / RESOLUTION)
    for start, end, _ in turns:
        for boundary in (start, end):
            centre = int(boundary / RESOLUTION)
            for index in range(max(0, centre - width), min(total_frames, centre + width + 1)):
                mask[index] = True
    return mask


def hungarian(cost: list[list[float]]) -> list[tuple[int, int | None]]:
    """Minimum-cost assignment over a rectangular matrix. Returns ``(row, column)`` pairs.

    The O(n^3) shortest-augmenting-path formulation. Rows may outnumber columns or the reverse;
    the matrix is padded square with zero-cost dummies, and a row matched only to a dummy comes
    back paired with ``None``.
    """
    rows = len(cost)
    columns = len(cost[0]) if rows else 0
    if rows == 0 or columns == 0:
        return [(row, None) for row in range(rows)]

    size = max(rows, columns)
    padded = [
        [cost[r][c] if r < rows and c < columns else 0.0 for c in range(size)]
        for r in range(size)
    ]

    # One-based internally, with index 0 used as the sentinel the algorithm needs.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    match = [0] * (size + 1)  # match[column] -> row
    parent = [0] * (size + 1)  # parent[column] -> previous column on the augmenting path

    for row in range(1, size + 1):
        match[0] = row
        column = 0
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)

        while True:
            used[column] = True
            current = match[column]
            delta = math.inf
            next_column = 0

            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = padded[current - 1][candidate - 1] - u[current] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    parent[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate

            for candidate in range(size + 1):
                if used[candidate]:
                    u[match[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta

            column = next_column
            if match[column] == 0:
                break

        while column:
            previous = parent[column]
            match[column] = match[previous]
            column = previous

    assigned: dict[int, int] = {}
    for column in range(1, size + 1):
        if match[column]:
            assigned[match[column] - 1] = column - 1

    result = []
    for row in range(rows):
        column_index = assigned.get(row)
        matched = (
            column_index if column_index is not None and column_index < columns else None
        )
        result.append((row, matched))
    return result


def optimal_mapping(ref_frames, hyp_frames, scored, ref_speakers, hyp_speakers) -> dict:
    """Assign hypothesis labels to reference labels so that frame agreement is maximised."""
    if not hyp_speakers or not ref_speakers:
        return {}

    hyp_index = {speaker: i for i, speaker in enumerate(hyp_speakers)}
    ref_index = {speaker: i for i, speaker in enumerate(ref_speakers)}
    agreement = [[0] * len(ref_speakers) for _ in hyp_speakers]

    for frame in scored:
        for hypothesis_speaker in hyp_frames[frame]:
            for reference_speaker in ref_frames[frame]:
                agreement[hyp_index[hypothesis_speaker]][ref_index[reference_speaker]] += 1

    # Hungarian minimises, so maximise agreement by minimising its negation.
    cost = [[float(-value) for value in row] for row in agreement]
    return {
        hyp_speakers[row]: ref_speakers[column]
        for row, column in hungarian(cost)
        # A pairing with no agreement at all is not a correspondence. Leaving the hypothesis
        # speaker unmapped is more honest than inventing one, and it scores as confusion.
        if column is not None and agreement[row][column] > 0
    }


def score(reference, hypothesis, duration: float, collar: float = 0.25) -> dict:
    """Diarisation error rate and its three components, overlap included."""
    total_frames = int(duration / RESOLUTION) + 1
    ref_frames = to_frames(reference, total_frames)
    hyp_frames = to_frames(hypothesis, total_frames)
    skip = collar_mask(reference, total_frames, collar)
    scored = [index for index in range(total_frames) if not skip[index]]

    ref_speakers = sorted({s for frame in ref_frames for s in frame})
    hyp_speakers = sorted({s for frame in hyp_frames for s in frame})
    mapping = optimal_mapping(ref_frames, hyp_frames, scored, ref_speakers, hyp_speakers)

    missed = false_alarm = confusion = reference_speech = 0
    overlap_frames = 0

    for frame in scored:
        reference_active = ref_frames[frame]
        mapped = {mapping.get(s) for s in hyp_frames[frame]}
        mapped.discard(None)

        n_ref = len(reference_active)
        n_hyp = len(hyp_frames[frame])
        correct = len(reference_active & mapped)

        reference_speech += n_ref
        if n_ref > 1:
            overlap_frames += 1
        # The NIST decomposition: whichever side has more speakers accounts for the
        # miss or false alarm, and the agreed-upon count that does not match is confusion.
        missed += max(0, n_ref - n_hyp)
        false_alarm += max(0, n_hyp - n_ref)
        confusion += min(n_ref, n_hyp) - correct

    if reference_speech == 0:
        sys.exit("error: the reference contains no scored speech")

    return {
        "der": (missed + false_alarm + confusion) / reference_speech,
        "missed": missed / reference_speech,
        "false_alarm": false_alarm / reference_speech,
        "confusion": confusion / reference_speech,
        "collar_seconds": collar,
        "reference_speakers": len(ref_speakers),
        "hypothesis_speakers": len(hyp_speakers),
        "mapping": mapping,
        "scored_speech_seconds": round(reference_speech * RESOLUTION, 2),
        "overlap_seconds": round(overlap_frames * RESOLUTION, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("reference", type=Path, help="Ground-truth RTTM.")
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--model", default=None, help="Whisper model size.")
    args = parser.parse_args()

    import tempfile

    from transcription import audio as audio_module
    from transcription.config import Settings
    from transcription.diarisation import diarise_wav
    from transcription.diarisation_onnx import OnnxDiariser

    overrides = {"diarise": True, "diarisation_backend": "onnx"}
    if args.model:
        overrides["whisper_model"] = args.model
    settings = Settings(**overrides)

    reference = read_rttm(args.reference)
    duration = audio_module.probe_duration(args.audio)

    with tempfile.TemporaryDirectory(prefix="der-") as tmp:
        wav = audio_module.normalize(
            args.audio, Path(tmp) / "n.wav", max_duration_seconds=settings.max_duration_seconds
        )
        diariser = OnnxDiariser(settings)
        diariser.warm()
        result = diarise_wav(wav, diariser=diariser, settings=settings, audio_duration=duration)

    hypothesis = [(t.start, t.end, t.speaker) for t in result.turns]
    report = score(reference, hypothesis, duration, collar=args.collar)

    print(f"audio                : {args.audio.name}  ({duration:.1f}s)")
    print(f"backend              : {result.info.backend}")
    print(f"threshold            : {result.info.clustering_threshold}")
    print(f"speakers  reference  : {report['reference_speakers']}")
    print(f"speakers  found      : {report['hypothesis_speakers']}")
    print(f"mapping              : {report['mapping']}")
    print(
        f"scored speech        : {report['scored_speech_seconds']}s "
        f"(collar {report['collar_seconds']}s, {report['overlap_seconds']}s overlapped)"
    )
    print()
    print(f"  missed speech      : {report['missed']:7.2%}")
    print(f"  false alarm        : {report['false_alarm']:7.2%}")
    print(f"  speaker confusion  : {report['confusion']:7.2%}")
    print(f"  DER                : {report['der']:7.2%}")

    if report["hypothesis_speakers"] != report["reference_speakers"]:
        print(
            "\nnote: the speaker count is wrong, which usually means the clustering threshold "
            "needs tuning for this audio."
        )
    if report["overlap_seconds"] > 0:
        print(
            f"\nnote: {report['overlap_seconds']}s of the reference is overlapped speech. This "
            "backend has no overlap detection, so those frames can only ever be scored as "
            "missed speech."
        )


if __name__ == "__main__":
    main()
