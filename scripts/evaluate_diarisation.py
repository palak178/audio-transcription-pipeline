#!/usr/bin/env python3
"""Score diarisation against RTTM ground truth, reporting diarisation error rate.

DER is the standard metric, and it is the sum of three different mistakes over the total
reference speech:

* **missed speech** — reference says someone spoke, the system says nobody did;
* **false alarm** — the system says someone spoke where the reference has silence;
* **confusion** — both agree someone spoke, and they disagree about who.

Confusion is the interesting one, and scoring it needs care: speaker labels are arbitrary.
`SPEAKER_00` and `Daniel` are the same person if that is the mapping that fits best, so the
score is taken over the **optimal** assignment of hypothesis labels to reference labels. This
implementation tries every permutation, which is exact and fine up to about seven speakers; a
real scorer uses the Hungarian algorithm to avoid the factorial. The limit is checked rather
than assumed.

A **collar** is applied by convention — boundaries within the collar of a reference boundary
are not scored, because human annotators do not agree to within a few hundred milliseconds
either. The standard is 250 ms. Report the collar with any number, since DER without one is a
different and much harsher metric.

Usage::

    python scripts/evaluate_diarisation.py out/two_speakers.wav out/two_speakers.rttm
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

RESOLUTION = 0.01  # 10 ms scoring grid, which is finer than any boundary we can defend.


def read_rttm(path: Path) -> list[tuple[float, float, str]]:
    turns = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        start, duration, speaker = float(fields[3]), float(fields[4]), fields[7]
        turns.append((start, start + duration, speaker))
    return sorted(turns)


def to_frames(turns, total_frames):
    """Who is speaking in each 10 ms frame. None where nobody is.

    A frame grid rather than interval arithmetic because it makes the three error types fall
    out as simple counts, and because overlap — which this project flags but does not resolve —
    would otherwise need interval splitting to score at all.
    """
    frames = [None] * total_frames
    for start, end, speaker in turns:
        for index in range(int(start / RESOLUTION), min(int(end / RESOLUTION), total_frames)):
            frames[index] = speaker
    return frames


def collar_mask(turns, total_frames, collar):
    """Frames to exclude: within ``collar`` of any reference boundary."""
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


def score(reference, hypothesis, duration, collar=0.25):
    total_frames = int(duration / RESOLUTION) + 1
    ref = to_frames(reference, total_frames)
    hyp = to_frames(hypothesis, total_frames)
    skip = collar_mask(reference, total_frames, collar)

    ref_speakers = sorted({s for s in ref if s})
    hyp_speakers = sorted({s for s in hyp if s})
    if len(hyp_speakers) > 7:
        sys.exit(
            f"error: {len(hyp_speakers)} hypothesis speakers is too many for exhaustive "
            "mapping; use a scorer with the Hungarian algorithm"
        )

    scored = [i for i in range(total_frames) if not skip[i]]
    reference_speech = sum(1 for i in scored if ref[i])
    if reference_speech == 0:
        sys.exit("error: the reference contains no scored speech")

    # Try every injective mapping of hypothesis labels onto reference labels, and keep the one
    # that minimises confusion. Labels are arbitrary, so this is part of the metric, not a
    # convenience.
    best = None
    # Candidates are the *reference* labels, padded with None so a hypothesis speaker who
    # corresponds to nobody can be left unmapped. Permuting the hypothesis labels instead maps
    # them onto themselves, which scores every frame as confusion — exactly 100%, which is how
    # this bug announced itself.
    candidates = list(ref_speakers) + [None] * max(0, len(hyp_speakers) - len(ref_speakers))
    assignments = (
        itertools.permutations(candidates, len(hyp_speakers)) if hyp_speakers else [()]
    )
    for assignment in assignments:
        mapping = dict(zip(hyp_speakers, assignment, strict=False))
        missed = false_alarm = confusion = 0
        for i in scored:
            r, h = ref[i], mapping.get(hyp[i]) if hyp[i] else None
            if r and not h:
                missed += 1
            elif h and not r:
                false_alarm += 1
            elif r and h and r != h:
                confusion += 1
        total = missed + false_alarm + confusion
        if best is None or total < best[0]:
            best = (total, missed, false_alarm, confusion, mapping)

    total, missed, false_alarm, confusion, mapping = best
    return {
        "der": total / reference_speech,
        "missed": missed / reference_speech,
        "false_alarm": false_alarm / reference_speech,
        "confusion": confusion / reference_speech,
        "collar_seconds": collar,
        "reference_speakers": len(ref_speakers),
        "hypothesis_speakers": len(hyp_speakers),
        "mapping": {k: v for k, v in mapping.items() if v},
        "scored_speech_seconds": round(reference_speech * RESOLUTION, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("reference", type=Path, help="Ground-truth RTTM.")
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--model", default=None, help="Whisper model size.")
    args = parser.parse_args()

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

    import tempfile

    with tempfile.TemporaryDirectory(prefix="der-") as tmp:
        wav = audio_module.normalize(
            args.audio, Path(tmp) / "n.wav", max_duration_seconds=settings.max_duration_seconds
        )
        diariser = OnnxDiariser(settings)
        diariser.warm()
        result = diarise_wav(
            wav, diariser=diariser, settings=settings, audio_duration=duration
        )

    hypothesis = [(t.start, t.end, t.speaker) for t in result.turns]
    report = score(reference, hypothesis, duration, collar=args.collar)

    print(f"audio                : {args.audio.name}  ({duration:.1f}s)")
    print(f"backend              : {result.info.backend}")
    print(f"threshold            : {result.info.clustering_threshold}")
    print(f"speakers  reference  : {report['reference_speakers']}")
    print(f"speakers  found      : {report['hypothesis_speakers']}")
    print(f"mapping              : {report['mapping']}")
    print(f"scored speech        : {report['scored_speech_seconds']}s "
          f"(collar {report['collar_seconds']}s)")
    print()
    print(f"  missed speech      : {report['missed']:7.2%}")
    print(f"  false alarm        : {report['false_alarm']:7.2%}")
    print(f"  speaker confusion  : {report['confusion']:7.2%}")
    print(f"  DER                : {report['der']:7.2%}")

    if report["hypothesis_speakers"] != report["reference_speakers"]:
        print("\nnote: the speaker count is wrong, which usually means the clustering "
              "threshold needs tuning for this audio.")


if __name__ == "__main__":
    main()
