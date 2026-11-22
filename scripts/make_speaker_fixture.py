#!/usr/bin/env python3
"""Generate a multi-speaker audio fixture with exact ground truth.

Diarisation is the one feature in this project whose quality cannot be judged by reading the
code. It needs measurement, and measurement needs labelled audio: who spoke, when, to the
millisecond. This script manufactures that.

**How.** macOS `say` renders each utterance with a named voice, so the exact start and end of
every turn is known by construction rather than annotated afterwards. The turns are concatenated
with ffmpeg, optional silence is inserted between them, and the boundaries are written out as
RTTM — the format every diarisation benchmark and scoring tool uses.

**What this is worth, stated honestly.** TTS voices are far more acoustically distinct than two
real people of the same gender and accent. Measured on this fixture, same-speaker cosine
distance is ~0.005 and different-speaker ~0.84; on real conversational audio the equivalent
numbers are much closer together, and the clustering threshold is the thing that suffers. So
this fixture proves the **plumbing** end to end — VAD, sub-segmentation, embedding, global
clustering, boundary de-duplication, alignment, output — and it does *not* establish that the
system handles hard audio. For that, point the evaluator at VoxConverse or AMI, which ship real
RTTM ground truth.

It is a generator rather than a committed binary for the same reason the other fixtures are:
`make fixtures` builds them, nothing large lives in git, and nobody's voice is published.

Usage::

    python scripts/make_speaker_fixture.py out/two_speakers
    python scripts/make_speaker_fixture.py out/meeting --speakers 3 --turns 12 --gap 0.4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Chosen for acoustic distance: different genders and different accents, so the fixture is an
# easy case on purpose. A harder fixture would use voices of one gender and accent.
VOICES = ["Daniel", "Samantha", "Rishi", "Karen", "Moira"]

# Deliberately varied in length. Equal-length turns would let a bug that quantises boundaries
# to a fixed grid score well for the wrong reason.
LINES = [
    "Right, shall we start with the quarterly numbers before anyone else joins the call?",
    "Yes, although I think the headline figure needs more context than the slide gives it.",
    "The deployment finished about twenty minutes ago and nothing has caught fire yet.",
    "I disagree, and I think the earlier estimate was far too optimistic about timelines.",
    "Can you send that spreadsheet over after the meeting ends, whenever you get a moment?",
    "We measured it twice and the second run came back almost exactly the same as the first.",
    "That sounds reasonable to me, so let us go ahead and book the time for next week.",
    "Hold on, I want to understand the failure mode properly before we agree to anything.",
    "The documentation is out of date, which is partly my fault and partly nobody's fault.",
    "Fine. I will write it up this afternoon and circulate a draft before the end of the day.",
    "One more thing, and then I promise I will let everybody get back to their actual work.",
    "Thanks all, that was genuinely useful, and far shorter than last week's version of it.",
]


def require(binary: str) -> str:
    found = shutil.which(binary)
    if not found:
        sys.exit(f"error: {binary} is not on PATH")
    return found


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            require("ffprobe"), "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def render(voice: str, text: str, destination: Path) -> Path:
    """Render one utterance, normalised to the project's canonical 16 kHz mono WAV."""
    aiff = destination.with_suffix(".aiff")
    subprocess.run([require("say"), "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(
        [
            require("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(aiff), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
        ],
        check=True,
    )
    aiff.unlink(missing_ok=True)
    return destination


def silence(seconds: float, destination: Path) -> Path:
    subprocess.run(
        [
            require("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", f"{seconds}",
            "-c:a", "pcm_s16le", str(destination),
        ],
        check=True,
    )
    return destination


def build(output_stem: Path, *, speakers: int, turns: int, gap: float) -> dict:
    if speakers > len(VOICES):
        sys.exit(f"error: only {len(VOICES)} voices are available")

    work = output_stem.parent / f".{output_stem.name}-parts"
    work.mkdir(parents=True, exist_ok=True)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    pieces: list[Path] = []
    truth: list[dict] = []
    clock = 0.0

    for index in range(turns):
        voice = VOICES[index % speakers]
        line = LINES[index % len(LINES)]
        piece = render(voice, line, work / f"turn_{index:03d}.wav")
        duration = probe_duration(piece)

        truth.append(
            {
                "speaker": voice,
                "start": round(clock, 3),
                "end": round(clock + duration, 3),
                "text": line,
            }
        )
        pieces.append(piece)
        clock += duration

        # Silence between turns, so VAD has a real boundary to find. Without it the turns run
        # together and the fixture stops testing boundary detection at all.
        if gap > 0 and index < turns - 1:
            pieces.append(silence(gap, work / f"gap_{index:03d}.wav"))
            clock += gap

    concat_list = work / "concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in pieces))
    wav = output_stem.with_suffix(".wav")
    subprocess.run(
        [
            require("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c:a", "pcm_s16le", str(wav),
        ],
        check=True,
    )

    # RTTM, because that is what every diarisation scorer reads. One line per turn:
    # SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    rttm = output_stem.with_suffix(".rttm")
    rttm.write_text(
        "".join(
            f"SPEAKER {wav.stem} 1 {t['start']:.3f} {t['end'] - t['start']:.3f} "
            f"<NA> <NA> {t['speaker']} <NA> <NA>\n"
            for t in truth
        )
    )
    # JSON too, since it carries the text and is easier to read in a failure message.
    meta = {
        "audio": wav.name,
        "duration_seconds": round(probe_duration(wav), 3),
        "speakers": sorted({t["speaker"] for t in truth}),
        "gap_seconds": gap,
        "turns": truth,
        "note": (
            "Synthetic TTS voices, which are far more acoustically distinct than two real "
            "people. Proves the pipeline end to end; does not establish accuracy on real "
            "conversational audio."
        ),
    }
    output_stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")

    shutil.rmtree(work, ignore_errors=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", type=Path, help="Output path without an extension.")
    parser.add_argument("--speakers", type=int, default=2)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument(
        "--gap", type=float, default=0.5, help="Silence between turns, in seconds."
    )
    parser.add_argument(
        "--voices",
        default=None,
        help=(
            "Comma-separated voice names, overriding the default set. Use this to build a "
            "deliberately hard case: two voices of the same gender are far closer in "
            "embedding space than the mixed default."
        ),
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        sys.exit("error: this script needs macOS `say`; use a public RTTM corpus elsewhere")

    if args.voices:
        global VOICES
        VOICES = [v.strip() for v in args.voices.split(",") if v.strip()]
        args.speakers = min(args.speakers, len(VOICES))

    meta = build(args.output, speakers=args.speakers, turns=args.turns, gap=args.gap)
    print(f"wrote {args.output.with_suffix('.wav')}")
    print(f"      {args.output.with_suffix('.rttm')}")
    print(f"      {args.output.with_suffix('.json')}")
    print(
        f"{meta['duration_seconds']}s, {len(meta['turns'])} turns, "
        f"{len(meta['speakers'])} speakers: {', '.join(meta['speakers'])}"
    )


if __name__ == "__main__":
    main()
