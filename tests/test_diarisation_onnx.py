"""Tests for the ONNX speaker-diarisation backend.

Split from ``test_diarisation.py`` because the requirements are different. Those tests use a
fake backend and run in milliseconds; these need `sherpa-onnx`, a 28 MB downloaded model, and
macOS `say` to build ground-truth audio. They are marked ``slow`` and skip themselves when any
of that is missing, so the fast suite is unaffected and a Linux contributor still gets green.

The measurement that matters is ``test_a_two_speaker_recording_is_attributed_correctly``: it
builds audio where the truth is known by construction, runs the real model, and scores
diarisation error rate. Everything else in this file supports that one.

**What a green run here does and does not prove.** It proves the pipeline is correct end to
end — VAD, sub-segmentation, embedding, global clustering, de-duplication, merging, alignment.
It does not prove accuracy on real conversational audio: synthetic voices sit about 0.84 cosine
distance apart where two real people of one gender and accent sit far closer, and the clustering
threshold is what suffers. See ``scripts/evaluate_diarisation.py`` for pointing this at a real
RTTM corpus.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from transcription import audio as audio_module
from transcription.config import Settings
from transcription.diarisation import diarise_wav
from transcription.errors import DiarisationUnavailableError

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _model_available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    from transcription.diarisation_onnx import resolve_model_path

    return resolve_model_path(Settings()).is_file()


backend_required = pytest.mark.skipif(
    not _model_available(),
    reason='needs `pip install -e ".[diarise]"` and `transcribe diarise-warm`',
)
macos_required = pytest.mark.skipif(
    sys.platform != "darwin", reason="ground-truth fixtures need macOS `say`"
)


def onnx_settings(**overrides) -> Settings:
    base = {
        "diarise": True,
        "diarisation_backend": "onnx",
        # One window over the whole fixture, so these tests exercise the backend rather than
        # the window grid, which test_diarisation.py already covers.
        "diarisation_window_seconds": 600.0,
        "whisper_model": "tiny",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(scope="module")
def two_speakers(tmp_path_factory) -> tuple[Path, list[tuple[float, float, str]]]:
    """A two-speaker recording plus its exact ground truth.

    Module-scoped because `say` plus ffmpeg takes a few seconds per turn, and every test here
    can share one fixture.
    """
    if sys.platform != "darwin":
        pytest.skip("needs macOS `say`")
    out = tmp_path_factory.mktemp("spk") / "two"
    subprocess.run(
        [
            sys.executable, str(SCRIPTS / "make_speaker_fixture.py"), str(out),
            "--speakers", "2", "--turns", "6", "--gap", "0.5",
        ],
        check=True, capture_output=True,
    )
    from scripts.evaluate_diarisation import read_rttm

    return out.with_suffix(".wav"), read_rttm(out.with_suffix(".rttm"))


def _diarise(wav: Path, settings: Settings, tmp_path: Path):
    from transcription.diarisation_onnx import OnnxDiariser

    normalised = audio_module.normalize(
        wav, tmp_path / "n.wav", max_duration_seconds=settings.max_duration_seconds
    )
    diariser = OnnxDiariser(settings)
    diariser.warm()
    return diarise_wav(
        normalised,
        diariser=diariser,
        settings=settings,
        audio_duration=audio_module.probe_duration(normalised),
    )


# --- The measurement ----------------------------------------------------------------------


@pytest.mark.slow
@macos_required
@backend_required
def test_a_two_speaker_recording_is_attributed_correctly(two_speakers, tmp_path) -> None:
    """The real model against audio whose truth is known by construction.

    Scored as diarisation error rate, with the standard 250 ms collar, over the optimal mapping
    of generated labels onto reference ones — because `SPEAKER_00` and `Daniel` are the same
    person if that is the mapping that fits.
    """
    from scripts.evaluate_diarisation import score

    wav, reference = two_speakers
    settings = onnx_settings()
    result = _diarise(wav, settings, tmp_path)

    assert result.info.status == "succeeded"
    assert result.info.speaker_count == 2, (
        f"expected 2 speakers, found {result.info.speaker_count}: "
        "the clustering threshold may need tuning"
    )

    hypothesis = [(t.start, t.end, t.speaker) for t in result.turns]
    report = score(
        reference, hypothesis, audio_module.probe_duration(wav), collar=0.25
    )

    # Synthetic voices are an easy case, so the bar is high on purpose: anything above a few
    # percent here would mean something is actually broken, not merely imperfect.
    assert report["der"] < 0.05, f"DER {report['der']:.2%} on an easy case: {report}"
    assert report["missed"] < 0.02, f"VAD lost speech: {report}"


@pytest.mark.slow
@macos_required
@backend_required
def test_turns_alternate_as_the_recording_does(two_speakers, tmp_path) -> None:
    """Six alternating turns must come back as six alternating turns.

    This is what the merge step buys. Without it the backend's 1.5 s sub-segments would arrive
    as roughly twenty turns, and `Speaker.turn_count` would be meaningless.
    """
    wav, reference = two_speakers
    result = _diarise(wav, onnx_settings(), tmp_path)

    labels = [turn.speaker for turn in result.turns]
    assert len(labels) == len(reference), f"expected {len(reference)} turns, got {labels}"
    # Strictly alternating, which is how the fixture was built.
    assert all(a != b for a, b in zip(labels, labels[1:], strict=False)), labels
    assert [speaker.turn_count for speaker in result.speakers] == [3, 3]


@pytest.mark.slow
@macos_required
@backend_required
def test_a_known_speaker_count_is_respected(two_speakers, tmp_path) -> None:
    """The cheapest accuracy on offer: most callers know how many people were in the room."""
    from transcription.diarisation import SpeakerHint
    from transcription.diarisation_onnx import OnnxDiariser

    wav, _ = two_speakers
    settings = onnx_settings()
    normalised = audio_module.normalize(wav, tmp_path / "n.wav", max_duration_seconds=3600)
    diariser = OnnxDiariser(settings)
    diariser.warm()

    forced = diarise_wav(
        normalised,
        diariser=diariser,
        settings=settings,
        audio_duration=audio_module.probe_duration(normalised),
        hint=SpeakerHint(exact=1),
    )

    # One speaker was asserted, so one speaker is reported even though the audio has two.
    # The caller is allowed to be wrong; the system is not allowed to ignore them.
    assert forced.info.speaker_count == 1
    assert forced.info.speakers_hint == 1


@pytest.mark.slow
@macos_required
@backend_required
def test_the_backend_never_claims_to_detect_overlap(two_speakers, tmp_path) -> None:
    """It has no overlap detector, so it must not pretend to.

    `pyannote`'s segmentation model does detect simultaneous speech. This pipeline — VAD plus
    fixed sub-segmentation — has nothing that could, so every turn reports `overlap=False` and
    the documentation says so.
    """
    wav, _ = two_speakers
    result = _diarise(wav, onnx_settings(), tmp_path)

    assert all(turn.overlap is False for turn in result.turns)
    assert result.info.overlap_seconds == 0.0


@pytest.mark.slow
@macos_required
@backend_required
def test_the_pipeline_labels_a_real_transcript(two_speakers, tmp_path) -> None:
    """Through `transcribe_file`, with a real Whisper model and real speaker labels."""
    from transcription.pipeline import transcribe_file

    wav, reference = two_speakers
    result = transcribe_file(wav, settings=onnx_settings())

    assert result.diarisation.status == "succeeded"
    assert result.diarisation.speaker_count == 2
    assert result.segments
    assert all(segment.speaker for segment in result.segments), "every segment needs a label"
    # Diarisation forces word timings on, and the words carry labels too.
    assert result.model.word_timestamps is True
    assert any(word.speaker for segment in result.segments for word in segment.words or [])
    # Confidence is high because turns and segments agree on this fixture; a low value would
    # mean segments straddling turn changes.
    assert min(s.speaker_confidence for s in result.segments) > 0.8


# --- Unit level, no model needed ----------------------------------------------------------


def test_sub_segmentation_absorbs_a_short_tail() -> None:
    """A 0.3 s fragment cannot be embedded, so it joins the piece before it."""
    from transcription.diarisation_onnx import _split

    pieces = _split(0.0, 4.3, 1.5, 1.0)

    assert pieces[0] == (0.0, 1.5)
    assert pieces[-1][1] == 4.3
    assert all(end - start >= 1.0 for start, end in pieces), pieces


def test_sub_segmentation_leaves_short_speech_whole() -> None:
    """Below about one and a half pieces there is nothing sensible to divide."""
    from transcription.diarisation_onnx import _split

    assert _split(2.0, 3.0, 1.5, 1.0) == [(2.0, 3.0)]
    assert _split(0.0, 0.4, 1.5, 1.0) == [(0.0, 0.4)]
    assert _split(5.0, 5.0, 1.5, 1.0) == []


def test_sub_segmentation_tiles_without_gaps_or_overlaps() -> None:
    """Pieces become turns, so a gap would lose speech and an overlap would double-count it."""
    from transcription.diarisation_onnx import _split

    pieces = _split(1.0, 11.0, 1.5, 1.0)

    assert pieces[0][0] == 1.0
    assert pieces[-1][1] == 11.0
    for (_, end), (next_start, _) in zip(pieces, pieces[1:], strict=False):
        assert end == next_start, pieces


def test_the_backend_refuses_clearly_when_the_model_is_missing(tmp_path) -> None:
    """A missing model is a deployment problem, and the message has to say what to run."""
    pytest.importorskip("sherpa_onnx")
    from transcription.diarisation_onnx import OnnxDiariser

    diariser = OnnxDiariser(
        onnx_settings(), model_path=tmp_path / "absent" / "nothing.onnx"
    )

    with pytest.raises(DiarisationUnavailableError, match="diarise-warm"):
        diariser.warm()


def test_the_backend_name_identifies_the_weights() -> None:
    """Provenance has to name the model: swapping it changes who the service says spoke."""
    pytest.importorskip("sherpa_onnx")
    from transcription.diarisation_onnx import OnnxDiariser

    name = OnnxDiariser(onnx_settings(), model_path="/tmp/some_model.onnx").name

    assert "some_model" in name
