"""Tests for speaker diarisation.

No diarisation model ships with this repository, so these tests carry more weight than usual:
they are the only thing standing behind the parts that *are* implemented. The three that
matter most, because they pin invariants that are easy to assert and hard to keep true:

* ``test_the_same_voice_in_different_windows_gets_one_label`` — the whole reason global
  clustering exists rather than per-window labels.
* ``test_a_turn_seen_by_two_windows_is_kept_once`` — the cost of overlapping windows, which
  the transcription path deliberately avoided paying.
* ``test_output_is_unchanged_when_nothing_is_diarised`` — that an additive feature really is
  additive.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeDiariser, embedding, ffmpeg_required
from transcription import audio as audio_module
from transcription import exporters
from transcription.config import Settings
from transcription.diarisation import (
    RawTurn,
    SpeakerHint,
    SpeakerTurn,
    assign_speakers,
    diarise_wav,
    iter_turn_blocks,
    load_diariser,
    window_authority,
)
from transcription.errors import DiarisationUnavailableError
from transcription.models import Segment, TranscriptionResult, Word
from transcription.pipeline import transcribe_file

# Two voices a full cosine distance apart, and a third close to the first.
VOICE_A = embedding(1.0, 0.0)
VOICE_B = embedding(0.0, 1.0)
VOICE_A_AGAIN = embedding(0.97, 0.05)


def diarisation_settings(**overrides) -> Settings:
    base = {
        "whisper_model": "fake",
        "diarise": True,
        "diarisation_window_seconds": 10.0,
        "diarisation_overlap_seconds": 2.0,
        "diarisation_threshold": 0.5,
        "diarisation_min_embedding_seconds": 1.0,
        "max_duration_seconds": 3600,
    }
    base.update(overrides)
    return Settings(**base)


def turn(start: float, end: float, vector=None, *, overlap: bool = False) -> RawTurn:
    return RawTurn(start=start, end=end, embedding=vector, overlap=overlap)


# --- The window grid ----------------------------------------------------------------------


@ffmpeg_required
def test_the_grid_overlaps_by_the_configured_amount(make_audio, tmp_path) -> None:
    source = make_audio("grid.wav", seconds=25.0)
    wav = audio_module.normalize(source, tmp_path / "n.wav", max_duration_seconds=60)

    windows = list(audio_module.iter_wav_grid(wav, window_seconds=10.0, overlap_seconds=2.0))
    offsets = [round(offset, 3) for offset, _ in windows]

    # Stride is window - overlap = 8 s. There is no window at 24 s because the one at 16 s
    # already reaches the end of the 25 s file: the grid stops as soon as the audio is
    # covered rather than emitting a redundant tail.
    assert offsets == [0.0, 8.0, 16.0]
    assert [len(samples) for _, samples in windows[:-1]] == [160_000] * 2
    # Coverage is complete: the last window ends at the end of the audio.
    assert offsets[-1] + len(windows[-1][1]) / 16_000 == pytest.approx(25.0)


@ffmpeg_required
def test_the_grid_covers_short_audio_in_one_window(make_audio, tmp_path) -> None:
    source = make_audio("short.wav", seconds=3.0)
    wav = audio_module.normalize(source, tmp_path / "n.wav", max_duration_seconds=60)

    windows = list(audio_module.iter_wav_grid(wav, window_seconds=10.0, overlap_seconds=2.0))

    assert len(windows) == 1
    assert windows[0][0] == 0.0


@ffmpeg_required
def test_the_grid_terminates_when_overlap_exceeds_the_window(make_audio, tmp_path) -> None:
    """A misconfigured overlap must not shrink the stride to nothing.

    Without the half-window clamp this advances one frame at a time, which on a long file is
    indistinguishable from a hang: 12 s of audio produced 112,001 windows before the clamp
    existed.
    """
    source = make_audio("clamp.wav", seconds=12.0)
    wav = audio_module.normalize(source, tmp_path / "n.wav", max_duration_seconds=60)

    windows = list(audio_module.iter_wav_grid(wav, window_seconds=5.0, overlap_seconds=99.0))

    # Overlap clamps to half the window, so the stride floors at 2.5 s. The window at 7.5 s
    # reaches the end of the 12 s file, so the grid stops there.
    assert [round(offset, 3) for offset, _ in windows] == [0.0, 2.5, 5.0, 7.5]


# --- De-duplication across overlapping windows --------------------------------------------


def test_authority_ranges_tile_the_recording_exactly() -> None:
    """The de-duplication rule only works if authority ranges have no gaps and no overlaps.

    This is the diarisation counterpart to ``test_windows_tile_the_audio_exactly``: it is what
    makes "keep the turn whose midpoint we own" provably keep every turn exactly once, rather
    than approximately once.
    """
    offsets = [0.0, 8.0, 16.0, 24.0]
    ranges = [
        window_authority(offsets, index, overlap_seconds=2.0, audio_end=30.0)
        for index in range(len(offsets))
    ]

    assert ranges[0][0] == 0.0
    assert ranges[-1][1] == 30.0
    for earlier, later in zip(ranges[:-1], ranges[1:], strict=True):
        assert earlier[1] == later[0]


def test_authority_ranges_sit_inside_their_own_window() -> None:
    """A window may only claim a turn it actually saw in full."""
    offsets = [0.0, 8.0, 16.0]
    window_seconds = 10.0

    for index, offset in enumerate(offsets):
        start, end = window_authority(offsets, index, overlap_seconds=2.0, audio_end=18.0)
        assert start >= offset
        assert end <= offset + window_seconds or index == len(offsets) - 1


@ffmpeg_required
def test_a_turn_seen_by_two_windows_is_kept_once(make_audio, tmp_path) -> None:
    """The cost overlapping windows reintroduce, and the rule that pays it.

    The backend here reports the *same absolute* turn from both windows that can see it, which
    is exactly what a real backend does in an overlap region.
    """
    settings = diarisation_settings()
    source = make_audio("dupe.wav", seconds=18.0)
    wav = audio_module.normalize(source, tmp_path / "n.wav", max_duration_seconds=60)

    # A turn at absolute 8.5-9.5 s falls in the overlap between window 0 (0-10 s) and
    # window 1 (8-18 s), so both windows report it.
    def turns(index, samples):
        offset = index * 8.0
        return [turn(8.5 - offset, 9.5 - offset, VOICE_A)]

    result = diarise_wav(
        wav,
        diariser=FakeDiariser(turns),
        settings=settings,
        audio_duration=18.0,
    )

    assert len(result.turns) == 1
    assert result.turns[0].start == pytest.approx(8.5)
    assert result.info.speaker_count == 1


# --- Global clustering --------------------------------------------------------------------


@ffmpeg_required
def test_the_same_voice_in_different_windows_gets_one_label(make_audio, tmp_path) -> None:
    """The problem global clustering exists to solve.

    Each window reports one turn. Windows 0 and 2 are the same person; window 1 is someone
    else. Per-window labels would call all three ``SPEAKER_00``. Clustering the embeddings
    once must find two speakers, and must give windows 0 and 2 the same label.
    """
    settings = diarisation_settings()
    # 34 s at a 10 s window and an 8 s stride is four windows: 0, 8, 16, 24.
    source = make_audio("two.wav", seconds=34.0)
    wav = audio_module.normalize(source, tmp_path / "n.wav", max_duration_seconds=60)

    voices = [VOICE_A, VOICE_B, VOICE_A_AGAIN, VOICE_B]

    def turns(index, samples):
        if index >= len(voices):
            return []
        # 1-3 s into each window, comfortably clear of the 2 s overlap region.
        return [turn(1.0, 3.0, voices[index])]

    result = diarise_wav(wav, diariser=FakeDiariser(turns), settings=settings, audio_duration=34.0)

    labels = [item.speaker for item in result.turns]
    assert len(labels) == 4
    assert result.info.speaker_count == 2
    assert labels[0] == labels[2], "the same voice in two windows must get one label"
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]
    # Numbered by first appearance, so the first voice heard is SPEAKER_00.
    assert labels[0] == "SPEAKER_00"


def test_a_monologue_stays_one_speaker() -> None:
    """Threshold-based clustering must not invent speakers out of one voice."""
    from transcription.diarisation import _cluster

    jitter = np.array(
        [[1.0, 0.01], [1.0, 0.03], [0.99, 0.0], [1.0, 0.02]],
        dtype=np.float32,
    )

    assignments = _cluster(jitter, threshold=0.5, hint=SpeakerHint())

    assert len(set(assignments)) == 1


def test_an_exact_speaker_count_overrides_the_threshold() -> None:
    """A caller who knows the room beats an unsupervised guess."""
    from transcription.diarisation import _cluster

    # Four mutually distant voices, so the threshold alone would find four.
    voices = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )

    assert len(set(_cluster(voices, threshold=0.5, hint=SpeakerHint()))) == 4
    assert len(set(_cluster(voices, threshold=0.5, hint=SpeakerHint(exact=2)))) == 2
    assert len(set(_cluster(voices, threshold=0.5, hint=SpeakerHint(exact=1)))) == 1


def test_a_speaker_range_brackets_the_threshold() -> None:
    from transcription.diarisation import _cluster

    voices = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )

    # Threshold would say four; a maximum of two forces merging past it.
    assert len(set(_cluster(voices, threshold=0.5, hint=SpeakerHint(maximum=2)))) == 2

    jitter = np.array([[1.0, 0.01], [1.0, 0.02], [0.99, 0.0]], dtype=np.float32)
    # Threshold would say one; a minimum of two stops merging short of it.
    assert len(set(_cluster(jitter, threshold=0.5, hint=SpeakerHint(minimum=2)))) == 2


def test_contradictory_speaker_hints_are_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        SpeakerHint(exact=2, minimum=1)
    with pytest.raises(ValueError, match="exceeds maximum"):
        SpeakerHint(minimum=4, maximum=2)
    with pytest.raises(ValueError, match="at least 1"):
        SpeakerHint(exact=0)


def test_a_zero_embedding_does_not_poison_every_comparison() -> None:
    """A zero vector has no direction, so its cosine distance is undefined.

    Dividing by its norm would put NaN in every comparison against it, and NaN loses every
    comparison silently — the clustering would stop merging for reasons nothing reports.
    """
    from transcription.diarisation import _cluster

    with_zero = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.01]], dtype=np.float32)

    assignments = _cluster(with_zero, threshold=0.5, hint=SpeakerHint())

    assert len(assignments) == 3
    assert all(isinstance(value, int) for value in assignments)


# --- Short turns --------------------------------------------------------------------------


@ffmpeg_required
def test_short_turns_are_labelled_from_their_neighbours(make_audio, tmp_path) -> None:
    """A turn too short to embed still gets a label, from the speech around it."""
    settings = diarisation_settings(diarisation_window_seconds=30.0)
    source = make_audio("short-turns.wav", seconds=20.0)
    wav = audio_module.normalize(source, tmp_path / "n.wav", max_duration_seconds=60)

    def turns(index, samples):
        return [
            turn(0.0, 4.0, VOICE_A),
            # 300 ms: too short to embed, so it is kept out of the clustering.
            turn(4.1, 4.4, None),
            turn(10.0, 14.0, VOICE_B),
        ]

    result = diarise_wav(wav, diariser=FakeDiariser(turns), settings=settings, audio_duration=20.0)

    labels = {(item.start, item.end): item.speaker for item in result.turns}
    assert len(result.turns) == 3
    assert result.info.speaker_count == 2
    # The short turn sits 0.1 s after voice A and 5.6 s before voice B.
    assert labels[(4.1, 4.4)] == labels[(0.0, 4.0)]


@ffmpeg_required
def test_no_speech_long_enough_to_embed_yields_no_speakers(make_audio, tmp_path) -> None:
    """An empty answer, not an error and not an invention.

    The same stance the pipeline already takes on silence.
    """
    settings = diarisation_settings(diarisation_window_seconds=30.0)
    source = make_audio("tiny.wav", seconds=5.0)
    wav = audio_module.normalize(source, tmp_path / "n.wav", max_duration_seconds=60)

    result = diarise_wav(
        wav,
        diariser=FakeDiariser(lambda index, samples: [turn(0.0, 0.2, VOICE_A)]),
        settings=settings,
        audio_duration=5.0,
    )

    assert result.turns == []
    assert result.speakers == []
    assert result.info.status == "succeeded"
    assert result.info.speaker_count == 0


# --- Alignment onto transcript segments ---------------------------------------------------


def test_segments_are_never_re_cut() -> None:
    """The published contract wins. Labels are added; nothing else moves."""
    segments = [
        Segment(id=0, start=0.0, end=10.0, text="One long segment with a turn change inside."),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=4.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=4.0, end=10.0, speaker="SPEAKER_01"),
    ]

    aligned = assign_speakers(segments, turns)

    assert len(aligned) == 1
    assert aligned[0].start == 0.0
    assert aligned[0].end == 10.0
    assert aligned[0].text == segments[0].text
    assert aligned[0].id == 0


def test_the_majority_speaker_wins_a_straddling_segment() -> None:
    segments = [Segment(id=0, start=0.0, end=10.0, text="Straddles a turn change.")]
    turns = [
        SpeakerTurn(start=0.0, end=3.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=3.0, end=10.0, speaker="SPEAKER_01"),
    ]

    aligned = assign_speakers(segments, turns)

    assert aligned[0].speaker == "SPEAKER_01"
    # 7 of 10 seconds, which is the honest signal that this one is a vote, not an answer.
    assert aligned[0].speaker_confidence == pytest.approx(0.7)


def test_a_clean_segment_reports_full_confidence() -> None:
    segments = [Segment(id=0, start=1.0, end=4.0, text="All one speaker.")]
    turns = [SpeakerTurn(start=0.0, end=5.0, speaker="SPEAKER_00")]

    aligned = assign_speakers(segments, turns)

    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].speaker_confidence == pytest.approx(1.0)


def test_word_timings_drive_the_vote_when_present() -> None:
    """Per-word assignment is the sharper path, and it must actually be taken.

    Whisper's segment edges drift by a few hundred milliseconds while word edges are DTW
    aligned, so a segment whose own boundaries suggest one answer can have words that say
    another. The words win.
    """
    words = [
        Word(start=0.2, end=0.8, word=" hello"),
        Word(start=0.9, end=1.4, word=" there"),
        Word(start=5.0, end=5.4, word=" yes"),
    ]
    segments = [Segment(id=0, start=0.0, end=6.0, text="hello there yes", words=words)]
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=4.5, end=6.0, speaker="SPEAKER_01"),
    ]

    aligned = assign_speakers(segments, turns)

    assert [word.speaker for word in aligned[0].words] == [
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    # 1.1 s of SPEAKER_00 against 0.4 s of SPEAKER_01 — the silent 3.1 s gap votes for nobody.
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].speaker_confidence == pytest.approx(1.1 / 1.5, abs=1e-3)


def test_a_segment_no_turn_covers_says_nothing() -> None:
    """The diariser and the transcriber can disagree about where speech is."""
    segments = [Segment(id=0, start=20.0, end=22.0, text="Outside every turn.")]
    turns = [SpeakerTurn(start=0.0, end=5.0, speaker="SPEAKER_00")]

    aligned = assign_speakers(segments, turns)

    assert aligned[0].speaker is None
    assert aligned[0].speaker_confidence is None


def test_overlap_is_flagged_not_resolved() -> None:
    """``speakers_overlap`` means simultaneous speech, not two speakers in sequence.

    Collapsing the two would make the flag meaningless: nearly every long segment has a turn
    change in it, and almost none have genuine crosstalk.
    """
    segments = [
        Segment(id=0, start=0.0, end=4.0, text="Crosstalk here."),
        Segment(id=1, start=5.0, end=9.0, text="Two speakers in sequence."),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=4.0, speaker="SPEAKER_00", overlap=True),
        SpeakerTurn(start=5.0, end=7.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=7.0, end=9.0, speaker="SPEAKER_01"),
    ]

    aligned = assign_speakers(segments, turns)

    assert aligned[0].speakers_overlap is True
    assert aligned[1].speakers_overlap is None
    # The sequential case is carried by a low confidence instead.
    assert aligned[1].speaker_confidence == pytest.approx(0.5)


def test_turn_blocks_group_consecutive_segments() -> None:
    segments = [
        Segment(id=0, start=0.0, end=1.0, text="A one.", speaker="SPEAKER_00"),
        Segment(id=1, start=1.0, end=2.0, text="A two.", speaker="SPEAKER_00"),
        Segment(id=2, start=2.0, end=3.0, text="B one.", speaker="SPEAKER_01"),
        Segment(id=3, start=3.0, end=4.0, text="A three.", speaker="SPEAKER_00"),
    ]

    blocks = [(speaker, len(group)) for speaker, group in iter_turn_blocks(segments)]

    assert blocks == [("SPEAKER_00", 2), ("SPEAKER_01", 1), ("SPEAKER_00", 1)]


# --- Exporters ----------------------------------------------------------------------------


def diarised_result(**overrides) -> TranscriptionResult:
    from transcription.models import AudioInfo, DiarisationInfo, ModelInfo, Speaker

    payload = {
        "id": "a" * 64,
        "audio": AudioInfo(filename="x.wav", sha256="a" * 64, size_bytes=1, duration_seconds=4.0),
        "model": ModelInfo(
            name="fake",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        ),
        "segments": [
            Segment(id=0, start=0.0, end=2.0, text="Morning.", speaker="SPEAKER_00"),
            Segment(id=1, start=2.0, end=4.0, text="Morning to you.", speaker="SPEAKER_01"),
        ],
        "speakers": [
            Speaker(id="SPEAKER_00", total_speech_seconds=2.0, turn_count=1),
            Speaker(id="SPEAKER_01", total_speech_seconds=2.0, turn_count=1),
        ],
        "diarisation": DiarisationInfo(status="succeeded", backend="fake", speaker_count=2),
        "processing_seconds": 1.0,
    }
    payload.update(overrides)
    return TranscriptionResult(**payload)


def test_vtt_uses_the_voice_span() -> None:
    """WebVTT is the one caption format with a real mechanism for this."""
    vtt = exporters.to_vtt(diarised_result())

    assert "<v SPEAKER_00>Morning.</v>" in vtt
    assert "<v SPEAKER_01>Morning to you.</v>" in vtt


def test_srt_prefixes_the_cue_text() -> None:
    """SRT has nowhere to put a speaker, so the label joins the transcript. Lossy on purpose."""
    srt = exporters.to_srt(diarised_result())

    assert "SPEAKER_00: Morning." in srt
    assert "<v" not in srt


def test_txt_merges_consecutive_segments_by_speaker() -> None:
    result = diarised_result(
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="One.", speaker="SPEAKER_00"),
            Segment(id=1, start=1.0, end=2.0, text="Two.", speaker="SPEAKER_00"),
            Segment(id=2, start=2.0, end=3.0, text="Three.", speaker="SPEAKER_01"),
        ]
    )

    txt = exporters.to_txt(result)

    assert txt == "SPEAKER_00: One. Two.\n\nSPEAKER_01: Three.\n"


def test_speaker_labels_can_be_suppressed() -> None:
    """For a caller who diarised for the JSON and wants clean captions from the same job."""
    result = diarised_result()

    assert "SPEAKER" not in exporters.to_srt(result, speaker_labels=False)
    assert "SPEAKER" not in exporters.to_vtt(result, speaker_labels=False)
    assert "SPEAKER" not in exporters.to_txt(result, speaker_labels=False)
    assert "SPEAKER" not in exporters.render(result, "vtt", speaker_labels=False)


def test_output_is_unchanged_when_nothing_is_diarised() -> None:
    """The proof that an additive feature is additive.

    Every caption consumer that exists today depends on this byte for byte.
    """
    plain = diarised_result(
        segments=[
            Segment(id=0, start=0.0, end=2.0, text="Morning."),
            Segment(id=1, start=2.0, end=4.0, text="Morning to you."),
        ],
        speakers=None,
    )

    assert exporters.to_srt(plain) == (
        "1\n00:00:00,000 --> 00:00:02,000\nMorning.\n"
        "\n2\n00:00:02,000 --> 00:00:04,000\nMorning to you.\n"
    )
    assert exporters.to_vtt(plain) == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nMorning.\n"
        "\n00:00:02.000 --> 00:00:04.000\nMorning to you.\n"
    )
    assert exporters.to_txt(plain) == "Morning. Morning to you.\n"


# --- Pipeline integration -----------------------------------------------------------------


@ffmpeg_required
def test_the_pipeline_labels_segments_end_to_end(make_audio, fake_transcriber) -> None:
    """Transcribe, diarise, align — through the real pipeline and real ffmpeg."""
    settings = diarisation_settings(diarisation_window_seconds=30.0)
    source = make_audio("meeting.wav", seconds=4.0)

    diariser = FakeDiariser(
        lambda index, samples: [
            turn(0.0, 1.5, VOICE_A),
            turn(1.5, 3.0, VOICE_B),
        ]
    )

    result = transcribe_file(
        source,
        transcriber=fake_transcriber,
        settings=settings,
        diariser=diariser,
    )

    assert result.diarisation.status == "succeeded"
    assert result.diarisation.speaker_count == 2
    assert result.diarisation.backend == "fake-diariser"
    assert [speaker.id for speaker in result.speakers] == ["SPEAKER_00", "SPEAKER_01"]
    # FakeTranscriber returns segments at 0-1.5 s and 1.5-3.0 s, matching the turns.
    assert [segment.speaker for segment in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert diariser.warm_calls == 1


@ffmpeg_required
def test_diarisation_forces_word_timestamps(make_audio, fake_transcriber) -> None:
    """Alignment is only as good as the timestamps it aligns against."""
    settings = diarisation_settings(diarisation_window_seconds=30.0)
    source = make_audio("words.wav", seconds=4.0)

    transcribe_file(
        source,
        transcriber=fake_transcriber,
        settings=settings,
        diariser=FakeDiariser(lambda index, samples: [turn(0.0, 2.0, VOICE_A)]),
        word_timestamps=False,
    )

    assert fake_transcriber.calls[0]["word_timestamps"] is True


@ffmpeg_required
def test_a_failed_diarisation_still_returns_the_transcript(make_audio, fake_transcriber) -> None:
    """The partial-success rule. Losing the expensive stage to the cheap one is the wrong trade."""
    settings = diarisation_settings(diarisation_window_seconds=30.0)
    source = make_audio("broken.wav", seconds=4.0)

    result = transcribe_file(
        source,
        transcriber=fake_transcriber,
        settings=settings,
        diariser=FakeDiariser(error=RuntimeError("embedding model died")),
    )

    assert result.segments, "the transcript must survive a diarisation failure"
    assert result.diarisation.status == "failed"
    assert result.diarisation.error_code == "diarisation_failed"
    assert result.speakers is None
    assert all(segment.speaker is None for segment in result.segments)


@ffmpeg_required
def test_a_missing_backend_fails_loudly_in_the_payload(make_audio, fake_transcriber) -> None:
    """No backend ships yet, so asking for one must say so rather than return silent nulls."""
    settings = diarisation_settings(diarisation_backend="none")
    source = make_audio("nobackend.wav", seconds=3.0)

    result = transcribe_file(source, transcriber=fake_transcriber, settings=settings)

    assert result.diarisation.status == "failed"
    assert result.diarisation.error_code == "diarisation_unavailable"
    assert result.segments


@ffmpeg_required
def test_not_requesting_diarisation_is_recorded_as_such(make_audio, fake_transcriber) -> None:
    settings = diarisation_settings(diarise=False)
    source = make_audio("plain.wav", seconds=3.0)

    result = transcribe_file(source, transcriber=fake_transcriber, settings=settings)

    assert result.diarisation.status == "not_requested"
    assert result.speakers is None
    assert all(segment.speaker is None for segment in result.segments)
    # And the existing contract is untouched: no word timings unless asked for.
    assert fake_transcriber.calls[0]["word_timestamps"] is False


def test_load_diariser_refuses_rather_than_pretending() -> None:
    with pytest.raises(DiarisationUnavailableError) as caught:
        load_diariser(Settings(diarisation_backend="none"))

    assert caught.value.code == "diarisation_unavailable"
    assert caught.value.retryable is False
    assert "DESIGN.md" in caught.value.message

    with pytest.raises(DiarisationUnavailableError):
        load_diariser(Settings(diarisation_backend="pyannote-3.1"))
