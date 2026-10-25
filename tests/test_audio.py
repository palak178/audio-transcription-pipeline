"""Tests for inspection and normalisation — the "different audio formats" requirement."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import ffmpeg_required

from transcription import audio
from transcription.errors import (
    CorruptAudioError,
    EmptyAudioError,
    NoAudioStreamError,
    NormalizationError,
)

pytestmark = ffmpeg_required


# --- Inspection -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "extra", "expected_codec"),
    [
        ("a.wav", ["-c:a", "pcm_s16le"], "pcm_s16le"),
        ("a.mp3", ["-c:a", "libmp3lame"], "mp3"),
        ("a.flac", ["-c:a", "flac"], "flac"),
        ("a.m4a", ["-c:a", "aac"], "aac"),
        ("a.ogg", ["-c:a", "libopus"], "opus"),
    ],
)
def test_inspect_identifies_each_codec(make_audio, name, extra, expected_codec) -> None:
    info = audio.inspect(make_audio(name, seconds=1.0, extra=extra))

    assert info.codec == expected_codec
    assert info.duration_seconds == pytest.approx(1.0, abs=0.2)
    assert info.channels == 1
    assert len(info.sha256) == 64


def test_inspect_trusts_content_not_extension(make_audio, tmp_path: Path) -> None:
    """A FLAC file named `.mp3` must be identified as FLAC.

    Client-supplied extensions are routinely wrong. Dispatching on them instead of on the
    probed codec is the most common way this kind of pipeline breaks.
    """
    flac = make_audio("real.flac", extra=["-c:a", "flac"])
    liar = tmp_path / "liar.mp3"
    liar.write_bytes(flac.read_bytes())

    assert audio.inspect(liar).codec == "flac"


def test_inspect_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp3"
    empty.touch()

    with pytest.raises(EmptyAudioError):
        audio.inspect(empty)


def test_inspect_rejects_non_media(tmp_path: Path) -> None:
    fake = tmp_path / "notes.mp3"
    fake.write_text("This is text, not audio.")

    with pytest.raises(CorruptAudioError):
        audio.inspect(fake)


def test_inspect_rejects_truncated_file(make_audio, tmp_path: Path) -> None:
    source = make_audio("full.mp3", seconds=2.0, extra=["-c:a", "libmp3lame"])
    truncated = tmp_path / "truncated.mp3"
    truncated.write_bytes(source.read_bytes()[:120])

    with pytest.raises(CorruptAudioError):
        audio.inspect(truncated)


def test_inspect_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CorruptAudioError):
        audio.inspect(tmp_path / "nope.wav")


def test_inspect_rejects_video_without_audio_track(tmp_path: Path) -> None:
    import subprocess

    video = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=160x120:r=10",
            "-t", "1", "-c:v", "mpeg4", "-an", str(video),
        ],
        check=True,
        capture_output=True,
    )

    with pytest.raises(NoAudioStreamError):
        audio.inspect(video)


def test_sha256_is_content_addressed(make_audio, tmp_path: Path) -> None:
    original = make_audio("a.wav", extra=["-c:a", "pcm_s16le"])
    copy = tmp_path / "copy-with-different-name.wav"
    copy.write_bytes(original.read_bytes())

    assert audio.sha256_file(original) == audio.sha256_file(copy)


# --- Normalisation --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("in.wav", ["-c:a", "pcm_s16le", "-ar", "8000"]),  # telephony
        ("in.wav", ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]),  # studio stereo
        ("in.mp3", ["-c:a", "libmp3lame"]),
        ("in.flac", ["-c:a", "flac"]),
        ("in.m4a", ["-c:a", "aac"]),
        ("in.ogg", ["-c:a", "libopus"]),
    ],
)
def test_normalize_always_produces_canonical_wav(make_audio, tmp_path: Path, name, extra) -> None:
    """Whatever goes in, 16 kHz mono 16-bit PCM comes out.

    This is the invariant the rest of the system is built on: no code after this point has
    to know anything about codecs, sample rates or channel layouts.
    """
    source = make_audio(name, seconds=1.0, channels=2, extra=extra)

    wav = audio.normalize(source, tmp_path / "out" / "normalized.wav")
    info = audio.inspect(wav)

    assert info.sample_rate == audio.TARGET_SAMPLE_RATE
    assert info.channels == audio.TARGET_CHANNELS
    assert info.codec == "pcm_s16le"


def test_normalize_extracts_audio_from_video_container(tmp_path: Path) -> None:
    import subprocess

    video = tmp_path / "with_audio.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=160x120:r=10",
            "-f", "lavfi", "-i", "sine=frequency=440:r=16000",
            "-t", "1", "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(video),
        ],
        check=True,
        capture_output=True,
    )

    wav = audio.normalize(video, tmp_path / "out.wav")

    assert audio.probe_duration(wav) == pytest.approx(1.0, abs=0.3)


def test_normalize_caps_duration_during_decode(make_audio, tmp_path: Path) -> None:
    """ffmpeg's own `-t` bound stops a long decode from ever being written.

    This is the guard against a small compressed file that expands into hours of audio.
    """
    source = make_audio("long.wav", seconds=10.0, extra=["-c:a", "pcm_s16le"])

    wav = audio.normalize(source, tmp_path / "out.wav", max_duration_seconds=2.0)

    # 2s limit plus the deliberate 1s margin.
    assert audio.probe_duration(wav) <= 3.1


def test_normalize_rejects_undecodable_input(tmp_path: Path) -> None:
    fake = tmp_path / "broken.wav"
    fake.write_bytes(b"RIFFbogus" + b"\x00" * 64)

    with pytest.raises(NormalizationError):
        audio.normalize(fake, tmp_path / "out.wav")


def test_normalize_gives_up_on_a_stalled_decoder(make_audio, tmp_path: Path, monkeypatch) -> None:
    """A decoder that never finishes must not hold an API concurrency slot indefinitely.

    ffprobe already had a timeout and ffmpeg did not, which is the asymmetry this closes.
    """
    monkeypatch.setattr(audio, "_DECODE_TIMEOUT_SECONDS", 0.001)
    source = make_audio("in.wav", seconds=2.0, extra=["-c:a", "pcm_s16le"])

    with pytest.raises(NormalizationError, match="did not finish decoding"):
        audio.normalize(source, tmp_path / "out.wav")


def test_read_wav_mono_f32_returns_normalised_samples(make_audio, tmp_path: Path) -> None:
    source = make_audio("in.wav", seconds=1.0, extra=["-c:a", "pcm_s16le"])
    wav = audio.normalize(source, tmp_path / "out.wav")

    samples = audio.read_wav_mono_f32(wav)

    assert samples.dtype.name == "float32"
    assert len(samples) == pytest.approx(audio.TARGET_SAMPLE_RATE, rel=0.1)
    assert abs(float(samples.max())) <= 1.0


def test_read_wav_rejects_unexpected_shape(make_audio) -> None:
    """Defends the contract between normalize() and the engine."""
    stereo = make_audio("stereo.wav", channels=2, extra=["-c:a", "pcm_s16le", "-ar", "44100"])

    with pytest.raises(NormalizationError):
        audio.read_wav_mono_f32(stereo)


# --- Windowed reading (the long-audio memory control) ---------------------------------


def _wav_from_pattern(path: Path, pattern: list[tuple[float, float]]) -> None:
    """Write a 16 kHz mono WAV from (seconds, amplitude) runs."""
    import wave

    import numpy as np

    chunks = [
        np.full(int(seconds * audio.TARGET_SAMPLE_RATE), amplitude, dtype=np.float32)
        for seconds, amplitude in pattern
    ]
    samples = np.concatenate(chunks)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio.TARGET_SAMPLE_RATE)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())


def _windows(path: Path, **kwargs):
    defaults = {"window_seconds": 2.0, "search_seconds": 0.5}
    return list(audio.iter_wav_windows(path, **{**defaults, **kwargs}))


def test_short_audio_is_a_single_window(make_audio, tmp_path: Path) -> None:
    """Below one window there are no seams at all, and no special-casing to get there."""
    source = make_audio("in.wav", seconds=1.0, extra=["-c:a", "pcm_s16le"])
    wav = audio.normalize(source, tmp_path / "out.wav")

    windows = _windows(wav, window_seconds=10.0)

    assert len(windows) == 1
    assert windows[0][0] == 0.0
    assert len(windows[0][1]) == pytest.approx(audio.TARGET_SAMPLE_RATE, rel=0.05)


def test_long_audio_is_split_into_windows(make_audio, tmp_path: Path) -> None:
    source = make_audio("in.wav", seconds=9.0, extra=["-c:a", "pcm_s16le"])
    wav = audio.normalize(source, tmp_path / "out.wav")

    windows = _windows(wav, window_seconds=2.0)

    assert len(windows) >= 4
    for _, samples in windows:
        assert len(samples) <= 2.0 * audio.TARGET_SAMPLE_RATE


def test_windows_tile_the_audio_exactly(tmp_path: Path) -> None:
    """No gaps and no overlaps: offsets are contiguous and the lengths sum to the whole.

    This is what makes a plain timestamp shift correct — anything else would need overlap
    reconciliation.
    """
    wav = tmp_path / "pattern.wav"
    _wav_from_pattern(wav, [(1.0, 0.5), (0.5, 0.0)] * 6)  # 9 seconds of speech/silence
    total = len(audio.read_wav_mono_f32(wav))

    windows = _windows(wav, window_seconds=2.0)

    position = 0
    for offset, samples in windows:
        assert offset == pytest.approx(position / audio.TARGET_SAMPLE_RATE, abs=1e-6)
        position += len(samples)
    assert position == total


def test_window_boundaries_land_in_silence(tmp_path: Path) -> None:
    """The point of searching for a quiet frame: cuts must not fall inside speech.

    The pattern alternates 0.4 s of tone with 0.4 s of silence. Since a cut shortens the
    window, boundaries drift relative to the pattern — so silence has to be frequent enough
    that *any* 0.6 s search band contains some, whatever the phase.
    """
    wav = tmp_path / "pattern.wav"
    _wav_from_pattern(wav, [(0.4, 0.8), (0.4, 0.0)] * 15)  # 12 seconds

    windows = _windows(wav, window_seconds=2.0, search_seconds=0.6)

    samples = audio.read_wav_mono_f32(wav)
    boundaries = []
    position = 0
    for _, window in windows[:-1]:
        position += len(window)
        boundaries.append(position)

    assert boundaries, "expected at least one interior boundary"
    for boundary in boundaries:
        assert abs(float(samples[boundary])) < 0.05, f"cut at {boundary} is inside speech"


def test_windowing_handles_audio_with_no_silence(tmp_path: Path) -> None:
    """Continuous speech still has to be windowed; it just cuts at the quietest point."""
    wav = tmp_path / "loud.wav"
    _wav_from_pattern(wav, [(6.0, 0.7)])

    windows = _windows(wav, window_seconds=2.0)

    assert len(windows) >= 3
    assert sum(len(s) for _, s in windows) == len(audio.read_wav_mono_f32(wav))


def test_search_band_cannot_exceed_half_the_window(tmp_path: Path) -> None:
    """Guards against boundaries marching backwards and looping forever."""
    wav = tmp_path / "pattern.wav"
    _wav_from_pattern(wav, [(1.0, 0.4)] * 6)

    windows = _windows(wav, window_seconds=2.0, search_seconds=60.0)

    assert sum(len(s) for _, s in windows) == len(audio.read_wav_mono_f32(wav))
    assert all(len(s) > 0 for _, s in windows)


def test_windowing_rejects_non_canonical_wav(make_audio) -> None:
    stereo = make_audio("stereo.wav", channels=2, extra=["-c:a", "pcm_s16le", "-ar", "44100"])

    with pytest.raises(NormalizationError):
        _windows(stereo)


def test_tiny_window_is_floored_to_one_second(tmp_path: Path) -> None:
    """A misconfigured window must not degenerate into frame-by-frame iteration."""
    wav = tmp_path / "pattern.wav"
    _wav_from_pattern(wav, [(4.0, 0.4)])

    windows = _windows(wav, window_seconds=0.001, search_seconds=0.5)

    # Each window still carries a meaningful span of audio rather than a handful of frames.
    assert all(len(samples) > 0.4 * audio.TARGET_SAMPLE_RATE for _, samples in windows)
    assert len(windows) <= 10
    assert sum(len(s) for _, s in windows) == len(audio.read_wav_mono_f32(wav))
