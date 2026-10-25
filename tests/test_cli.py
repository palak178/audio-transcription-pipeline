"""Tests for the command-line interface.

These cover argument handling and the `inspect` command, neither of which needs a model.
Transcription itself is exercised through the pipeline and integration tests.
"""

from __future__ import annotations

import json

import pytest
from tests.conftest import ffmpeg_required
from typer.testing import CliRunner

from transcription.cli import DEFAULT_COMMAND, app, expand_default_command

runner = CliRunner()


# --- Implied default subcommand -------------------------------------------------------
#
# Regression guard. A multi-command CLI parses its first argument as a subcommand name, so
# without this expansion `transcribe audio.mp3` fails with "No such command 'audio.mp3'" and
# every documented example is wrong.


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        # A path gets the default command prepended.
        (["audio.mp3"], ["transcribe", "audio.mp3"]),
        (["audio.mp3", "--format", "srt"], ["transcribe", "audio.mp3", "--format", "srt"]),
        (["dir/with spaces/clip.wav"], ["transcribe", "dir/with spaces/clip.wav"]),
        # Explicit subcommands are left exactly as they are.
        (["transcribe", "audio.mp3"], ["transcribe", "audio.mp3"]),
        (["inspect", "audio.mp3"], ["inspect", "audio.mp3"]),
        (["warm"], ["warm"]),
        # Options and empty input must reach the parser untouched, so --help and
        # no-args-shows-help keep working.
        (["--help"], ["--help"]),
        (["--bogus"], ["--bogus"]),
        ([], []),
    ],
)
def test_expand_default_command(argv, expected) -> None:
    assert expand_default_command(argv) == expected


def test_default_command_is_registered() -> None:
    """The expansion is worthless if it inserts a command that does not exist."""
    from typer.main import get_command

    assert DEFAULT_COMMAND in get_command(app).commands


def test_expansion_does_not_mutate_its_input() -> None:
    argv = ["audio.mp3"]

    expand_default_command(argv)

    assert argv == ["audio.mp3"]


# --- The inspect command --------------------------------------------------------------


@ffmpeg_required
def test_inspect_reports_real_codec(make_audio) -> None:
    source = make_audio("real.flac", extra=["-c:a", "flac"])

    result = runner.invoke(app, expand_default_command(["inspect", str(source)]))

    assert result.exit_code == 0
    assert json.loads(result.stdout)["codec"] == "flac"


@ffmpeg_required
def test_inspect_reports_a_typed_error_and_fails(tmp_path) -> None:
    broken = tmp_path / "notes.mp3"
    broken.write_text("not audio")

    result = runner.invoke(app, expand_default_command(["inspect", str(broken)]))

    assert result.exit_code == 1
    assert "corrupt_audio" in result.output


def test_unknown_format_is_rejected_before_any_work(make_audio) -> None:
    result = runner.invoke(app, expand_default_command(["clip.mp3", "--format", "docx"]))

    assert result.exit_code == 2
    assert "unknown format" in result.output
