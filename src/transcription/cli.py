"""Command-line interface.

The CLI is the primary entry point for long files: it has no request timeout, so a
three-hour recording is simply a matter of waiting, with progress printed as it goes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from transcription import audio as audio_module
from transcription import exporters
from transcription.config import get_settings
from transcription.errors import TranscriptionError
from transcription.models import TranscriptionResult
from transcription.pipeline import transcribe_file
from transcription.transcriber import FasterWhisperTranscriber

app = typer.Typer(
    add_completion=False,
    help="Transcribe audio to timestamped text.",
    no_args_is_help=True,
)

DEFAULT_COMMAND = "transcribe"


def _echo_error(exc: TranscriptionError) -> None:
    typer.secho(f"error [{exc.code}]: {exc.message}", fg=typer.colors.RED, err=True)
    for key, value in exc.details.items():
        typer.secho(f"  {key}: {value}", fg=typer.colors.RED, err=True)


@app.command()
def transcribe(
    file: Annotated[Path, typer.Argument(help="Audio file in any ffmpeg-readable format.")],
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="json, srt, vtt, txt, or all."),
    ] = "json",
    out_dir: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write files to this directory instead of stdout."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Whisper model size, e.g. tiny, base, small."),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Force a language code; omit to auto-detect."),
    ] = None,
    word_timestamps: Annotated[
        bool,
        typer.Option("--word-timestamps", help="Also emit per-word timings (~10-25% slower)."),
    ] = False,
    max_duration: Annotated[
        float | None,
        typer.Option("--max-duration", help="Reject audio longer than this many seconds."),
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress progress.")] = False,
) -> None:
    """Transcribe an audio file."""
    formats = list(exporters.FORMATS) if output_format == "all" else [output_format]
    for fmt in formats:
        if fmt not in exporters.FORMATS:
            typer.secho(
                f"error: unknown format {fmt!r}. Choose from: "
                f"{', '.join(exporters.FORMATS)}, all",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

    settings = get_settings()
    transcriber = FasterWhisperTranscriber(settings, model_name=model)

    # Progress goes to stderr so that `transcribe x.mp3 > out.json` stays clean.
    def report(done: float, total: float) -> None:
        if total <= 0:
            print(f"\r  transcribed {done:.0f}s", end="", file=sys.stderr, flush=True)
            return
        pct = min(100.0, done / total * 100)
        bar = "#" * int(pct // 4)
        print(f"\r  [{bar:<25}] {pct:5.1f}%", end="", file=sys.stderr, flush=True)

    if not quiet:
        typer.secho(f"Transcribing {file} with model `{transcriber.name}`...", err=True)

    try:
        result = transcribe_file(
            file,
            transcriber=transcriber,
            settings=settings,
            language=language,
            word_timestamps=word_timestamps,
            progress=None if quiet else report,
            max_duration_seconds=max_duration,
        )
    except TranscriptionError as exc:
        if not quiet:
            print(file=sys.stderr)
        _echo_error(exc)
        raise typer.Exit(code=1) from exc

    if not quiet:
        print(file=sys.stderr)
        _echo_summary(result)

    if out_dir is None:
        for index, fmt in enumerate(formats):
            if index:
                typer.echo()
            typer.echo(exporters.render(result, fmt))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.audio.filename).stem or "transcript"
    for fmt in formats:
        destination = out_dir / f"{stem}{exporters.extension_for(fmt)}"
        destination.write_text(exporters.render(result, fmt), encoding="utf-8")
        typer.secho(f"  wrote {destination}", fg=typer.colors.GREEN, err=True)


@app.command()
def inspect(
    file: Annotated[Path, typer.Argument(help="Audio file to probe.")],
) -> None:
    """Report what a file really is, according to ffprobe rather than its extension."""
    try:
        info = audio_module.inspect(file)
    except TranscriptionError as exc:
        _echo_error(exc)
        raise typer.Exit(code=1) from exc
    typer.echo(info.model_dump_json(indent=2))


@app.command()
def warm() -> None:
    """Download and load the configured model, so the first real run is fast."""
    settings = get_settings()
    typer.secho(f"Loading model `{settings.whisper_model}`...", err=True)
    FasterWhisperTranscriber(settings).warm()
    typer.secho("Model ready.", fg=typer.colors.GREEN, err=True)


def _echo_summary(result: TranscriptionResult) -> None:
    rtf = result.real_time_factor
    typer.secho(
        f"  {len(result.segments)} segments | "
        f"{result.audio.duration_seconds:.1f}s audio | "
        f"{result.processing_seconds:.1f}s compute"
        + (f" | RTF {rtf:.2f}" if rtf else "")
        + f" | language {result.language or 'unknown'}"
        + (
            f" ({result.language_probability:.0%})"
            if result.language_probability is not None
            else ""
        ),
        fg=typer.colors.CYAN,
        err=True,
    )


def _known_commands() -> set[str]:
    """Names of the registered subcommands, read from the generated Click group."""
    from typer.main import get_command

    return set(getattr(get_command(app), "commands", {}))


def expand_default_command(args: list[str]) -> list[str]:
    """Insert the default subcommand when the first argument is clearly not one.

    A multi-command CLI parses its first argument as a subcommand name, which would force
    users to write ``transcribe transcribe audio.mp3``. This makes the common case read the
    way anyone would expect — ``transcribe audio.mp3`` — while leaving ``transcribe inspect``
    and an explicit ``transcribe transcribe`` working untouched.

    Doing it on the argument list keeps it independent of Click internals: Typer vendors its
    own copy of Click, so subclassing the command group and catching ``click.UsageError``
    silently fails to match the vendored exception type.

    A file literally named ``inspect`` or ``warm`` is the one ambiguity; pass such a path
    explicitly as ``transcribe transcribe ./inspect``.
    """
    if not args or args[0].startswith("-"):
        return args
    if args[0] in _known_commands():
        return args
    return [DEFAULT_COMMAND, *args]


@app.command()
def worker(
    latency_class: Annotated[
        str | None,
        typer.Option(
            "--latency-class",
            help="Only take `short` or `long` jobs. Omit to take either.",
        ),
    ] = None,
    migrate: Annotated[
        bool, typer.Option("--migrate/--no-migrate", help="Apply migrations on startup.")
    ] = True,
) -> None:
    """Run a transcription worker: claim jobs from the queue and transcribe them.

    This is the process that does the slow work. Run as many as you have cores to spare;
    they coordinate through Postgres, so nothing needs to know how many there are.

    Stops gracefully on SIGTERM or Ctrl-C — it finishes the job in flight first, so a routine
    deploy does not throw away minutes of transcription.
    """
    from transcription.worker import run as run_worker

    if latency_class not in (None, "short", "long"):
        typer.secho(
            f"error: --latency-class must be `short` or `long`, got {latency_class!r}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    run_worker(latency_class=latency_class, migrate=migrate)


apikey_app = typer.Typer(help="Manage per-tenant API keys for the job API.")
app.add_typer(apikey_app, name="apikey")


@apikey_app.command("create")
def apikey_create(
    tenant: Annotated[
        str | None,
        typer.Option("--tenant", help="Tenant UUID. Omit to generate a fresh tenant."),
    ] = None,
    name: Annotated[
        str, typer.Option("--name", help="Label, e.g. who or what holds it.")
    ] = "default",
) -> None:
    """Mint an API key and print it once.

    The secret is shown here and nowhere else — only its SHA-256 digest is stored, so a lost
    key is replaced rather than recovered. That is the property that makes a stolen database
    dump useless.
    """
    import uuid as _uuid

    from transcription import auth, db
    from transcription.config import Settings

    settings = Settings()
    try:
        tenant_id = _uuid.UUID(tenant) if tenant else _uuid.uuid4()
    except ValueError:
        typer.secho(
            f"error: --tenant must be a UUID, got {tenant!r}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2) from None

    db.migrate(settings)
    plaintext, record = auth.create_api_key(settings, tenant_id=tenant_id, name=name)

    typer.secho(
        "API key created. This is the only time it is shown.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    typer.echo(f"tenant: {record.tenant_id}", err=True)
    typer.echo(f"name:   {record.name}", err=True)
    typer.echo(plaintext)


@apikey_app.command("list")
def apikey_list(
    tenant: Annotated[
        str | None, typer.Option("--tenant", help="Filter by tenant UUID.")
    ] = None,
) -> None:
    """List keys. Shows prefixes and usage, never secrets."""
    import uuid as _uuid

    from transcription import auth, db
    from transcription.config import Settings

    settings = Settings()
    db.migrate(settings)
    tenant_id = _uuid.UUID(tenant) if tenant else None

    keys = auth.list_api_keys(settings, tenant_id=tenant_id)
    if not keys:
        typer.echo("no keys", err=True)
        return
    for key in keys:
        state = "revoked" if key.revoked_at else "active"
        used = key.last_used_at.isoformat() if key.last_used_at else "never"
        typer.echo(
            f"{key.id}  {key.key_prefix}...  {state:7}  "
            f"{key.tenant_id}  used={used}  {key.name}"
        )


@apikey_app.command("revoke")
def apikey_revoke(
    key_id: Annotated[str, typer.Argument(help="Key id from `apikey list`.")],
) -> None:
    """Revoke a key. Recorded as a timestamp rather than deleted, so there is an audit trail."""
    import uuid as _uuid

    from transcription import auth, db
    from transcription.config import Settings

    settings = Settings()
    db.migrate(settings)
    try:
        parsed = _uuid.UUID(key_id)
    except ValueError:
        typer.secho(f"error: not a key id: {key_id!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None

    if auth.revoke(settings, key_id=parsed):
        typer.secho(f"revoked {parsed}", fg=typer.colors.GREEN, err=True)
    else:
        typer.secho(f"no active key {parsed}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)



def main(argv: list[str] | None = None) -> None:
    """Console-script entry point."""
    app(expand_default_command(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":  # pragma: no cover
    main()
