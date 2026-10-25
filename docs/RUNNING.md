# Running and testing, step by step

Covers **macOS, Linux and Windows**, natively or in Docker.

| Part | What it covers |
|---|---|
| [Part A](#part-a--run-it-without-docker) | Run it without Docker — macOS, Linux, Windows |
| [Part B](#part-b--exercise-every-feature) | Exercise every feature: CLI, formats, long audio, HTTP API |
| [Part C](#part-c--run-the-test-suite) | Run the test suite |
| [Part D](#part-d--run-it-with-docker) | Run it with Docker (any platform) |
| [Troubleshooting](#troubleshooting) | Symptoms and fixes, including platform-specific ones |

> **Verification note, so you know what to trust:** everything here was executed on
> **macOS (Apple Silicon)**. The Linux and Windows commands are the standard equivalents but
> were *not* run on this machine. The code itself is platform-neutral — no shell invocations,
> no POSIX-only imports, and ffmpeg is located with `shutil.which`, which resolves `.exe` on
> Windows. The two places Windows genuinely differs are called out inline: the **Makefile**
> and **`scripts/make_fixtures.sh`** both need a POSIX shell.

Run every command from the repository root.

---

## Part A — Run it without Docker

### Step 1: Install ffmpeg

The one hard dependency. It *is* the format-handling layer — nothing works without it.

**macOS**
```bash
brew install ffmpeg
```

**Linux**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg   # Debian / Ubuntu
sudo dnf install -y ffmpeg                              # Fedora / RHEL (may need RPM Fusion)
sudo pacman -S ffmpeg                                   # Arch
```

**Windows** (PowerShell)
```powershell
winget install Gyan.FFmpeg
# or: choco install ffmpeg
# or: scoop install ffmpeg
```
Then **close and reopen the terminal** so the updated `PATH` takes effect.

Verify on any platform — both binaries are needed:
```bash
ffmpeg -version
ffprobe -version
```

### Step 2: Check your Python version

Python **3.11 or newer** is required.

```bash
python3 --version        # macOS / Linux
python --version         # Windows
```

- **macOS:** the system `/usr/bin/python3` is often 3.9, which is too old. Use
  `brew install python@3.12` and call it `python3.12`.
- **Linux (Ubuntu 22.04 or older):** the default is 3.10. Install a newer one:
  ```bash
  sudo apt-get install -y python3.12 python3.12-venv
  ```
  On Ubuntu 22.04 that may need `sudo add-apt-repository ppa:deadsnakes/ppa` first.
  **The `python3.x-venv` package matters** — without it, `python3.12 -m venv` fails.
- **Windows:** `winget install Python.Python.3.12`, or the installer from python.org with
  *"Add python.exe to PATH"* ticked.

### Step 3: Create the environment and install

Pick **one** option. Both give the same result.

#### Option A — with [uv](https://docs.astral.sh/uv/) (fastest, ~15 s)

**macOS / Linux**
```bash
make setup
```

**Windows** (no `make`, so run the two commands directly)
```powershell
uv venv --python 3.12
uv pip install -e ".[dev]"
```

<details>
<summary>What <code>make setup</code> does</summary>

```bash
uv venv --python 3.12        # create .venv
uv pip install -e ".[dev]"   # install the project plus test tooling
```
It also checks ffmpeg is present and stops with a clear message if it isn't.
</details>

#### Option B — with plain `venv` and `pip` (no extra tools, ~2 min)

**macOS / Linux**
```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

**Windows** (PowerShell)
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

> If a previous attempt left a broken environment, delete it and start again:
> `rm -rf .venv` (macOS/Linux) or `Remove-Item -Recurse -Force .venv` (Windows).

### Step 4: Activate the environment

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows — PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```
If that is blocked by execution policy:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**Windows — Command Prompt**
```cmd
.venv\Scripts\activate.bat
```

Confirm the CLI is available:
```bash
transcribe --help
```

> Prefer not to activate? Prefix commands with `.venv/bin/` (macOS/Linux) or
> `.venv\Scripts\` (Windows). The `make` targets never need activation.

### Step 5: Download the model

The first transcription downloads weights (~74 MB for the default `base`). Do it deliberately
now so it doesn't look like a hang mid-demo:

```bash
transcribe warm
```
or `make warm` on macOS/Linux. Needs internet **once**; everything afterwards runs offline.

### Step 6: The 30-second check

```bash
transcribe samples/speech_en.mp3
```

Expected — a progress bar and summary on stderr, JSON on stdout:

```
Transcribing samples/speech_en.mp3 with model `base`...
  [#########################] 100.0%
  2 segments | 7.2s audio | 0.6s compute | RTF 0.09 | language en (100%)
{
  "schema_version": "1.0",
  "id": "33b72396207f5be6...",
  "language": "en",
  ...
}
```

**If you see that, the whole pipeline works.** `RTF 0.09` means roughly eleven times faster
than real time, on CPU.

### Command equivalents on Windows

There's no `make` on Windows. Every target maps to one command:

| `make` target | Windows equivalent |
|---|---|
| `make setup` | `uv venv --python 3.12` then `uv pip install -e ".[dev]"` |
| `make warm` | `transcribe warm` |
| `make demo` | `transcribe samples\speech_en.mp3 --format json` |
| `make test` | `python -m pytest` |
| `make test-all` | `python -m pytest -m "slow or not slow"` |
| `make lint` | `python -m ruff check src tests` |
| `make serve` | `python -m uvicorn transcription.api:app --host 0.0.0.0 --port 8000` |
| `make fixtures` | Needs a POSIX shell — see [Step 8](#step-8-generate-the-test-audio-set) |
| `make clean` | `Remove-Item -Recurse -Force out, samples\generated, .pytest_cache, .ruff_cache` |

---

## Part B — Exercise every feature

### Step 7: Output formats and options

```bash
transcribe samples/speech_en.mp3 --format srt      # subtitles
transcribe samples/speech_en.mp3 --format vtt      # WebVTT
transcribe samples/speech_en.mp3 --format txt      # plain prose
transcribe samples/speech_en.mp3 --format all --out out/   # all four, written to out/
```

```bash
transcribe samples/speech_en.mp3 --word-timestamps   # per-word timings (~10-25% slower)
transcribe samples/speech_en.mp3 --model tiny        # faster, less accurate
transcribe samples/speech_en.mp3 --language en       # skip language detection
transcribe samples/speech_en.mp3 --quiet             # suppress progress
transcribe samples/speech_en.mp3 > result.json       # progress goes to stderr, so this stays clean
```

### Step 8: Generate the test audio set

This builds a dozen awkward inputs from the one committed recording — different codecs, odd
sample rates, and deliberately broken files.

**macOS / Linux**
```bash
make fixtures          # or: ./scripts/make_fixtures.sh
ls samples/generated/
```

**Windows** — the script is bash, so use one of:
- **Git Bash** (ships with Git for Windows): `bash scripts/make_fixtures.sh`
- **WSL**: `wsl bash scripts/make_fixtures.sh`
- Or skip it — the fixtures are only needed for the manual checks in Steps 9-10. The test
  suite generates its own audio and does **not** depend on this script.

### Step 9: Prove the format handling works

**Every format yields the same transcript:**

```bash
transcribe samples/generated/speech.flac --format txt --quiet
transcribe samples/generated/speech.m4a --format txt --quiet
transcribe samples/generated/speech.ogg --format txt --quiet
transcribe samples/generated/speech_8k_telephony.wav --format txt --quiet
transcribe samples/generated/speech_48k_stereo.wav --format txt --quiet
transcribe samples/generated/video_with_audio.mp4 --format txt --quiet
```

**The interesting one** — a FLAC file deliberately misnamed `.mp3`:

```bash
transcribe inspect samples/generated/lying_extension.mp3
# -> "codec": "flac"    <- content wins over the extension

transcribe samples/generated/lying_extension.mp3 --format txt --quiet
# -> transcribes correctly anyway
```

**Failure cases.** Each gives a typed error and **exit code 1**, not a stack trace:

| Command | Expected |
|---|---|
| `transcribe samples/generated/empty.mp3` | `error [empty_audio]` |
| `transcribe samples/generated/not_audio.mp3` | `error [corrupt_audio]` |
| `transcribe samples/generated/truncated.mp3` | `error [corrupt_audio]` |
| `transcribe samples/generated/no_audio_track.mp4` | `error [no_audio_stream]` |

Check the exit code with `echo $?` (macOS/Linux) or `echo $LASTEXITCODE` (PowerShell).

**Silence and non-speech** — an empty transcript at exit 0, not invented text:

```bash
transcribe samples/generated/silence.wav --format txt      # empty
transcribe samples/generated/tone.wav --format txt         # empty
transcribe samples/generated/speech_then_long_silence.wav --format txt
#   ^ 7s speech + 40s silence. No repeated "thank you for watching" loop.
```

### Step 10: Prove the long-audio handling works

```bash
transcribe samples/generated/long_6min.mp3 --format txt --out out/
```

Watch the progress bar — that's the progress callback firing as each window's segments are
decoded. At RTF ~0.2 a 6-minute file takes roughly 70 seconds.

Peak memory tracks the **window size**, not the file length. See it directly:

```bash
# macOS / Linux
TRANSCRIBE_WINDOW_SECONDS=60  transcribe samples/generated/long_6min.mp3 --format txt --quiet
TRANSCRIBE_WINDOW_SECONDS=900 transcribe samples/generated/long_6min.mp3 --format txt --quiet
```

```powershell
# Windows (PowerShell)
$env:TRANSCRIBE_WINDOW_SECONDS=60;  transcribe samples\generated\long_6min.mp3 --format txt --quiet
$env:TRANSCRIBE_WINDOW_SECONDS=900; transcribe samples\generated\long_6min.mp3 --format txt --quiet
Remove-Item Env:\TRANSCRIBE_WINDOW_SECONDS
```

To measure peak memory:

```bash
/usr/bin/time -l transcribe samples/generated/long_6min.mp3 --format txt --quiet   # macOS: "maximum resident set size" (bytes)
/usr/bin/time -v transcribe samples/generated/long_6min.mp3 --format txt --quiet   # Linux: "Maximum resident set size (kbytes)"
```

```powershell
# Windows
Measure-Command { transcribe samples\generated\long_6min.mp3 --format txt --quiet }
# peak memory: watch the python process in Task Manager, or use Get-Process python
```

Measured on this machine, 12-minute file: **733 MB** at a 60 s window versus **1660 MB** at
900 s (where the window exceeds the file, so it is effectively a single pass). A full hour of
audio stays at **999 MB** with the default 300 s window.

### Step 11: Run the HTTP API

**Terminal 1** — start the server:

```bash
make serve                                                      # macOS / Linux
python -m uvicorn transcription.api:app --host 0.0.0.0 --port 8000   # any platform
```

Wait for `Application startup complete.` — it loads the model first.

**Terminal 2** — call it. On **Windows use `curl.exe`**, because in PowerShell bare `curl` is
an alias for `Invoke-WebRequest` and will not accept `-F`:

```bash
# health
curl -s localhost:8000/healthz

# transcribe
curl -s -F file=@samples/speech_en.mp3 localhost:8000/v1/transcriptions

# subtitles straight out of the API
curl -s -F file=@samples/speech_en.mp3 -F response_format=srt localhost:8000/v1/transcriptions

# per-word timings
curl -s -F file=@samples/speech_en.mp3 -F word_timestamps=true localhost:8000/v1/transcriptions

# force a language
curl -s -F file=@samples/speech_en.mp3 -F language=en localhost:8000/v1/transcriptions
```

```powershell
# Windows equivalents
curl.exe -s -F file=@samples/speech_en.mp3 localhost:8000/v1/transcriptions
curl.exe -s -F file=@samples/speech_en.mp3 -F response_format=srt localhost:8000/v1/transcriptions
```

Pretty-print by piping to `python -m json.tool` (or `jq` if installed).

**Error handling** — check the status codes:

```bash
curl -s -w '\n[HTTP %{http_code}]\n' -F file=@samples/generated/not_audio.mp3      localhost:8000/v1/transcriptions   # 415
curl -s -w '\n[HTTP %{http_code}]\n' -F file=@samples/generated/empty.mp3          localhost:8000/v1/transcriptions   # 400
curl -s -w '\n[HTTP %{http_code}]\n' -F file=@samples/generated/no_audio_track.mp4 localhost:8000/v1/transcriptions   # 415
curl -s -w '\n[HTTP %{http_code}]\n' localhost:8000/v1/transcriptions                                                 # 422 (no file)
```

**The documented duration limit** — the synchronous endpoint refuses long audio and says what
to do instead. Restart the server with a 5-second cap rather than waiting on a long file:

```bash
TRANSCRIBE_MAX_SYNC_DURATION_SECONDS=5 make serve                          # macOS / Linux
```
```powershell
$env:TRANSCRIBE_MAX_SYNC_DURATION_SECONDS=5; python -m uvicorn transcription.api:app --port 8000
```
Then:
```bash
curl -s -F file=@samples/generated/long_6min.mp3 localhost:8000/v1/transcriptions
# -> 413 audio_too_long, with a hint pointing at the CLI and docs/DESIGN.md
```

**Concurrency** — six uploads at once; all succeed, and all return the same `id`, because the
id is a hash of the audio:

```bash
for i in $(seq 1 6); do
  curl -s -F file=@samples/speech_en.mp3 localhost:8000/v1/transcriptions \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['id'][:12])" &
done; wait
```

```powershell
# Windows, PowerShell 7+ (ForEach-Object -Parallel does not exist in PowerShell 5.1)
1..6 | ForEach-Object -Parallel { curl.exe -s -F file=@samples/speech_en.mp3 localhost:8000/v1/transcriptions }

# Windows, any PowerShell version
1..6 | ForEach-Object { Start-Job { curl.exe -s -F file=@samples/speech_en.mp3 localhost:8000/v1/transcriptions } } | Wait-Job | Receive-Job
```

**Interactive API docs — the easiest way to test it.** Open <http://localhost:8000/docs>,
expand **POST /v1/transcriptions**, click **Try it out**, choose an audio file and press
**Execute**. No curl required. `response_format` is a dropdown, every field has help text,
and all the error codes are listed with examples. `/` redirects here; `/redoc` is the same
spec in a reference layout.

Stop the server with `Ctrl-C`.

---

## Part C — Run the test suite

Works identically on all three platforms.

### The fast suite

```bash
make test            # macOS / Linux
python -m pytest     # any platform
```

Expected: **130 passed** in a few seconds. No model weights needed — the engine is faked,
while ffmpeg, the pipeline, the windowing and the real ASGI stack are all exercised for real.

### The full suite

```bash
make test-all                              # macOS / Linux
python -m pytest -m "slow or not slow"     # any platform
```

Expected: **140 passed** in ~10 seconds once the weights are cached. The extra ten tests run a
real `tiny` Whisper model, covering what a fake cannot: real word alignment, real behaviour on
silence, and a long file crossing real window boundaries.

> **The first run is much slower than ten seconds.** It downloads ~40 MB of weights from
> Hugging Face, and on a slow or rate-limited connection that has taken ten minutes or more
> here. It is not hung. Run `transcribe warm` first if you would rather see download progress
> than a silent pytest.

### Useful variations

```bash
python -m pytest -v                  # verbose, test by test
python -m pytest tests/test_audio.py # one file
python -m pytest -k "window"         # just the long-audio logic
python -m pytest -k "format"         # just format handling
python -m pytest -m slow             # only the real-model tests
python -m pytest -x                  # stop at the first failure
```

### Linting

```bash
make lint                            # macOS / Linux
python -m ruff check src tests       # any platform
```

Expected: `All checks passed!`

### What the tests cover

| File | Count | What it checks |
|---|---|---|
| `tests/test_audio.py` | 32 | Format detection, lying extensions, corrupt/empty files, normalisation to 16 kHz mono, decode timeout, windowed reading, silence-aligned cuts |
| `tests/test_pipeline.py` | 31 | Orchestration, content-hash ids, duration limits, progress, multi-window stitching, filename sanitisation |
| `tests/test_exporters.py` | 19 | SRT/VTT timecodes incl. past one hour, blank cues, unicode, the JSON contract |
| `tests/test_api.py` | 34 | Status codes, response formats, `Server-Timing` on every format and on errors, oversized uploads (incl. rejection before the body is read), hostile filenames, concurrency, no path leaks in errors |
| `tests/test_cli.py` | 14 | Argument handling (including the implied default subcommand), `inspect`, error exit codes |
| `tests/test_integration.py` | 10 | Real model: accuracy, word alignment, no hallucination on silence, windowed long audio |

The suite does **not** need `make fixtures` — it generates its own audio with ffmpeg into a
temporary directory. Tests skip rather than fail if ffmpeg is missing.

---

## Part D — Run it with Docker

Useful if you'd rather not install ffmpeg or Python locally. Identical on all platforms. The
image bundles ffmpeg **and** the model weights, so it needs no network at runtime.

### Step 1: Make sure Docker is running

```bash
docker info
```

- **macOS / Windows:** start Docker Desktop first and give it 30-60 seconds.
  (macOS: `open -a Docker`.)
- **Linux:** `sudo systemctl start docker`. If you aren't in the `docker` group, prefix the
  commands below with `sudo`.

### Step 2: Build

```bash
make docker-build                                                        # macOS / Linux
docker build --build-arg MODEL=base -t audio-transcription-service .     # any platform
```

A few minutes: it installs ffmpeg, the Python dependencies, and pre-downloads the model. The
result is ~1.5 GB. For a smaller, faster build use the `tiny` model:

```bash
docker build --build-arg MODEL=tiny -t audio-transcription-service .
```

### Step 3: Transcribe

```bash
make docker-demo       # macOS / Linux
```

Or directly:

```bash
docker run --rm -v "$PWD/samples:/data:ro" audio-transcription-service \
  transcribe /data/speech_en.mp3 --format srt
```

```powershell
# Windows PowerShell — note ${PWD} and the quoting
docker run --rm -v "${PWD}/samples:/data:ro" audio-transcription-service `
  transcribe /data/speech_en.mp3 --format srt
```

```cmd
REM Windows Command Prompt
docker run --rm -v "%cd%/samples:/data:ro" audio-transcription-service transcribe /data/speech_en.mp3 --format srt
```

### Step 4: Prove it needs no network

The real check that nothing is fetched at runtime:

```bash
docker run --rm --network none -v "$PWD/samples:/data:ro" audio-transcription-service \
  transcribe /data/speech_en.mp3 --format srt
```

### Step 5: Serve the API from the container

```bash
docker run -p 8000:8000 --entrypoint uvicorn audio-transcription-service \
  transcription.api:app --host 0.0.0.0 --port 8000
```

Then use the same `curl` commands as Step 11.

### Transcribing your own files

Mount the directory containing them:

```bash
docker run --rm -v "/path/to/your/audio:/data:ro" -v "$PWD/out:/out" \
  audio-transcription-service transcribe /data/your-file.mp3 --format all --out /out
```

---

## Troubleshooting

### Any platform

| Symptom | Cause and fix |
|---|---|
| `ffmpeg was not found on PATH` | ffmpeg isn't installed, or isn't on this shell's PATH. See Step 1. |
| `transcribe: command not found` | The environment isn't activated. See Step 4, or prefix with `.venv/bin/` (`.venv\Scripts\` on Windows). |
| Install fails on `requires-python` | Python is older than 3.11. See Step 2. |
| First run seems to hang | It's downloading model weights. Run `transcribe warm` to see progress. |
| `Warning: unauthenticated requests to the HF Hub` | Harmless rate-limit notice from Hugging Face during the download. Ignore it. |
| Transcription seems slow | Check `RTF` in the summary line. Try `--model tiny`. `base` on CPU is ~0.1-0.2 RTF. |
| `413 audio_too_long` from the API | Working as designed — the sync endpoint caps duration. Use the CLI, or raise `TRANSCRIBE_MAX_SYNC_DURATION_SECONDS`. |
| Out of memory on a very long file | Lower `TRANSCRIBE_WINDOW_SECONDS` (e.g. `60`). Peak memory tracks the window, not the file. |
| Empty transcript | Expected for silence or non-speech. Check `transcribe inspect <file>` and the `no_speech_prob` values. |
| Tests report `ffmpeg/ffprobe not installed` | They skip rather than fail. Install ffmpeg to run them. |

### macOS

| Symptom | Fix |
|---|---|
| `python3` is 3.9 | Use `brew install python@3.12`, then `python3.12`. Don't use `/usr/bin/python3`. |
| `make: command not found` | `xcode-select --install`. |

### Linux

| Symptom | Fix |
|---|---|
| `No module named venv` / venv creation fails | Install the matching package: `sudo apt-get install python3.12-venv`. |
| `dnf` can't find ffmpeg | Enable RPM Fusion, or use the static build from johnvansickle.com/ffmpeg. |
| Docker `permission denied` on the socket | `sudo usermod -aG docker $USER`, then log out and back in — or prefix with `sudo`. |
| ALSA/JACK warnings in ffmpeg output | Harmless; no audio device is used. All I/O is to files. |

### Windows

| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `curl: -F is not recognized` / odd curl behaviour | PowerShell aliases `curl` to `Invoke-WebRequest`. Use **`curl.exe`**. |
| `make: command not found` | Expected — there's no `make`. Use the [command equivalents table](#command-equivalents-on-windows). |
| `./scripts/make_fixtures.sh` won't run | It's a bash script. Use Git Bash or WSL, or skip it — the tests don't need it. |
| `ffmpeg` not found right after installing | Reopen the terminal so the new `PATH` is picked up. |
| Long-path or permission errors in temp | Ensure `%TEMP%` is writable; antivirus scanning temp files can also interfere. |
| Docker volume mount appears empty | Quote the mount and use `${PWD}` in PowerShell / `%cd%` in cmd. Confirm the drive is shared in Docker Desktop settings. |

---

## The 2-minute version

```bash
# macOS
brew install ffmpeg && make setup && make warm && make demo && make test

# Linux
sudo apt-get install -y ffmpeg && make setup && make warm && make demo && make test
```

```powershell
# Windows
winget install Gyan.FFmpeg          # then reopen the terminal
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv\Scripts\Activate.ps1
transcribe warm
transcribe samples\speech_en.mp3
python -m pytest
```
