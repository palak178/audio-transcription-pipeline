#!/usr/bin/env bash
#
# Generate the test-fixture audio set.
#
# Only one real recording is committed to git
# (samples/speech_en.mp3, ~43 KB of synthesised English speech); every other fixture is
# derived from it here. That keeps the repository small while still giving the test suite a
# genuine multi-format matrix and the awkward inputs that break naive implementations.
#
# Usage: ./scripts/make_fixtures.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE="samples/speech_en.mp3"
OUT="samples/generated"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found; install it first." >&2; exit 1; }
[ -f "$SOURCE" ] || { echo "missing $SOURCE" >&2; exit 1; }

mkdir -p "$OUT"
ff() { ffmpeg -nostdin -hide_banner -loglevel error -y "$@"; }

echo "Generating fixtures in $OUT/"

# --- The same speech in a spread of containers and codecs -------------------------------
ff -i "$SOURCE" -ac 1 -ar 16000 -c:a pcm_s16le   "$OUT/speech.wav"
ff -i "$SOURCE" -ac 1 -ar 44100 -c:a aac -b:a 64k "$OUT/speech.m4a"
ff -i "$SOURCE" -ac 1 -ar 44100 -c:a flac         "$OUT/speech.flac"
ff -i "$SOURCE" -ac 1 -ar 48000 -c:a libopus -b:a 32k "$OUT/speech.ogg"

# --- Sample-rate and channel-layout extremes, all of which must normalise cleanly -------
ff -i "$SOURCE" -ac 1 -ar 8000  -c:a pcm_s16le "$OUT/speech_8k_telephony.wav"
ff -i "$SOURCE" -ac 2 -ar 48000 -c:a pcm_s16le "$OUT/speech_48k_stereo.wav"

# --- A file whose extension lies: FLAC bytes under an .mp3 name -------------------------
cp "$OUT/speech.flac" "$OUT/lying_extension.mp3"

# --- Non-speech and empty-speech inputs -------------------------------------------------
# Pure silence: must yield an empty transcript, not a crash and not a hallucination.
ff -f lavfi -i anullsrc=r=16000:cl=mono -t 10 -c:a pcm_s16le "$OUT/silence.wav"
# A tone, i.e. audio with no speech in it at all.
ff -f lavfi -i "sine=frequency=440:r=16000" -t 5 -ac 1 -c:a pcm_s16le "$OUT/tone.wav"
# Speech followed by a long silence, the classic trigger for Whisper's repetition loop.
ff -i "$SOURCE" -f lavfi -i anullsrc=r=16000:cl=mono \
   -filter_complex "[0:a]aformat=sample_rates=16000:channel_layouts=mono[a];[1:a]atrim=0:40[b];[a][b]concat=n=2:v=0:a=1" \
   -c:a pcm_s16le "$OUT/speech_then_long_silence.wav"

# --- Malformed inputs -------------------------------------------------------------------
: > "$OUT/empty.mp3"                                   # zero bytes
head -c 300 "$SOURCE" > "$OUT/truncated.mp3"           # header fragment only
printf 'This is plain text, not audio.\n' > "$OUT/not_audio.mp3"
# A video container carrying no audio track.
ff -f lavfi -i color=c=black:s=160x120:r=10 -t 2 -c:v mpeg4 -an "$OUT/no_audio_track.mp4"
# A video container that *does* carry audio: we accept these.
ff -f lavfi -i color=c=black:s=160x120:r=10 -t 7 -i "$SOURCE" \
   -c:v mpeg4 -c:a aac -shortest "$OUT/video_with_audio.mp4"

# --- A long recording, for the windowing/progress demo ----------------------------------
# ~6 minutes, built by looping the sample. Generated rather than committed: it is 2 MB.
ff -stream_loop 49 -i "$SOURCE" -c:a libmp3lame -b:a 48k "$OUT/long_6min.mp3"

echo
printf '%-34s %10s  %s\n' FILE BYTES DURATION
for f in "$OUT"/*; do
  dur=$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$f" 2>/dev/null || echo '-')
  printf '%-34s %10s  %s\n' "$(basename "$f")" "$(wc -c < "$f" | tr -d ' ')" "$dur"
done
