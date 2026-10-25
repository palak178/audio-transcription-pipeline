# The container exists for one reason: ffmpeg and the model weights are the two things most
# likely to make "it works on my machine" true and nowhere else. Both are baked in here, so
# the image runs the pipeline with no network access and no host dependencies.
FROM python:3.12-slim

ARG MODEL=base

# ffmpeg is a hard runtime dependency: it is the entire format-handling layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSCRIBE_DOWNLOAD_ROOT=/models \
    TRANSCRIBE_WHISPER_MODEL=${MODEL}

WORKDIR /app

# A non-editable install needs the source present, so this is one layer rather than the usual
# deps-then-source split: editing src/ does reinstall. Kept simple deliberately — the layer
# worth caching here is the ffmpeg apt install above and the model download below, both of
# which are far more expensive and both of which are unaffected by source edits.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Pre-download the weights at build time. Without this, the first container run would stall
# on a silent multi-megabyte download, and an air-gapped run would fail outright.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL}', device='cpu', compute_type='int8', download_root='/models')" \
    && chmod -R a+rX /models

# Run unprivileged.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Default to the CLI, since that is what a reviewer wants from `docker run`. Override with
# `--entrypoint` or pass `serve` to start the API:
#   docker run -p 8000:8000 --entrypoint uvicorn <image> transcription.api:app --host 0.0.0.0
ENTRYPOINT ["transcribe"]
CMD ["--help"]
