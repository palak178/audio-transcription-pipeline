PY := .venv/bin/python
UV := uv
MODEL ?= base
SAMPLE ?= samples/speech_en.mp3

.DEFAULT_GOAL := help
.PHONY: help setup fixtures warm test test-all lint demo demo-long serve docker-build docker-demo clean

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install the project
	@command -v ffmpeg >/dev/null || { \
		echo "ffmpeg is required. macOS: brew install ffmpeg | Debian: apt-get install ffmpeg"; \
		exit 1; }
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev]"
	@echo "Done. Next: make warm && make demo"

warm:  ## Download and load the Whisper model ahead of first use
	TRANSCRIBE_WHISPER_MODEL=$(MODEL) $(PY) -m transcription.cli warm

fixtures:  ## Generate the multi-format test audio set
	./scripts/make_fixtures.sh

test:  ## Fast suite: no model weights needed, a few seconds
	$(PY) -m pytest

test-all:  ## Full suite, including real-model integration tests
	$(PY) -m pytest -m "slow or not slow"

lint:  ## Static checks
	$(PY) -m ruff check src tests

demo:  ## Transcribe the committed sample and show JSON + SRT
	TRANSCRIBE_WHISPER_MODEL=$(MODEL) $(PY) -m transcription.cli transcribe $(SAMPLE) --format json
	@echo
	TRANSCRIBE_WHISPER_MODEL=$(MODEL) $(PY) -m transcription.cli transcribe $(SAMPLE) --format srt --quiet

demo-long:  ## Transcribe a ~6 minute file, showing per-segment progress (run `make fixtures` first)
	TRANSCRIBE_WHISPER_MODEL=$(MODEL) $(PY) -m transcription.cli \
		transcribe samples/generated/long_6min.mp3 --format txt --out out/

serve:  ## Run the HTTP API on :8000
	TRANSCRIBE_WHISPER_MODEL=$(MODEL) $(PY) -m uvicorn transcription.api:app --host 0.0.0.0 --port 8000

docker-build:  ## Build the container (bundles ffmpeg and the model)
	docker build --build-arg MODEL=$(MODEL) -t audio-transcription-service .

docker-demo:  ## Transcribe the sample inside the container
	docker run --rm -v "$(PWD)/samples:/data:ro" audio-transcription-service \
		transcribe /data/speech_en.mp3 --format srt

clean:  ## Remove generated artefacts
	rm -rf out samples/generated .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
