-- Window-level checkpoints, so a retry does not re-transcribe audio it already finished.
--
-- This is the direct payoff of the windowed long-audio path: the windows are already the
-- natural checkpoint boundary, because each one's segments are complete and absolute the
-- moment it finishes. Without this, a worker evicted three hours into a four-hour file throws
-- away three hours of CPU; with it, the retry re-reads the WAV (cheap, disk-bound) and skips
-- straight to window N.
--
-- `plan_fingerprint` is what makes resuming safe. Window boundaries are content-dependent —
-- each cut lands at the quietest point in its search band — so they are only reproducible if
-- the inputs that decide them are unchanged. If the window size, the search band, the model or
-- the decoding options differ from the attempt that wrote the checkpoint, window 7 is no
-- longer the same stretch of audio and the checkpoints are discarded rather than trusted.

CREATE TABLE IF NOT EXISTS job_checkpoints (
    job_id            uuid    NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    window_index      integer NOT NULL,
    -- Where this window started, in seconds from the beginning of the recording. Stored for
    -- debugging and for progress reporting on resume, not used to reconstruct the timeline:
    -- the segments below are already absolute.
    offset_seconds    numeric NOT NULL,
    -- The window's segments, already rebased onto the whole recording's timeline and already
    -- numbered, so resuming is concatenation rather than re-derivation.
    segments          jsonb   NOT NULL,
    -- Detected language, pinned by the first window. Carried so a resumed run does not
    -- re-detect and risk disagreeing with the segments it is resuming from.
    language          text,
    language_probability numeric,
    plan_fingerprint  char(64) NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (job_id, window_index)
);

-- Loading a job's checkpoints in window order is the only read pattern, and the primary key
-- already serves it. No extra index.
