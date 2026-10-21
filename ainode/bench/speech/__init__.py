"""The speech bench: whether a transcription model on this fleet can actually hear.

The other five sections measure a model that writes. Throughput answers how fast it
writes tokens, the harness bench whether it can drive a coding agent to green tests,
the agentic rubric whether it can hold an agent loop together, the decision bench
whether its typed answers can be trusted, and the embedding bench what a retrieval
pipeline gets from a model that returns a vector. None of them says anything about a
speech-to-text model, which takes audio in and is scored against words nobody typed at
it.

Three things decide whether one is usable, and this section measures all three:

  * **Word error rate against known text.** Ten clips made with macOS ``say`` from
    sentences committed in ``clips.py``, so the reference is exactly what was spoken
    and is fixed before the run. Scored with a documented normaliser and reported twice:
    once with number words folded to digits, because a transcript that heard every word
    and wrote "9" for "nine" is not a hearing error, and once orthographically, which
    is the stricter number. The gap between them is how much of the error was spelling.
  * **Latency per clip.** p50 and p95 over one upload at a time, which is the shape an
    interactive transcription actually has, with the measured transport floor beside it
    so a reader can tell the engine's time from the wire's.
  * **Real-time factor.** Seconds of wall per second of audio, pooled over the run and
    per clip. Below 1 means the engine transcribes faster than the clip plays, which is
    what decides whether a live stream can be kept up with.

    ``clips.py``    the 10 clips: the manifest, the committed WAVs, the rebuild path
    ``client.py``   the multipart transcription call, request/parse split
    ``metrics.py``  the normaliser, word error rate, percentiles, real-time factor
    ``runner.py``   the loop, the tables, the record
    ``cli.py``      ``scripts/ainode-bench.py speech ...``

This is the one section whose request body is not JSON: OpenAI's speech-to-text API is
a multipart upload with the model id as a form field, which is also what AINode's proxy
routes on, so a run through a node's ``:3000/v1`` exercises the fleet's own audio path
end to end.

Stdlib only, like the rest of ``ainode/bench``. Record format: the ``speech`` block in
``bench/SCHEMA.md``.
"""

from ainode.bench.speech.client import (
    BOUNDARY,
    DEFAULT_API_KEY,
    DEFAULT_TIMEOUT,
    RESPONSE_FORMAT,
    Reply,
    Request,
    SpeechClient,
    SpeechError,
    encode_multipart,
    parse,
    post_multipart,
    request_for,
)
from ainode.bench.speech.clips import (
    CLIPS,
    CLIPS_ID,
    CLIPS_VERSION,
    ENV_CLIPS_DIR,
    ClipError,
    clips_block,
    clips_dir,
    clips_name,
    generate_clips,
    load_clips,
    wav_duration_seconds,
    wav_shape,
)
from ainode.bench.speech.metrics import (
    NORMALIZER_ID,
    NORMALIZER_VERSION,
    accuracy_block,
    clip_row,
    edit_counts,
    latency_block,
    normalize,
    normalizer_block,
    percentile,
    rate,
    real_time_factor,
    rtf_block,
)
from ainode.bench.speech.runner import (
    FLOOR_PROBES,
    SCHEMA,
    SOURCE,
    build_notes,
    build_record,
    build_speech_block,
    pct,
    print_table,
    run_clips,
    run_transport_floor,
    run_warmup,
)

__all__ = ["BOUNDARY", "CLIPS", "CLIPS_ID", "CLIPS_VERSION", "ClipError",
           "DEFAULT_API_KEY", "DEFAULT_TIMEOUT", "ENV_CLIPS_DIR", "FLOOR_PROBES",
           "NORMALIZER_ID", "NORMALIZER_VERSION", "RESPONSE_FORMAT", "Reply",
           "Request", "SCHEMA", "SOURCE", "SpeechClient", "SpeechError",
           "accuracy_block", "build_notes", "build_record", "build_speech_block",
           "clip_row", "clips_block", "clips_dir", "clips_name", "edit_counts",
           "encode_multipart", "generate_clips", "latency_block", "load_clips",
           "normalize", "normalizer_block", "parse", "pct", "percentile",
           "post_multipart", "print_table", "rate", "real_time_factor",
           "request_for", "rtf_block", "run_clips", "run_transport_floor",
           "run_warmup",
           "wav_duration_seconds", "wav_shape"]
