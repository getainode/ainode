"""The embedding bench: what an embedding model is actually like to build on.

The other four sections measure generation. Throughput answers how fast a model
writes tokens, the harness bench whether it can drive a coding agent to green
tests, the agentic rubric whether it can hold an agent loop together, and the
decision bench whether its typed answers can be trusted. None of them says anything
about the model a retrieval pipeline spends all day calling, because that model
writes no tokens at all: it returns a vector.

Four things decide whether an embedding model is usable, and this section measures
all four:

  * **Dimensions.** The width of the vector, read off the response rather than off a
    model card, because it is what every index downstream has to be built for.
  * **Single-request latency.** p50 and p95 over 50 short texts, one text per
    request, which is the shape an interactive lookup actually has.
  * **Throughput at a batch size.** Texts per second and tokens per second at
    batches of 1, 16 and 64. Batching is the whole difference between indexing a
    corpus overnight and indexing it in an hour, and on this hardware it is the
    number that moves most.
  * **A quality sanity check, not a leaderboard.** Six hand-written pairs, three
    related and three not, with one question asked of them: does every related pair
    score above every unrelated pair. That catches the failure this bench exists to
    catch, which is an engine returning well-formed vectors that mean nothing (a
    pooling runner serving the wrong checkpoint, a truncated window, a broken
    normalisation). It is deliberately not a claim about retrieval quality: MTEB
    exists and this is not it.

    ``client.py``     the OpenAI-compatible embeddings call, request/parse split
    ``corpus.py``     the 50 texts and the 6 pairs, versioned as repo data
    ``metrics.py``    percentiles, throughput, cosine similarity
    ``runner.py``     the loop, the tables, the record
    ``cli.py``        ``scripts/ainode-bench.py embed ...``

Stdlib only, like the rest of ``ainode/bench``. Record format: the ``embed`` block
in ``bench/SCHEMA.md``.
"""

from ainode.bench.embed.client import (
    DEFAULT_API_KEY,
    DEFAULT_TIMEOUT,
    EmbedClient,
    EmbedError,
    Reply,
    Request,
    parse,
    post_json,
    request_for,
)
from ainode.bench.embed.corpus import (
    BATCH_SIZES,
    CORPUS,
    CORPUS_ID,
    CORPUS_VERSION,
    LATENCY_TEXTS,
    PAIRS,
    PAIRS_ID,
    TEXTS_PER_BATCH_SIZE,
    corpus_block,
    cycle_texts,
)
from ainode.bench.embed.metrics import (
    cosine,
    latency_block,
    percentile,
    quality_block,
    throughput_row,
)
from ainode.bench.embed.runner import (
    FLOOR_PROBES,
    SCHEMA,
    SOURCE,
    build_embed_block,
    build_notes,
    build_record,
    print_table,
    run_latency,
    run_quality,
    run_throughput,
    run_transport_floor,
)

__all__ = ["BATCH_SIZES", "CORPUS", "CORPUS_ID", "CORPUS_VERSION",
           "DEFAULT_API_KEY", "DEFAULT_TIMEOUT", "EmbedClient", "EmbedError",
           "FLOOR_PROBES", "LATENCY_TEXTS", "PAIRS", "PAIRS_ID", "Reply",
           "Request", "SCHEMA",
           "SOURCE", "TEXTS_PER_BATCH_SIZE", "build_embed_block", "build_notes",
           "build_record", "corpus_block", "cosine", "cycle_texts",
           "latency_block", "parse", "percentile", "post_json", "print_table",
           "quality_block", "request_for", "run_latency", "run_quality",
           "run_throughput", "run_transport_floor", "throughput_row"]
