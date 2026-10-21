"""The texts this bench sends, and the six pairs it checks the meaning with.

Repo data, versioned next to its results, for the reason ``bench/decide/items.json``
is: two records are only comparable over the same corpus, so the corpus has an id and
a version and the record carries both. It lives in the package rather than in a JSON
file beside it because it is 56 short strings with no labels to curate, and a file
would add a load path and a load error for nothing.

The 50 latency texts are deliberately short and deliberately mixed: English,
Chinese, Spanish, German, Japanese and Arabic, with a few code and log lines, because
a multilingual model tokenises those very differently and a corpus of one language
measures one path through it. Nothing here is long enough to touch the 8192-token
window, which is the point: this section measures the per-request floor, not what
happens at depth.
"""
from __future__ import annotations

#: Bumped when a text changes. A record carries it so two runs are never compared
#: across a corpus edit.
CORPUS_VERSION = 1
CORPUS_ID = "embed-50"
PAIRS_ID = "pairs-6"

#: Batch sizes the throughput sweep runs. 1 is the interactive shape, 16 the shape a
#: modest indexer uses, 64 the engine's own --max-num-seqs in the proven recipe.
BATCH_SIZES = (1, 16, 64)

#: Texts each batch size is measured over, so every size does the same amount of
#: work and the rows compare directly: 64 requests at batch 1, 4 at 16, 1 at 64.
TEXTS_PER_BATCH_SIZE = 64

#: 50 short texts, one per request, for the latency percentiles.
LATENCY_TEXTS = (
    "How do I restart the AINode service?",
    "The engine bound on port 8001 after 87 seconds.",
    "Tensor parallelism splits one layer across several GPUs.",
    "A GB10 has 128 GB of unified memory.",
    "vLLM refused the flag and exited during startup.",
    "Which node is serving the embedding model right now?",
    "Decode is bandwidth bound on this hardware.",
    "The cluster found three peers over UDP discovery.",
    "Set gpu_memory_utilization to 0.06 for a stacked load.",
    "CUDA out of memory while capturing graphs.",
    "The checkpoint is quantized to NVFP4.",
    "Retrieval augmented generation needs an index first.",
    "Cosine similarity of two unit vectors is their dot product.",
    "The chat template switches thinking off with a keyword argument.",
    "A pooling runner has no language modelling head.",
    "Docker pull is the only install step for an end user.",
    "The master proxies every request by model id.",
    "Prefix caching reuses the shared start of a prompt.",
    "Write the result file into bench/results and regenerate the table.",
    "This model answers in 1024 dimensions.",
    "def cosine(a, b): return dot(a, b) / (norm(a) * norm(b))",
    "SELECT id, embedding FROM documents ORDER BY distance LIMIT 10;",
    "ERROR 2026-09-19T21:41:18Z engine exited with status 1",
    "git rebase --onto main feature~3 feature",
    "curl -s http://localhost:3000/v1/models | jq .data",
    "The kettle boiled while the cat slept on the windowsill.",
    "She bought a ticket to the opera for Friday evening.",
    "Add two cups of flour and a pinch of salt.",
    "The bridge was closed for repairs until October.",
    "Rain is forecast for the whole of next week.",
    "He learned to sail on a lake in the mountains.",
    "The library keeps its oldest maps in a cold room.",
    "Migrating birds navigate partly by the earth's magnetic field.",
    "A sourdough starter needs feeding every day.",
    "The train arrives at platform nine at half past six.",
    "嵌入模型返回一个向量，而不是文本。",
    "请问这个节点现在在服务哪个模型？",
    "El modelo devuelve un vector de mil veinticuatro dimensiones.",
    "¿Cuántos nodos hay en el clúster ahora mismo?",
    "Das Modell gibt einen Vektor zurück, keinen Text.",
    "Der Dienst wurde nach dem Neustart wieder gestartet.",
    "このモデルはテキストではなくベクトルを返します。",
    "クラスタには現在いくつのノードがありますか。",
    "النموذج يعيد متجهًا وليس نصًا.",
    "كم عدد العقد في المجموعة الآن؟",
    "Le modèle renvoie un vecteur de mille vingt-quatre dimensions.",
    "Il servizio è stato riavviato dopo l'aggiornamento.",
    "O modelo devolve um vetor em vez de texto.",
    "Модель возвращает вектор, а не текст.",
    "Bu model metin yerine bir vektör döndürür.",
)

#: Six hand-written pairs: three that mean close to the same thing and three that do
#: not. The check is ordering, not a threshold: every related pair must score above
#: every unrelated pair. A threshold would be a number about this checkpoint, while
#: the ordering is a statement about whether the vectors mean anything at all, which
#: is the failure this catches (a pooling runner on the wrong checkpoint, a window
#: silently truncating, a normalisation that never ran).
PAIRS = (
    {"id": "rel-1", "related": True,
     "a": "How do I restart the AINode service?",
     "b": "What is the command to restart AINode?"},
    {"id": "rel-2", "related": True,
     "a": "The GPU ran out of memory while the engine was starting.",
     "b": "The engine died with a CUDA out-of-memory error during launch."},
    {"id": "rel-3", "related": True,
     "a": "Cats sleep for most of the day.",
     "b": "Domestic cats spend the majority of their time asleep."},
    {"id": "unrel-1", "related": False,
     "a": "The GPU ran out of memory while the engine was starting.",
     "b": "Cats sleep for most of the day."},
    {"id": "unrel-2", "related": False,
     "a": "How do I restart the AINode service?",
     "b": "Add two cups of flour and a pinch of salt."},
    {"id": "unrel-3", "related": False,
     "a": "Tensor parallelism splits one layer across several GPUs.",
     "b": "She bought a ticket to the opera for Friday evening."},
)

#: Every distinct text a pair needs, in a stable order, so all twelve sides are
#: embedded in ONE request and no pair's score depends on which request it landed in.
PAIR_TEXTS = tuple(dict.fromkeys(
    [side for pair in PAIRS for side in (pair["a"], pair["b"])]))


def cycle_texts(count: int, texts=LATENCY_TEXTS) -> list:
    """``count`` texts, taken from ``texts`` and wrapping when it runs out.

    Wrapping rather than repeating one string: a batch of 64 copies of the same text
    is a prefix-cache measurement, not a throughput one.
    """
    if count <= 0 or not texts:
        return []
    return [texts[i % len(texts)] for i in range(count)]


def corpus_block() -> dict:
    """What the record says about the texts, so a reader can tell two runs apart."""
    return {"id": CORPUS_ID, "version": CORPUS_VERSION,
            "latency_texts": len(LATENCY_TEXTS),
            "pairs_id": PAIRS_ID,
            "pairs": len(PAIRS),
            "related_pairs": sum(1 for p in PAIRS if p["related"]),
            "source": "ainode/bench/embed/corpus.py"}


#: Alias kept because the package docstring and the CLI talk about "the corpus".
CORPUS = LATENCY_TEXTS

__all__ = ["BATCH_SIZES", "CORPUS", "CORPUS_ID", "CORPUS_VERSION",
           "LATENCY_TEXTS", "PAIRS", "PAIRS_ID", "PAIR_TEXTS",
           "TEXTS_PER_BATCH_SIZE", "corpus_block", "cycle_texts"]
