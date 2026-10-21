"""The Jevals recipe, suite 0.1.0, as functions over plain decision dicts.

The recipe is recorded in ``bench/decide/JEVALS.md`` with the URL and the date it was
read, and every formula here cites the section it came from. Nothing in this module
talks to a model, a node or a file: a metric is a pure function of the decisions, which
is what lets ``tests/test_bench_decide_jevals.py`` score hand-computed examples with no
network.

Why a second metrics module beside :mod:`ainode.bench.decide.metrics` rather than a
rewrite of it: the two measure different things and older records carry the first one.
``metrics.py`` scores the 110-item AINode set one pass per item, with a one-term Brier
on the labeled option and five bins. This module scores the three public Jevals sets
five passes per item, with the recipe's own multiclass Brier, its ranked probability
score, its ten bins, its Decision Score against the label prior and its hand-off share.
A record can carry both; a number from one is not a number from the other.

The unit of measurement here is a **decision**: one (item, repeat) pair. A decision dict
is what :mod:`ainode.bench.decide.suite` builds:

    ``id``           the item it belongs to
    ``set``          which suite set (``pubmedqa``, ``banking77``, ``helpsteer2``)
    ``repeat``       0-based repeat index
    ``order``        which presented option order this repeat used
    ``type``         ``noul``, ``choice`` or ``score``
    ``options``      the task's options in CANONICAL order, never the presented one
    ``label``        the labeled option, as an option string
    ``vector``       probability per option, already normalized
    ``pick``         the option the system picked, or None (which counts as wrong)
    ``malformed``    the answer could not be read as a vector over these options
    ``one_hot``      the system reported no probabilities, so the vector is its pick
    ``confidence``   the probability of the pick, or None
    ``wall_ms``      measured round trip
    ``tokens_in``    prompt tokens the endpoint reported, or None
    ``tokens_out``   completion tokens the endpoint reported, or None
    ``error``        a transport or protocol failure, or None

Two exclusion rules run through everything below, both the recipe's:

  * **A transport failure is never scored.** A decision with an ``error`` is out of the
    accuracy, out of the losses, out of every rate, and counted on its own as ``failed``.
    A board cannot be built while any (item, repeat) is missing, so a run with failures
    reports them rather than quietly scoring 299 items as 300.
  * **A malformed or refused answer IS scored**, as the uniform distribution and a wrong
    pick. It counts in the Decision Score and the accuracy and is excluded from the ECE,
    the flip rates and the gate.
"""
from __future__ import annotations

import math

#: Question types, Jev's three primitives.
NOUL = "noul"
CHOICE = "choice"
SCORE = "score"
TYPES = (NOUL, CHOICE, SCORE)

#: Repeats per item. Five is the suite's figure, and a listing on a board needs a
#: complete run of every task at five.
REPEATS = 5

#: Bins for the calibration gap. Ten equal-width bins, which is the recipe's number and
#: not ``metrics.BINS``' five.
BINS = 10

#: The grid the gate and the hand-off threshold are searched on.
GRID = 0.01

#: Hand-off at 95 percent: the accuracy a threshold has to reach, and the smallest
#: number of decisions that may stand behind it.
HANDOFF_ACCURACY = 0.95
HANDOFF_MIN_DECISIONS = 100

#: The shared gate's rule: pooled error at most this, over at least this many decisions.
GATE_MAX_ERROR = 0.05
GATE_MIN_DECISIONS = 100

#: The frozen gates jevals.com publishes for suite 0.1.0. They are pooled across every
#: listed system and frozen from the first release, so one run cannot recompute them:
#: a run reports its coverage AT these, and its own one-system gate separately.
PUBLISHED_GATES = {CHOICE: 0.96, SCORE: None, NOUL: 0.91}
PUBLISHED_GATE_SOURCE = "jevals.com/methodology, suite 0.1.0, frozen"

#: Comparisons on the 0.01 grid absorb float representation error rather than dropping
#: a stated 0.96 that does not survive being written down as a double.
EPSILON = 1e-9


# ------------------------------------------------------------------ small helpers

def round_half_up(value: float) -> int:
    """``Math.round`` semantics, which is what the published bin formula assumes.

    Python's ``round`` breaks a tie to even (``round(0.5) == 0``), so a confidence of
    exactly 0.105 would bin differently here than on the board. The ECE formula is
    published as ``min(9, floor(round(100*c) / 10))`` over a JavaScript harness, so the
    rounding has to be the JavaScript one.
    """
    return int(math.floor(float(value) + 0.5))


def is_probability(value) -> bool:
    """A finite number in [0, 1]. Booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and -EPSILON <= number <= 1.0 + EPSILON


def uniform(options) -> dict:
    """The vector a refused or malformed answer is scored as."""
    if not options:
        return {}
    share = 1.0 / len(options)
    return {option: share for option in options}


#: JevBench's two sum tolerances. Its headline renormalizes anything inside a 2 percent
#: band and publishes a strict column under the 0.001 tolerance its v1 froze, because
#: the loose band mostly catches three-decimal rounding on a nine-option question. Both
#: rates are reported here under its own names.
SUM_TOLERANCE = 0.02
SUM_TOLERANCE_STRICT = 0.001


def normalize_vector(raw, options):
    """``(vector, None)`` or ``(None, why)``: the recipe's malformed rules, in order.

    Malformed when the map is missing, names an unknown option, repeats one, carries a
    value that is not a finite number in [0, 1], or sums to 0. Otherwise the listed
    values are divided by their sum, which covers both published normalizations for a
    full vector: a sum above 1 is scaled down, and a sum below 1 (Jev rounds to two
    decimals, so 0.99 happens) is scaled up.

    An option the answer did not name is 0 rather than absent, because every metric
    below reads the vector as a distribution over the task's whole option set.
    """
    if not isinstance(raw, dict) or not raw:
        return None, "no probability map"
    allowed = set(options)
    seen = set()
    values = {}
    for key, value in raw.items():
        name = key if isinstance(key, str) else str(key)
        if name not in allowed:
            return None, f"unknown option {name!r}"
        if name in seen:
            return None, f"duplicate option {name!r}"
        seen.add(name)
        if not is_probability(value):
            return None, f"option {name!r} has probability {value!r}"
        values[name] = min(1.0, max(0.0, float(value)))
    total = sum(values.values())
    if total <= 0:
        return None, "probabilities sum to 0"
    return {option: values.get(option, 0.0) / total for option in options}, None


def pick_from_vector(vector, options, presented=None, qtype: str = CHOICE,
                     stated=None):
    """The most probable option, with the recipe's tie rules.

    ``choice``: ties go to the option the system listed first, which is its own stated
    answer when that is one of the winners and otherwise the first winner in the order
    the options were PRESENTED in. ``score``: ties go to the lower level, which is the
    first winner in canonical (level) order. ``noul``: a yes/no answer of exactly 0.5
    has no pick and counts as wrong, so this returns None.
    """
    if not vector:
        return None
    top = max(vector.get(option, 0.0) for option in options)
    winners = [option for option in options if vector.get(option, 0.0) >= top - EPSILON]
    if not winners:
        return None
    if qtype == NOUL:
        # Exactly 0.5 either way is the published no-pick case. Two options, equal mass.
        if len(winners) > 1:
            return None
        return winners[0]
    if qtype == SCORE:
        return winners[0]
    if stated in winners:
        return stated
    order = list(presented or options)
    for option in order:
        if option in winners:
            return option
    return winners[0]


# ------------------------------------------------------------------ partitions

def scored(decisions) -> list:
    """Decisions the recipe scores: everything that came back at all.

    A transport failure is not one of them. That is the line between "the model was
    wrong" and "the request never happened", and collapsing it is how a refused run
    turns into a set of confident misses.
    """
    return [d for d in decisions if not d.get("error")]


def failed(decisions) -> list:
    return [d for d in decisions if d.get("error")]


def calibratable(decisions) -> list:
    """Decisions the ECE, the gate, the hand-off and the flip rates run over.

    Malformed answers are out (the recipe says so) and so are one-hot answers, because
    a system that reports no probabilities has a confidence of 1.00 by construction and
    a board shows it a dash rather than a perfect gate.
    """
    return [d for d in scored(decisions)
            if not d.get("malformed") and not d.get("one_hot")
            and d.get("confidence") is not None]


def by_item(decisions) -> dict:
    """``{item id: [decisions in repeat order]}``, which is the loss's unit."""
    out = {}
    for decision in decisions:
        out.setdefault(decision["id"], []).append(decision)
    for rows in out.values():
        rows.sort(key=lambda d: d.get("repeat", 0))
    return out


def is_correct(decision) -> bool:
    """A pick equal to the label. No pick is wrong, which malformed answers rely on."""
    pick = decision.get("pick")
    return pick is not None and pick == decision.get("label")


# ------------------------------------------------------------------ the two losses

def brier_loss(vector, options, label) -> float:
    """``sum_k (p_k - y_k)^2``: the multiclass Brier score, for choice and noul.

    Not the one-term version :mod:`ainode.bench.decide.metrics` uses. Two options with
    all the mass on the wrong one scores 2.0, and a uniform answer over K options
    scores ``1 - 1/K``.
    """
    total = 0.0
    for option in options:
        target = 1.0 if option == label else 0.0
        total += (float(vector.get(option, 0.0)) - target) ** 2
    return total


def rps_loss(vector, options, label) -> float:
    """``sum_{k<K} (P_k - Y_k)^2 / (K-1)``: the ranked probability score, for score.

    Over CUMULATIVE probabilities, which is what makes it ordinal: putting the mass one
    level away from the label costs less than putting it four levels away, and a Brier
    score would charge both the same. The sum runs over the first K-1 cut points,
    because the last cumulative pair is 1 against 1 for every possible answer.
    """
    count = len(options)
    if count < 2:
        return 0.0
    total = 0.0
    cumulative_p = 0.0
    cumulative_y = 0.0
    for option in options[:-1]:
        cumulative_p += float(vector.get(option, 0.0))
        cumulative_y += 1.0 if option == label else 0.0
        total += (cumulative_p - cumulative_y) ** 2
    return total / (count - 1)


def loss_for(qtype: str, vector, options, label) -> float:
    """The loss the recipe uses for that primitive."""
    if qtype == SCORE:
        return rps_loss(vector, options, label)
    return brier_loss(vector, options, label)


def decision_loss(decision) -> float:
    """One decision's loss. A malformed answer's vector is already the uniform one."""
    vector = decision.get("vector") or uniform(decision["options"])
    return loss_for(decision.get("type", CHOICE), vector, decision["options"],
                    decision["label"])


def mean_item_loss(decisions):
    """``L``: the mean over items of the mean over that item's repeats.

    Per item first, so an item answered five times weighs the same as one answered
    once, which is what makes the item-cluster bootstrap on the board meaningful and
    what keeps a partly failed item from pulling the mean.
    """
    groups = by_item(scored(decisions))
    if not groups:
        return None
    per_item = [sum(decision_loss(d) for d in rows) / len(rows)
                for rows in groups.values() if rows]
    if not per_item:
        return None
    return sum(per_item) / len(per_item)


# ------------------------------------------------------------------ the label prior

def label_prior(decisions):
    """The baseline that defines 0: the base rates of the EVALUATED items.

    One vector, over the options, from how often each label occurs among the items this
    run scored. Counted per item and not per decision, so five repeats of one item do
    not make its label five times as common.
    """
    groups = by_item(scored(decisions))
    if not groups:
        return None, None
    options = None
    counts = {}
    for rows in groups.values():
        first = rows[0]
        options = options or list(first["options"])
        counts[first["label"]] = counts.get(first["label"], 0) + 1
    total = sum(counts.values())
    if not options or not total:
        return None, None
    return {option: counts.get(option, 0) / total for option in options}, options


def prior_loss(decisions):
    """``L_prior``: the same loss, for the label prior, on the same items."""
    prior, options = label_prior(decisions)
    if prior is None:
        return None
    groups = by_item(scored(decisions))
    qtype = next(iter(groups.values()))[0].get("type", CHOICE)
    losses = [loss_for(qtype, prior, options, rows[0]["label"])
              for rows in groups.values()]
    if not losses:
        return None
    return sum(losses) / len(losses)


def prior_accuracy(decisions):
    """The guessing floor: how often the label prior's own pick is right.

    Its pick is the most common label, so this is that label's share of the evaluated
    items. Every metrics block here carries it, because an accuracy of 0.62 on a set
    whose majority class is 0.62 is a model that has learned nothing, and the reporting
    rule this bench follows is that the floor travels with the figure.
    """
    prior, options = label_prior(decisions)
    if prior is None:
        return None
    groups = by_item(scored(decisions))
    qtype = next(iter(groups.values()))[0].get("type", CHOICE)
    if qtype == NOUL and len(set(prior.values())) == 1:
        # A 50/50 prior has no pick under the yes/no rule, so it is right never.
        return 0.0
    guess = pick_from_vector(prior, options, presented=options,
                             qtype=CHOICE if qtype == NOUL else qtype)
    if guess is None:
        return 0.0
    hits = sum(1 for rows in groups.values() if rows[0]["label"] == guess)
    return hits / len(groups)


def decision_score(system_loss, baseline_loss):
    """``100 * (1 - L_system / L_prior)``. 100 is perfect, 0 is the base rates.

    Negative is worse than the base rates and is returned as it is, never clamped. None
    when either loss is missing, or when the prior's loss is 0, which happens only on a
    set where every item carries the same label and no system can do better than it.
    """
    if system_loss is None or baseline_loss is None or baseline_loss <= 0:
        return None
    return 100.0 * (1.0 - (system_loss / baseline_loss))


# ------------------------------------------------------------------ accuracy, ECE

def accuracy(decisions):
    """Correct picks over ALL scored decisions, items times repeats.

    Malformed and refused answers are in the denominator and count as wrong, which is
    the recipe's rule and the reason a system cannot buy accuracy by refusing.
    """
    rows = scored(decisions)
    if not rows:
        return None
    return sum(1 for d in rows if is_correct(d)) / len(rows)


def bin_index(confidence: float, bins: int = BINS) -> int:
    """``min(9, floor(round(100*c) / 10))``, so a confidence of 1.00 lands in the last."""
    return min(bins - 1, max(0, round_half_up(100.0 * float(confidence)) // (100 // bins)))


def reliability(decisions, bins: int = BINS) -> list:
    """The table behind the ECE: per bin, how many decisions, how often right, how sure.

    An empty bin keeps its row with ``count: 0`` and nulls, so the shape of the table
    does not change between runs and a reader can see which part of the range a system
    never used.
    """
    buckets = [[] for _ in range(bins)]
    for decision in calibratable(decisions):
        buckets[bin_index(decision["confidence"], bins)].append(decision)
    table = []
    for index, bucket in enumerate(buckets):
        block = {"lo": round(index / bins, 2), "hi": round((index + 1) / bins, 2),
                 "count": len(bucket), "accuracy": None, "confidence": None}
        if bucket:
            block["accuracy"] = sum(1 for d in bucket if is_correct(d)) / len(bucket)
            block["confidence"] = sum(float(d["confidence"])
                                      for d in bucket) / len(bucket)
        table.append(block)
    return table


def ece(decisions, bins: int = BINS):
    """``sum_b (n_b/N) * |accuracy_b - mean confidence_b|`` as a ratio, or None.

    None when nothing could be calibrated, which is the board's dash: a system with no
    probabilities has no calibration gap, and reporting 0 for it would make the most
    opaque row look like the most honest one.
    """
    rows = calibratable(decisions)
    if not rows:
        return None
    total = 0.0
    for block in reliability(rows, bins):
        if block["count"]:
            total += abs(block["accuracy"] - block["confidence"]) * block["count"]
    return total / len(rows)


def ece_points(decisions, bins: int = BINS):
    """The ECE the way a board prints it: in points, so 0.058 reads as 5.8."""
    value = ece(decisions, bins)
    return None if value is None else value * 100.0


# ------------------------------------------------------------------ gate, hand-off

def grid_thresholds(step: float = GRID) -> list:
    """``[0.00, 0.01, ..., 1.00]``, built off integers so the steps are exact."""
    count = int(round(1.0 / step))
    return [index / count for index in range(count + 1)]


def above(decisions, threshold: float) -> list:
    """Calibratable decisions whose confidence clears the threshold."""
    return [d for d in calibratable(decisions)
            if float(d["confidence"]) >= threshold - EPSILON]


def handoff(decisions, target: float = HANDOFF_ACCURACY,
            min_decisions: int = HANDOFF_MIN_DECISIONS, step: float = GRID):
    """Hand-off at 95 percent: the system's own threshold and the share it can take.

    The LOWEST confidence on the grid at which the decisions clearing it are at least
    ``target`` correct, with at least ``min_decisions`` of them. ``share`` is those
    decisions over ALL of this system's decisions, refused and malformed included,
    which is the published denominator and the reason the share is not simply coverage
    among the answers it was confident about.

    None when no threshold qualifies, which is the board's dash for a row whose accuracy
    never reaches 95 percent. The threshold is chosen on the same decisions it is
    measured on, so the share is optimistic; it is optimistic the same way for every
    system, which is what makes it comparable.
    """
    total = len(scored(decisions))
    if not total:
        return None
    for threshold in grid_thresholds(step):
        kept = above(decisions, threshold)
        if len(kept) < min_decisions:
            continue
        correct = sum(1 for d in kept if is_correct(d))
        if correct / len(kept) >= target - EPSILON:
            return {"threshold": round(threshold, 2), "n": len(kept),
                    "accuracy": correct / len(kept), "share": len(kept) / total,
                    "decisions": total}
    return None


def gate_local(decisions, max_error: float = GATE_MAX_ERROR,
               min_decisions: int = GATE_MIN_DECISIONS, step: float = GRID):
    """The published gate rule applied to THIS run's decisions alone.

    The board's gate is pooled over every listed system and frozen from the first
    release, so it is not a thing one run can recompute; this is the same arithmetic
    over one system, and the record labels it as such. None when no threshold on the
    grid gets the pooled error to ``max_error`` over at least ``min_decisions``.
    """
    for threshold in grid_thresholds(step):
        kept = above(decisions, threshold)
        if len(kept) < min_decisions:
            continue
        wrong = sum(1 for d in kept if not is_correct(d))
        if wrong / len(kept) <= max_error + EPSILON:
            return round(threshold, 2)
    return None


def gate_coverage(decisions, gate):
    """Coverage and accuracy at a gate: the two numbers a board row shows beside it.

    ``coverage`` is over ALL scored decisions, the same denominator the hand-off share
    uses, so a system that refuses half the set cannot read as covering everything it
    answered. None when the primitive has no gate (``score`` has none in 0.1.0) or when
    nothing here can be gated.
    """
    if gate is None:
        return None
    total = len(scored(decisions))
    if not total or not calibratable(decisions):
        return None
    kept = above(decisions, float(gate))
    block = {"gate": float(gate), "n": len(kept), "coverage": len(kept) / total,
             "accuracy": None}
    if kept:
        block["accuracy"] = sum(1 for d in kept if is_correct(d)) / len(kept)
    return block


# ------------------------------------------------------------------ flips

def _picks_at(rows, repeats):
    """The picks at those repeat indices, or None when any of them cannot be compared.

    A malformed or failed answer in one of the compared repeats takes the whole item out
    of the rate, numerator and denominator both, which is the recipe's exclusion. Half
    an item's picks would otherwise read as a flip.
    """
    wanted = []
    for index in repeats:
        match = [d for d in rows if d.get("repeat") == index]
        if not match:
            return None
        decision = match[0]
        if decision.get("error") or decision.get("malformed"):
            return None
        if decision.get("pick") is None:
            return None
        wanted.append(decision["pick"])
    return wanted


def flip_rate(decisions, repeats):
    """Share of items whose pick is not the same across those repeat indices.

    ``(rate, n)``, where ``n`` is how many items could be compared at all. ``(None, 0)``
    when none could, which is the honest answer for a run with fewer repeats than the
    rate needs rather than a 0 that reads as perfect determinism.
    """
    comparable = 0
    flipped = 0
    for rows in by_item(decisions).values():
        picks = _picks_at(rows, repeats)
        if picks is None:
            continue
        comparable += 1
        if len(set(picks)) > 1:
            flipped += 1
    if not comparable:
        return None, 0
    return flipped / comparable, comparable


def repeat_flip_rate(decisions):
    """Nondeterminism: repeats 0 and 1 are byte-identical requests."""
    return flip_rate(decisions, (0, 1))


def pick_flips(decisions):
    """``(rate, n)`` over ALL repeats: share of questions whose pick changed at least once.

    The board's repeat flip rate looks only at repeats 0 and 1 and its order flip rate
    only at the four distinct orders, so neither answers "did this question ever get two
    different answers out of five". This does, and it is the number a caller who has to
    trust one answer wants: a question that flipped once in five is a question this model
    does not actually have an opinion about.

    A question is comparable when at least two of its repeats came back with a readable
    pick; a failed or malformed repeat drops out of that question's comparison, and a
    question with fewer than two comparable repeats drops out of the rate entirely
    rather than counting as stable.
    """
    comparable = 0
    flipped = 0
    for rows in by_item(decisions).values():
        picks = [d["pick"] for d in rows
                 if not d.get("error") and not d.get("malformed")
                 and d.get("pick") is not None]
        if len(picks) < 2:
            continue
        comparable += 1
        if len(set(picks)) > 1:
            flipped += 1
    if not comparable:
        return None, 0
    return flipped / comparable, comparable


def confidence_swing(decisions):
    """How far one question's stated confidence moved across its repeats.

    ``{"max": .., "mean": .., "question": id, "over": n}``. The max is the largest
    spread any single question showed, which is the honest headline for "how stable is
    the number I would gate on": a model whose confidence on one question ran from 0.51
    to 0.99 across five identical-shaped requests has a gate that means something
    different on every call. The mean is beside it because one pathological question
    should not be read as the whole set.

    Over the calibratable decisions only, grouped per question, and a question with
    fewer than two of them contributes nothing. ``None`` when no question had two.
    """
    swings = []
    for item_id, rows in by_item(calibratable(decisions)).items():
        confidences = [float(d["confidence"]) for d in rows]
        if len(confidences) < 2:
            continue
        swings.append((max(confidences) - min(confidences), item_id))
    if not swings:
        return None
    worst, worst_id = max(swings)
    return {"max": worst, "mean": sum(s for s, _ in swings) / len(swings),
            "question": worst_id, "over": len(swings)}


def order_flip_rate(decisions):
    """Choice only: the four distinct option orders, repeats 0, 2, 3 and 4.

    It mixes option-order sensitivity with nondeterminism, so it is only readable next
    to the repeat flip rate. Returns ``(None, 0)`` for a set whose options are never
    reordered, which is every ``noul`` and ``score`` set.
    """
    rows = scored(decisions)
    if rows and rows[0].get("type") != CHOICE:
        return None, 0
    return flip_rate(decisions, (0, 2, 3, 4))


# ------------------------------------------------------------------ latency, cost

def percentile(values, fraction: float):
    """Nearest-rank: sort, take the ``ceil(fraction * n)``th, no interpolation.

    So a p95 is always a request that really took that long, which is the same rule
    ``metrics.percentile`` follows and the one the recipe states.
    """
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def latency(decisions) -> dict:
    """p50 and p95 of the round trip, over the decisions that came back."""
    walls = [d.get("wall_ms") for d in scored(decisions)
             if d.get("wall_ms") is not None]
    return {"p50_ms": percentile(walls, 0.5), "p95_ms": percentile(walls, 0.95)}


def tokens(decisions) -> dict:
    """Summed reported usage. A decision the endpoint reported nothing for adds 0."""
    return {"in": sum(int(d.get("tokens_in") or 0) for d in decisions),
            "out": sum(int(d.get("tokens_out") or 0) for d in decisions)}


def cost_usd(counts: dict, input_usd_per_mtok: float = 0.0,
             output_usd_per_mtok: float = 0.0) -> float:
    """Reported tokens at a posted rate. Both rates 0 means the column reads $0.

    Never an estimate: an endpoint that reports no usage contributes no tokens, and a
    backend nobody bills per token for is $0 rather than a guess at electricity.
    """
    return (counts.get("in", 0) * float(input_usd_per_mtok)
            + counts.get("out", 0) * float(output_usd_per_mtok)) / 1e6


def usd_per_1k(cost: float, decisions_count: int):
    """``$ per 1k decisions``: the board's cost column. None over no decisions."""
    if not decisions_count:
        return None
    return cost / decisions_count * 1000.0


def questions_per_second(decisions_count: int, seconds: float):
    """Decisions divided by the run's own wall clock, or None when it took no time.

    It is a throughput of the RUN and not of the endpoint: it moves with
    ``--concurrency`` and with whatever else the node was serving, which is why the
    record carries the concurrency next to it.
    """
    if not seconds or seconds <= 0 or not decisions_count:
        return None
    return decisions_count / float(seconds)


# ------------------------------------------------- against a gold DISTRIBUTION

#: Predicted probabilities are floored here before a log, so one confident miss is a
#: large KL rather than an infinite one that makes the mean unreadable. The floor is in
#: the record beside the number.
KL_FLOOR = 1e-6


def with_gold(decisions) -> list:
    """Scored decisions whose set ships a gold DISTRIBUTION, not just a label."""
    return [d for d in scored(decisions) if isinstance(d.get("gold"), dict)
            and d["gold"]]


def soft_accuracy(decisions):
    """Mean gold probability of the option the system picked.

    The figure a soft-labelled set asks for instead of accuracy: on a question where the
    teacher itself split 0.55/0.45, picking the 0.45 option is most of a right answer and
    exact-match accuracy calls it a miss. A pick the gold gave no mass contributes 0, and
    a decision with no pick (malformed, or a yes/no at exactly 0.5) contributes 0 too.
    """
    rows = with_gold(decisions)
    if not rows:
        return None
    total = 0.0
    for decision in rows:
        pick = decision.get("pick")
        if pick is not None:
            total += float(decision["gold"].get(pick, 0.0))
    return total / len(rows)


def total_variation(decisions):
    """Mean total-variation distance between the answer and the gold: ``0.5 * sum|p-g|``.

    0 is an exact match of the teacher's spread and 1 is disjoint. Unlike the KL it is
    bounded and symmetric, so it is the one to read when a system is confident and the
    gold is not.
    """
    rows = with_gold(decisions)
    if not rows:
        return None
    total = 0.0
    for decision in rows:
        vector = decision.get("vector") or uniform(decision["options"])
        gold = decision["gold"]
        total += 0.5 * sum(abs(float(vector.get(option, 0.0))
                               - float(gold.get(option, 0.0)))
                           for option in decision["options"])
    return total / len(rows)


def kl_from_gold(decisions, floor: float = KL_FLOOR):
    """Mean ``sum_k g_k * log(g_k / p_k)``, gold first, predictions floored.

    This is the number that separates "picks the right label" from "reproduces the
    teacher's uncertainty": a one-hot answer that happens to be right scores well on
    accuracy and badly here. A zero gold term contributes nothing, which is the usual
    convention; a zero PREDICTED probability under positive gold would be infinite, so
    predictions are floored at ``floor`` and the floor is recorded with the figure.
    """
    rows = with_gold(decisions)
    if not rows:
        return None
    total = 0.0
    for decision in rows:
        vector = decision.get("vector") or uniform(decision["options"])
        gold = decision["gold"]
        for option in decision["options"]:
            g = float(gold.get(option, 0.0))
            if g <= 0:
                continue
            p = max(float(vector.get(option, 0.0)), floor)
            total += g * math.log(g / p)
    return total / len(rows)


def brier_from_gold(decisions):
    """Mean ``sum_k (p_k - g_k)^2`` against the gold distribution.

    Explicitly OUR definition. A soft-gold set's own card may print a column called
    Brier without publishing the arithmetic behind it, so this number is not that
    number and must not be put in its column.
    """
    rows = with_gold(decisions)
    if not rows:
        return None
    total = 0.0
    for decision in rows:
        vector = decision.get("vector") or uniform(decision["options"])
        gold = decision["gold"]
        total += sum((float(vector.get(option, 0.0)) - float(gold.get(option, 0.0))) ** 2
                     for option in decision["options"])
    return total / len(rows)


def ordinal_mae(decisions):
    """JevBench's ordinal MAE: the probability-weighted level against the labeled one.

    ``mean |E[level] - label level|`` over the score decisions. Reported BESIDE argmax
    accuracy and never instead of it, which is the rule JevBench states: a model that
    spreads its mass either side of the right level is wrong on accuracy and close here,
    and both facts matter to something that sorts by the number. ``None`` when nothing
    here is a score question.
    """
    rows = [d for d in scored(decisions) if d.get("type") == SCORE]
    if not rows:
        return None
    total = 0.0
    for decision in rows:
        options = decision["options"]
        vector = decision.get("vector") or uniform(options)
        expected = sum(index * float(vector.get(option, 0.0))
                       for index, option in enumerate(options))
        total += abs(expected - options.index(decision["label"]))
    return total / len(rows)


def schema_validity(decisions, tolerance: float = SUM_TOLERANCE):
    """Share of answers that were a usable distribution over the exact label set.

    JevBench's name and JevBench's rule: a distribution has to cover the label set, sit
    in [0, 1] and sum to 1. ``tolerance`` is how far the sum may be off before the answer
    is invalid rather than renormalized. ``None`` when nothing came back.
    """
    rows = scored(decisions)
    if not rows:
        return None
    valid = 0
    for decision in rows:
        if decision.get("malformed"):
            continue
        total = decision.get("sum_before_normalize")
        if total is None:
            # A one-hot answer has no stated sum to check; it is a valid answer that
            # simply carries no spread, which `one_hot` already says.
            valid += 1
            continue
        if abs(float(total) - 1.0) <= tolerance + EPSILON:
            valid += 1
    return valid / len(rows)


def within_one_level(decisions):
    """Score questions only: share of picks within one level of the labeled one.

    An ordinal set's accuracy punishes a one-level miss exactly as hard as a four-level
    miss, which is what the ranked probability score exists to fix for the loss; this is
    the same correction for the pick. None when nothing here is a score question.
    """
    rows = [d for d in scored(decisions) if d.get("type") == SCORE]
    if not rows:
        return None
    hits = 0
    for decision in rows:
        options = decision["options"]
        pick = decision.get("pick")
        if pick is None or pick not in options:
            continue
        if abs(options.index(pick) - options.index(decision["label"])) <= 1:
            hits += 1
    return hits / len(rows)


def gold_block(decisions) -> dict:
    """The against-the-gold-distribution block, or ``{}`` when the set ships no gold."""
    rows = with_gold(decisions)
    if not rows:
        return {}
    return {"over": len(rows),
            "soft_accuracy": soft_accuracy(decisions),
            "total_variation": total_variation(decisions),
            "kl": kl_from_gold(decisions),
            "kl_floor": KL_FLOOR,
            "brier_vs_gold": brier_from_gold(decisions),
            "definition": "soft_accuracy is the gold probability of the pick; "
                          "total_variation is 0.5*sum|p-g|; kl is sum g*log(g/p) with "
                          "p floored at kl_floor; brier_vs_gold is sum (p-g)^2. These "
                          "are AINode's definitions, not a set card's column names"}


# ------------------------------------------------------------------ the block

def _round(value, places=4):
    return None if value is None else round(float(value), places)


#: Which recipe a metrics block's figures follow. Every block carries it, because two
#: numbers under one name and two recipes are the way a comparison becomes a lie.
RECIPE_JEVALS = "jevals-0.1.0"


def types_present(decisions) -> list:
    """The primitives among these decisions, in ``TYPES`` order."""
    seen = {d.get("type") for d in scored(decisions)}
    return [t for t in TYPES if t in seen]


def answer_spaces(decisions) -> dict:
    """``{name: [decisions]}`` grouped by ANSWER SPACE, in first-seen order.

    An answer space is one ``(type, options)`` pair, and it is the unit the Decision Score
    is really defined over: the label prior is the base rates of the labels IN that option
    list, so pooling two questions with different option lists would build a prior over an
    answer space neither of them has. A set asking five differently typed questions over
    one state has five of these, and the set's own card says to read every score against
    its own question rather than against the mean.

    The name is the question's own ``space`` (a mixed manifest writes ``group/question``),
    and ``type#n`` in first-seen order when the questions do not carry one, with the option
    list recorded beside the block so a reader can see which space it was.
    """
    out = {}
    keys = {}
    for decision in scored(decisions):
        key = (decision.get("type"), tuple(decision.get("options") or ()))
        if key not in keys:
            name = decision.get("space")
            if not name:
                seen = sum(1 for k in keys if k[0] == key[0])
                name = f"{key[0]}#{seen + 1}"
            keys[key] = name
            out[name] = []
        out[keys[key]].append(decision)
    return out


def summarize(decisions, input_usd_per_mtok: float = 0.0,
              output_usd_per_mtok: float = 0.0, seconds: float = 0.0,
              bins: int = BINS, recipe: str = RECIPE_JEVALS) -> dict:
    """One set's metrics block: every number the recipe defines, over its decisions.

    The guessing floor travels with the figures rather than sitting in a footnote:
    ``prior_accuracy`` is the base-rate answer's accuracy and ``loss_prior`` is the
    baseline the Decision Score is measured against, so a reader never has to go and
    find what 0 meant on this set.

    **A set holding more than one primitive is broken down by primitive**, because the
    two losses are different arithmetic, the label prior is per answer space, and the
    board's gate is per primitive. Such a block carries ``type: "mixed"``, a ``types``
    map of one full block each, and a ``decision_score`` that is the plain mean of the
    per-primitive scores, which is the recipe's rule for a tab holding more than one
    task. Its own accuracy and latency are over all the decisions, because those do mean
    the same thing across primitives.
    """
    rows = scored(decisions)
    groups = by_item(rows)
    present = types_present(decisions)
    spaces = answer_spaces(decisions)
    if len(spaces) > 1:
        return _summarize_multi(decisions, present, spaces, input_usd_per_mtok,
                                output_usd_per_mtok, seconds, bins, recipe)
    qtype = present[0] if present else None
    system_loss = mean_item_loss(decisions)
    baseline_loss = prior_loss(decisions)
    token_counts = tokens(rows)
    cost = cost_usd(token_counts, input_usd_per_mtok, output_usd_per_mtok)
    gate = PUBLISHED_GATES.get(qtype) if qtype else None
    repeat_flips, repeat_flip_n = repeat_flip_rate(decisions)
    order_flips, order_flip_n = order_flip_rate(decisions)
    any_flips, any_flip_n = pick_flips(decisions)
    swing = confidence_swing(decisions)
    block = {
        "recipe": recipe,
        "type": qtype,
        "items": len(groups),
        "repeats": max((len(r) for r in groups.values()), default=0),
        "decisions": len(rows),
        "failed": len(failed(decisions)),
        "malformed": sum(1 for d in rows if d.get("malformed")),
        "one_hot": sum(1 for d in rows if d.get("one_hot")),
        "calibrated_over": len(calibratable(decisions)),
        "accuracy": _round(accuracy(decisions)),
        # JevBench calls this majority_class_accuracy and Jevals calls its baseline the
        # label prior. Same number on a set like these, and the record carries both
        # names so a reader of either board knows what it is.
        "prior_accuracy": _round(prior_accuracy(decisions)),
        "majority_class_accuracy": _round(prior_accuracy(decisions)),
        "schema_validity": _round(schema_validity(decisions)),
        "schema_validity_strict": _round(
            schema_validity(decisions, SUM_TOLERANCE_STRICT)),
        "decision_score": _round(decision_score(system_loss, baseline_loss), 2),
        "loss": _round(system_loss, 6),
        "loss_prior": _round(baseline_loss, 6),
        "ece_points": _round(ece_points(decisions, bins), 2),
        "bins": [{**b, "accuracy": _round(b["accuracy"]),
                  "confidence": _round(b["confidence"])}
                 for b in reliability(decisions, bins)],
        "handoff_95": None,
        "gate": None,
        "gate_local": gate_local(decisions),
        "pick_flip_rate": _round(any_flips),
        "pick_flip_over": any_flip_n,
        "confidence_swing": (None if swing is None else
                             {"max": _round(swing["max"]),
                              "mean": _round(swing["mean"]),
                              "question": swing["question"],
                              "over": swing["over"]}),
        "repeat_flip_rate": _round(repeat_flips),
        "repeat_flip_over": repeat_flip_n,
        "order_flip_rate": _round(order_flips),
        "order_flip_over": order_flip_n,
        "questions_per_second": _round(questions_per_second(len(rows), seconds), 3),
        "tokens": token_counts,
        "cost_usd": round(cost, 6),
        "usd_per_1k_decisions": _round(usd_per_1k(cost, len(rows)), 6),
    }
    hand = handoff(decisions)
    if hand:
        block["handoff_95"] = {"threshold": hand["threshold"],
                               "share": _round(hand["share"]),
                               "n": hand["n"],
                               "accuracy": _round(hand["accuracy"])}
    coverage = gate_coverage(decisions, gate)
    if coverage:
        block["gate"] = {"threshold": coverage["gate"],
                         "source": PUBLISHED_GATE_SOURCE,
                         "coverage": _round(coverage["coverage"]),
                         "n": coverage["n"],
                         "accuracy": _round(coverage["accuracy"])}
    elif qtype and gate is None:
        block["gate"] = {"threshold": None, "source": PUBLISHED_GATE_SOURCE,
                         "coverage": None, "n": 0, "accuracy": None}
    gold = gold_block(decisions)
    if gold:
        block["vs_gold"] = {key: (_round(value, 6) if isinstance(value, float)
                                  else value)
                            for key, value in gold.items()}
    if qtype == SCORE:
        block["within_one_level"] = _round(within_one_level(decisions))
        block["ordinal_mae"] = _round(ordinal_mae(decisions))
    block.update(latency(decisions))
    return block


def _summarize_multi(decisions, present, spaces, input_usd_per_mtok,
                     output_usd_per_mtok, seconds, bins, recipe) -> dict:
    """A set holding more than one ANSWER SPACE: one block per space, plus the means.

    The Decision Score is the plain mean over the spaces, which is the recipe's rule for
    more than one task applied to the unit the score is defined over. ``types`` rolls the
    spaces up per primitive the way a board shows one, and the pooled ``loss`` and
    ``loss_prior`` are null because a loss over two different answer spaces is not a
    number.
    """
    rows = scored(decisions)
    per_space = {}
    for name, mine in spaces.items():
        block = summarize(mine, input_usd_per_mtok=input_usd_per_mtok,
                          output_usd_per_mtok=output_usd_per_mtok,
                          seconds=seconds, bins=bins, recipe=recipe)
        block["options"] = list(mine[0].get("options") or ())
        per_space[name] = block
    space_scores = [b["decision_score"] for b in per_space.values()]
    mean_score = (round(sum(space_scores) / len(space_scores), 2)
                  if space_scores and all(s is not None for s in space_scores)
                  else None)
    per_type = {}
    for qtype in present:
        mine = [b for b in per_space.values() if b["type"] == qtype]
        scores = [b["decision_score"] for b in mine]
        per_type[qtype] = {
            "recipe": recipe, "type": qtype, "spaces": len(mine),
            "decisions": sum(b["decisions"] for b in mine),
            "accuracy": _round(accuracy([d for d in decisions
                                         if d.get("type") == qtype])),
            "prior_accuracy": _round(
                sum(b["prior_accuracy"] or 0.0 for b in mine) / len(mine)
                if mine else None),
            "decision_score": (round(sum(scores) / len(scores), 2)
                               if scores and all(s is not None for s in scores)
                               else None),
            "ece_points": _round(ece_points([d for d in decisions
                                             if d.get("type") == qtype], bins), 2),
        }
    token_counts = tokens(rows)
    cost = cost_usd(token_counts, input_usd_per_mtok, output_usd_per_mtok)
    any_flips, any_flip_n = pick_flips(decisions)
    swing = confidence_swing(decisions)
    block = {
        "recipe": recipe,
        "type": "mixed" if len(present) > 1 else (present[0] if present else None),
        "spaces": per_space,
        "types": per_type,
        "items": len(by_item(rows)),
        "repeats": max((len(r) for r in by_item(rows).values()), default=0),
        "decisions": len(rows),
        "failed": len(failed(decisions)),
        "malformed": sum(1 for d in rows if d.get("malformed")),
        "one_hot": sum(1 for d in rows if d.get("one_hot")),
        "calibrated_over": len(calibratable(decisions)),
        "accuracy": _round(accuracy(decisions)),
        "prior_accuracy": _round(
            sum(b["prior_accuracy"] or 0.0 for b in per_space.values())
            / len(per_space) if per_space else None),
        "majority_class_accuracy": _round(
            sum(b["prior_accuracy"] or 0.0 for b in per_space.values())
            / len(per_space) if per_space else None),
        "schema_validity": _round(schema_validity(decisions)),
        "schema_validity_strict": _round(
            schema_validity(decisions, SUM_TOLERANCE_STRICT)),
        "decision_score": mean_score,
        "decision_score_is": "the plain mean of the per-answer-space Decision Scores, "
                             "which is the recipe's rule for more than one task applied "
                             "to the unit the score is defined over",
        "loss": None,
        "loss_prior": None,
        "ece_points": _round(ece_points(decisions, bins), 2),
        "bins": [{**b, "accuracy": _round(b["accuracy"]),
                  "confidence": _round(b["confidence"])}
                 for b in reliability(decisions, bins)],
        "handoff_95": None,
        "gate": None,
        "gate_local": gate_local(decisions),
        "pick_flip_rate": _round(any_flips),
        "pick_flip_over": any_flip_n,
        "confidence_swing": (None if swing is None else
                             {"max": _round(swing["max"]),
                              "mean": _round(swing["mean"]),
                              "question": swing["question"],
                              "over": swing["over"]}),
        "repeat_flip_rate": _round(repeat_flip_rate(decisions)[0]),
        "repeat_flip_over": repeat_flip_rate(decisions)[1],
        "order_flip_rate": None,
        "order_flip_over": 0,
        "questions_per_second": _round(questions_per_second(len(rows), seconds), 3),
        "tokens": token_counts,
        "cost_usd": round(cost, 6),
        "usd_per_1k_decisions": _round(usd_per_1k(cost, len(rows)), 6),
    }
    hand = handoff(decisions)
    if hand:
        block["handoff_95"] = {"threshold": hand["threshold"],
                               "share": _round(hand["share"]), "n": hand["n"],
                               "accuracy": _round(hand["accuracy"])}
    gold = gold_block(decisions)
    if gold:
        block["vs_gold"] = {key: (_round(value, 6) if isinstance(value, float)
                                  else value)
                            for key, value in gold.items()}
    if SCORE in present:
        block["within_one_level"] = _round(within_one_level(decisions))
        block["ordinal_mae"] = _round(ordinal_mae(decisions))
    block.update(latency(decisions))
    return block


def summarize_sets(decisions, set_names, **kw) -> dict:
    """One block per set, in the order given, and nothing for a set with no decisions.

    A set nobody ran is absent rather than a block of nulls, the same rule the rest of
    ``bench/`` follows: a measurement nobody took is not a zero.
    """
    out = {}
    for name in set_names:
        mine = [d for d in decisions if d.get("set") == name]
        if mine:
            out[name] = summarize(mine, **kw)
    return out


def mean_decision_score(blocks) -> dict:
    """The plain mean of the set Decision Scores, and which sets went into it.

    The recipe's rule for a tab with more than one task. It is only reported when every
    set asked for has a score, because a mean over two of three sets is a different
    number wearing the same name.
    """
    scores = [(name, block.get("decision_score")) for name, block in blocks.items()]
    have = [(name, value) for name, value in scores if value is not None]
    out = {"sets": [name for name, _ in scores],
           "scored": [name for name, _ in have], "mean_decision_score": None}
    if have and len(have) == len(scores):
        out["mean_decision_score"] = round(sum(v for _, v in have) / len(have), 2)
    return out


__all__ = ["BINS", "CHOICE", "EPSILON", "GATE_MAX_ERROR", "GATE_MIN_DECISIONS",
           "GRID", "HANDOFF_ACCURACY", "HANDOFF_MIN_DECISIONS", "KL_FLOOR", "NOUL",
           "PUBLISHED_GATES", "PUBLISHED_GATE_SOURCE", "RECIPE_JEVALS", "REPEATS",
           "SCORE", "SUM_TOLERANCE", "SUM_TOLERANCE_STRICT", "TYPES", "above",
           "accuracy", "bin_index", "brier_from_gold", "brier_loss", "by_item",
           "calibratable", "confidence_swing", "cost_usd", "decision_loss",
           "decision_score", "ece", "ece_points", "failed", "flip_rate",
           "gate_coverage", "gate_local", "gold_block", "grid_thresholds", "handoff",
           "is_correct", "is_probability", "kl_from_gold", "label_prior", "latency",
           "loss_for", "mean_decision_score", "mean_item_loss", "normalize_vector",
           "order_flip_rate", "ordinal_mae", "percentile", "pick_flips",
           "pick_from_vector", "prior_accuracy", "prior_loss", "questions_per_second",
           "reliability", "repeat_flip_rate", "round_half_up", "rps_loss",
           "schema_validity", "scored", "soft_accuracy", "summarize", "summarize_sets",
           "answer_spaces", "tokens", "total_variation", "types_present", "uniform",
           "usd_per_1k", "with_gold", "within_one_level"]
