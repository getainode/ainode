/* ============================================================
 * AINode metrics charts: the shaping, with no DOM in it.
 *
 * Everything between /api/metrics/history and a canvas that is arithmetic
 * rather than pixels lives here: which window and step to ask the store for,
 * how a grid of points becomes drawable segments, where the gaps are, and what
 * a counter's rate is between two samples. metrics.js draws; this file decides
 * what there is to draw.
 *
 * Split out for one reason: it can then be run under node with plain values and
 * no browser (tests/test_metrics_chart.py runs it that way), and the rules that
 * matter are rules about values, not about pixels.
 *
 * The rule the whole file is written around, the same one the store and the
 * collector are written around (#215, #234): a figure the node could not
 * measure is null, and null is never turned into a number. It is not carried
 * forward from the previous sample, not interpolated across, and never drawn at
 * the axis: a zero on a GPU chart reads as an idle GPU on a node that is
 * serving. A gap in the data is a gap in the line, and a series with nothing
 * measured in the whole window is reported as such so the panel can say so in
 * words.
 * ============================================================ */

(function (global) {
  'use strict';

  // ----------------------------------------------------------------------
  // What the view asks the store for
  // ----------------------------------------------------------------------

  // The ranges of the picker. `step` is chosen so the store answers from the
  // table it should: metrics/store.py picks the one minute roll-up whenever the
  // step is 60 seconds or more, or the window reaches past the raw retention,
  // so 1 h is the only range served raw and the rest are roll-ups by
  // construction rather than by luck. `resolution` states the same choice
  // explicitly, so a node configured with a shorter raw retention answers the
  // same shape as one at the default.
  //
  // `points` per range (seconds / step) is deliberately a few hundred: the store
  // splits ONE budget (store.MAX_HISTORY_POINTS) across every series asked for,
  // and coarsens the step when the request is over it. A range whose points
  // times series count exceeds the budget would silently be answered at a
  // coarser step than the picker says. tests/test_metrics_chart.py pins the
  // arithmetic against the store's own constant.
  var RANGES = [
    { key: '1h', label: '1 h', seconds: 3600, step: 15, resolution: 'raw', refreshMs: 15000 },
    { key: '6h', label: '6 h', seconds: 21600, step: 60, resolution: '1m', refreshMs: 60000 },
    { key: '24h', label: '24 h', seconds: 86400, step: 300, resolution: '1m', refreshMs: 60000 },
    { key: '7d', label: '7 d', seconds: 604800, step: 3600, resolution: '1m', refreshMs: 300000 },
  ];

  // Every series the view reads, named exactly as the store names them
  // (metrics/store.py: GPU_KEYS, REQUEST_KEYS, LATENCY_KEYS, uptime).
  //
  // uptime_seconds is here because it is what makes a restart readable: it
  // sawtooths back to zero the moment the process came up, so a gap in the other
  // series can be told apart from a dead GPU.
  //
  // NOT here: requests.tokens_generated and requests.tokens_per_second. The
  // store keeps both and the collector reports both, but nothing in the product
  // ever passes `tokens_generated` to `MetricsCollector.record_request`, so both
  // are 0 on every node forever. Drawn, that is a flat line at zero saying "this
  // node generated no tokens", when the truth is "no code path counts tokens".
  // A chart may not say the first when it means the second. Wire the proxy to
  // pass the usage block through (and tally the SSE path), and then a tokens
  // panel is one entry here and one in metrics.js::PANELS.
  var SERIES = [
    'gpu.memory_used_mb',
    'gpu.memory_total_mb',
    'gpu.system_memory_used_mb',
    'gpu.utilization_percent',
    'gpu.temperature_c',
    'requests.total',
    'requests.errors',
    'requests.latency_ms.p50',
    'requests.latency_ms.p95',
    'requests.latency_ms.p99',
    'uptime_seconds',
  ];

  function rangeFor(key) {
    for (var i = 0; i < RANGES.length; i++) {
      if (RANGES[i].key === key) return RANGES[i];
    }
    return RANGES[0];
  }

  /** Points the store will put on the grid for one range, ends included. */
  function expectedPoints(range) {
    var r = (typeof range === 'string') ? rangeFor(range) : (range || RANGES[0]);
    return Math.floor(r.seconds / r.step) + 1;
  }

  /**
   * The query string for one range, one node and a list of series.
   *
   * `since` is always RELATIVE (-3600s, never a timestamp): the window is the
   * node's own last hour, decided by the clock that took the samples. A browser
   * whose clock is minutes off would otherwise ask for a window the node has not
   * reached yet and get a chart of nulls.
   */
  function historyQuery(rangeKey, series, nodeId) {
    var range = (typeof rangeKey === 'string') ? rangeFor(rangeKey) : (rangeKey || RANGES[0]);
    var names = (series && series.length) ? series : SERIES;
    var parts = [
      'series=' + encodeURIComponent(names.join(',')),
      'since=-' + range.seconds + 's',
      'step=' + range.step + 's',
      'resolution=' + encodeURIComponent(range.resolution),
    ];
    if (nodeId) parts.push('node=' + encodeURIComponent(nodeId));
    return parts.join('&');
  }

  // ----------------------------------------------------------------------
  // One payload into drawable points
  // ----------------------------------------------------------------------

  /**
   * One series of a /api/metrics/history payload as [{t: ms, v: number|null}].
   *
   * Timestamps become milliseconds because that is what Date wants and what the
   * axis labels are formatted from. A slot the store answered null stays null in
   * the list: dropping it would close the gap, and the length of the grid is
   * also how the caller knows how much of the window the node was up for.
   */
  function toPoints(payload, name) {
    var grid = (payload && payload.series && payload.series[name]) || null;
    if (!grid || !grid.length) return [];
    var out = [];
    for (var i = 0; i < grid.length; i++) {
      var slot = grid[i] || {};
      var value = (slot.value === undefined) ? null : slot.value;
      if (value !== null && !isFinite(value)) value = null;
      out.push({ t: Number(slot.ts) * 1000, v: (value === null ? null : Number(value)) });
    }
    return out;
  }

  /** Multiply every measured value, keeping nulls null (MB to GB, and so on). */
  function scalePoints(points, factor) {
    return (points || []).map(function (p) {
      return { t: p.t, v: (p.v === null || p.v === undefined) ? null : p.v * factor };
    });
  }

  /**
   * The contiguous runs of measured points, which is what a line is drawn from.
   *
   * A run ends at a null and at a hole in the grid wider than `maxGapMs`. Both
   * are gaps, and a line that spans one is a claim about a value nobody
   * measured. A single measured point between two nulls comes back as a run of
   * one, so the caller can draw it as a dot rather than lose it: a node that was
   * up for one tick of the window did measure something.
   */
  function segments(points, maxGapMs) {
    var out = [];
    var run = [];
    var limit = (maxGapMs && maxGapMs > 0) ? maxGapMs : Infinity;
    var previous = null;
    for (var i = 0; i < (points || []).length; i++) {
      var p = points[i];
      if (p.v === null || p.v === undefined) {
        if (run.length) out.push(run);
        run = [];
        previous = null;
        continue;
      }
      if (previous !== null && (p.t - previous) > limit) {
        if (run.length) out.push(run);
        run = [];
      }
      run.push(p);
      previous = p.t;
    }
    if (run.length) out.push(run);
    return out;
  }

  /**
   * What is in a series: enough for a panel to decide what to say.
   *
   * `measured === 0` is the case the panel has to put into words. On a DGX Spark
   * the driver exposes no GPU utilisation counter at all, so that series is null
   * in every slot of every window forever, and a chart that drew it as a flat
   * line at zero would report an idle GPU. The panel says the node does not
   * expose it instead.
   */
  function summary(points) {
    var out = {
      count: (points || []).length,
      measured: 0,
      missing: 0,
      min: null,
      max: null,
      first: null,
      last: null,
      lastAt: null,
      firstAt: null,
    };
    for (var i = 0; i < (points || []).length; i++) {
      var p = points[i];
      if (p.v === null || p.v === undefined) { out.missing += 1; continue; }
      out.measured += 1;
      if (out.min === null || p.v < out.min) out.min = p.v;
      if (out.max === null || p.v > out.max) out.max = p.v;
      if (out.first === null) { out.first = p.v; out.firstAt = p.t; }
      out.last = p.v;
      out.lastAt = p.t;
    }
    return out;
  }

  /**
   * How much of the window the node could have measured, and how much it did.
   *
   * A missing slot has two very different causes and a panel should not report
   * them as one:
   *
   *   * BEFORE the node's oldest sample: it was not recording yet. A node that
   *     came up ten minutes ago has 50 missing minutes in a one hour window and
   *     nothing is wrong. Saying "83% of this window was not measured" about that
   *     is true and useless.
   *   * AFTER it: the sampler was down, the node was down, or the figure was not
   *     readable. That is a gap worth naming.
   *
   * *startedAtMs* is the store's oldest sample (``store.oldest_sample`` times
   * 1000). Pass null when the node does not say, and every hole counts as a gap,
   * which is the conservative reading.
   */
  function coverage(points, startedAtMs) {
    var out = { slots: 0, measured: 0, missing: 0, beforeStart: 0, gaps: 0 };
    var start = (typeof startedAtMs === 'number' && isFinite(startedAtMs)) ? startedAtMs : null;
    for (var i = 0; i < (points || []).length; i++) {
      var p = points[i];
      out.slots += 1;
      if (p.v !== null && p.v !== undefined) { out.measured += 1; continue; }
      out.missing += 1;
      if (start !== null && p.t < start) out.beforeStart += 1;
      else out.gaps += 1;
    }
    return out;
  }

  /**
   * A monotonic counter turned into a rate, per `perSeconds` seconds.
   *
   * The store keeps `requests.total`, `requests.errors` and
   * `requests.tokens_generated` as the counters the collector holds, which climb
   * for the life of the process. A rate is the only readable form of those, and
   * the arithmetic has three cases that must not become numbers:
   *
   *   * either end of the interval null: the rate is null, because part of the
   *     interval was not measured.
   *   * the counter went DOWN: the process restarted and the counter is a new
   *     one. The rate over that interval is unknowable, so it is null, never the
   *     negative number the subtraction gives or the zero that hides it.
   *   * the two samples are further apart than `maxGapMs`: the sampler was down
   *     in between, so the average over the hole would be spread across time
   *     nobody measured.
   *
   * The first point of the result is always null: a rate needs two samples, and
   * the alternative is to report the counter's whole lifetime as if it happened
   * in one step.
   */
  function rate(points, perSeconds, maxGapMs) {
    var per = (perSeconds && perSeconds > 0) ? perSeconds : 1;
    var limit = (maxGapMs && maxGapMs > 0) ? maxGapMs : Infinity;
    var out = [];
    var prev = null;
    for (var i = 0; i < (points || []).length; i++) {
      var p = points[i];
      if (i === 0) { out.push({ t: p.t, v: null }); }
      else if (p.v === null || p.v === undefined || prev === null) { out.push({ t: p.t, v: null }); }
      else {
        var dt = (p.t - prev.t) / 1000;
        var dv = p.v - prev.v;
        if (dt <= 0 || (p.t - prev.t) > limit || dv < 0) out.push({ t: p.t, v: null });
        else out.push({ t: p.t, v: (dv / dt) * per });
      }
      prev = (p.v === null || p.v === undefined) ? null : p;
    }
    return out;
  }

  /**
   * The y range for one chart, across however many lines it draws.
   *
   * Returns null when not one of the lines measured anything, which is the
   * panel's cue to say so rather than to draw an axis around no data. A series
   * whose values never move (a total that held steady all day) still gets a
   * usable axis via `minSpan`, because a flat line at the top of an axis of zero
   * height is not readable.
   */
  function extent(seriesList, opts) {
    var options = opts || {};
    var min = null;
    var max = null;
    var list = seriesList || [];
    for (var i = 0; i < list.length; i++) {
      var s = summary(list[i]);
      if (!s.measured) continue;
      if (min === null || s.min < min) min = s.min;
      if (max === null || s.max > max) max = s.max;
    }
    if (min === null) return null;
    if (options.zeroFloor) min = Math.min(0, min);
    if (typeof options.max === 'number') max = Math.max(max, options.max);
    if (typeof options.min === 'number') min = Math.min(min, options.min);
    var span = max - min;
    var minSpan = (typeof options.minSpan === 'number') ? options.minSpan : 1;
    if (span < minSpan) {
      var mid = (max + min) / 2;
      min = mid - minSpan / 2;
      max = mid + minSpan / 2;
      if (options.zeroFloor && min < 0) { min = 0; max = Math.max(minSpan, max); }
      span = max - min;
    }
    var pad = (typeof options.pad === 'number') ? options.pad : 0.08;
    if (pad > 0) {
      max = max + span * pad;
      if (!options.zeroFloor || min > 0) min = min - span * pad;
      if (options.zeroFloor && min < 0) min = 0;
    }
    return { min: min, max: max };
  }

  /** Round axis values: 1, 2, 2.5 or 5 times a power of ten, inside [min, max]. */
  function ticks(min, max, count) {
    if (!(isFinite(min) && isFinite(max)) || max <= min) return [];
    var want = Math.max(2, count || 4);
    var raw = (max - min) / want;
    var magnitude = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var normalized = raw / magnitude;
    var stepMultiple = normalized > 5 ? 10 : normalized > 2.5 ? 5 : normalized > 2 ? 2.5 : normalized > 1 ? 2 : 1;
    var step = stepMultiple * magnitude;
    var out = [];
    var first = Math.ceil(min / step) * step;
    for (var v = first; v <= max + step * 1e-9; v += step) {
      // Re-round: repeated addition of a non-binary step drifts (0.30000000000000004).
      out.push(Math.round(v / step) * step);
      if (out.length > 40) break;
    }
    return out;
  }

  /** Evenly spaced instants across the window, for the x labels. */
  function timeTicks(fromMs, toMs, count) {
    var want = Math.max(2, count || 5);
    if (!(isFinite(fromMs) && isFinite(toMs)) || toMs <= fromMs) return [];
    var out = [];
    for (var i = 0; i < want; i++) out.push(fromMs + (toMs - fromMs) * (i / (want - 1)));
    return out;
  }

  /** A short, honest number: null stays null, and nothing is padded with zeros. */
  function fmt(value, digits) {
    if (value === null || value === undefined || !isFinite(value)) return null;
    var places = (typeof digits === 'number') ? digits : (Math.abs(value) >= 100 ? 0 : Math.abs(value) >= 10 ? 1 : 2);
    var rounded = Number(value.toFixed(places));
    return String(rounded);
  }

  /**
   * The window the x axis covers.
   *
   * Taken from the payload's own `since` / `until` rather than from the points,
   * so a node that has only been up for two minutes of a one hour window draws
   * two minutes of line against an hour of axis instead of stretching it to fill
   * the panel. That is the difference between "this node has an hour of history"
   * and "this node came up two minutes ago", which the chart should not hide.
   */
  function windowOf(payload, range) {
    var r = (typeof range === 'string') ? rangeFor(range) : range;
    var until = (payload && isFinite(payload.until)) ? Number(payload.until) * 1000 : Date.now();
    var since = (payload && isFinite(payload.since)) ? Number(payload.since) * 1000 : null;
    if (since === null) since = until - ((r ? r.seconds : 3600) * 1000);
    return { from: since, to: until };
  }

  /**
   * How wide a hole has to be before a line breaks, for one payload.
   *
   * Two steps plus a little: one missing sample is a gap the line has to break
   * across (it is a tick nobody measured), and the slack keeps a grid whose
   * timestamps are a fraction off from breaking every segment.
   */
  function gapLimitMs(payload, range) {
    var r = (typeof range === 'string') ? rangeFor(range) : range;
    var step = (payload && isFinite(payload.step) && payload.step > 0)
      ? Number(payload.step)
      : (r ? r.step : 15);
    return step * 1000 * 2.5;
  }

  var AINodeMetricsData = {
    RANGES: RANGES,
    SERIES: SERIES,
    rangeFor: rangeFor,
    expectedPoints: expectedPoints,
    historyQuery: historyQuery,
    toPoints: toPoints,
    scalePoints: scalePoints,
    segments: segments,
    summary: summary,
    coverage: coverage,
    rate: rate,
    extent: extent,
    ticks: ticks,
    timeTicks: timeTicks,
    fmt: fmt,
    windowOf: windowOf,
    gapLimitMs: gapLimitMs,
  };

  global.AINodeMetricsData = AINodeMetricsData;
  // Loadable by a node test without a browser.
  if (typeof module !== 'undefined' && module.exports) module.exports = AINodeMetricsData;
})(typeof window !== 'undefined' ? window : globalThis);
