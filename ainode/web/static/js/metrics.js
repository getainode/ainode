/* AINode Metrics view: the node's own history, drawn.
 *
 * Self-contained on purpose, the same deal bench.js has: app.js owns one nav
 * pill and one case arm, every other line of this view lives here, so this file
 * and the chat, auth or TLS panels can be worked on at the same time without
 * touching each other.
 *
 * Where the numbers come from: GET /api/metrics/history, which serves what the
 * node wrote to <AINODE_HOME>/metrics.db (#234). That is the only source. There
 * is no ring buffer of live polls behind these charts, because a buffer and a
 * store would be two sources for one line, and the store already holds 48 hours
 * of raw samples and 30 days of one minute roll-ups. The live edge is at most
 * one sampling interval old, and the view refetches on its own cadence.
 *
 * Canvas, hand drawn, no library: the repo has none and a chart of one node's
 * telemetry does not justify the first one. Colours come from the CSS tokens
 * through getComputedStyle, so the charts follow the design system rather than
 * carrying a second copy of it.
 *
 * The one rule to keep in mind when editing this file: a figure the node could
 * not measure is null, and null is a HOLE. It is not drawn at the axis, not
 * joined across, and not averaged into a neighbour. A panel whose series
 * measured nothing in the window says so in words. Zero on these charts means
 * the node measured zero.
 */

const AINodeMetrics = {
  state: {
    built: false,
    range: '1h',
    nodeId: null,          // null means this node, the one serving the page
    payload: null,
    snapshot: null,        // /api/metrics, for the "used" source and the live line
    error: null,
    loading: false,
    fetchedAt: 0,
    resizeBound: false,
  },

  // Each panel is one canvas, one legend and one note. `lines` is a pure
  // function of the payload so the drawing has no opinions about the data.
  PANELS: [
    {
      key: 'memory',
      title: 'GPU memory',
      hint: 'in use against the total this node reports',
      unit: ' GB',
      axis: { zeroFloor: true, minSpan: 4, pad: 0.06 },
    },
    {
      key: 'util',
      title: 'GPU utilization',
      hint: 'percent busy, averaged across the devices that answer',
      unit: '%',
      axis: { zeroFloor: true, min: 0, max: 100, minSpan: 10, pad: 0.04 },
    },
    {
      key: 'temp',
      title: 'Temperature',
      hint: 'the hottest device on the node',
      unit: '°C',
      axis: { minSpan: 10, pad: 0.12 },
    },
    {
      key: 'requests',
      title: 'Request rate',
      hint: 'requests and errors per minute, from the counters this node keeps',
      unit: '/min',
      axis: { zeroFloor: true, minSpan: 2, pad: 0.1 },
    },
    {
      key: 'latency',
      title: 'Latency percentiles',
      hint: 'p50, p95 and p99 over the requests this node has timed',
      unit: ' ms',
      axis: { zeroFloor: true, minSpan: 20, pad: 0.12 },
    },
    {
      key: 'uptime',
      title: 'Process uptime',
      hint: 'sawtooths to zero at a restart, which is how a gap above reads as one',
      unit: ' h',
      axis: { zeroFloor: true, minSpan: 0.5, pad: 0.08 },
    },
  ],

  // There is no tokens per second panel, and that is deliberate. See the note on
  // SERIES in metrics-data.js: the counter behind it is never incremented by
  // anything in the product, so the panel would be a flat zero reading "this node
  // generated no tokens" when the truth is "nothing counts tokens". Uptime took
  // the slot because it is a series that is actually measured, and because it is
  // what tells a reader whether a gap in the charts above was a restart.

  // ======================================================================
  //  ENTRY POINT (called from app.js's refresh switch, once per poll tick)
  // ======================================================================

  render(app) {
    this.app = app || window.AINode || null;
    var mount = document.getElementById('metrics-content');
    if (!mount) return;
    if (!this.state.built) {
      mount.innerHTML = this.shellHtml();
      this.bind();
      this.state.built = true;
      this.load(true);
      return;
    }
    // A poll tick refreshes the node pills (a peer can appear or go) and refetches
    // only when this range's own cadence says so: a 7 day chart has nothing to
    // gain from a request every five seconds.
    this.renderNodePills();
    var range = this.data().rangeFor(this.state.range);
    if (!this.state.loading && (Date.now() - this.state.fetchedAt) >= range.refreshMs) {
      this.load(false);
    }
  },

  data() {
    return window.AINodeMetricsData;
  },

  esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  },

  // ======================================================================
  //  MARKUP
  // ======================================================================

  shellHtml() {
    var D = this.data();
    var html = '';
    html += '<div class="view-header">';
    html += '  <h2>Metrics</h2>';
    html += '  <p>What this node measured, read back from its own retained history.</p>';
    html += '</div>';
    html += '<div class="metrics-toolbar">';
    html += '  <div class="metrics-ranges" id="metrics-ranges">';
    for (var i = 0; i < D.RANGES.length; i++) {
      var r = D.RANGES[i];
      html += '    <button class="metrics-range-pill' + (r.key === this.state.range ? ' active' : '') +
              '" data-range="' + this.esc(r.key) + '">' + this.esc(r.label) + '</button>';
    }
    html += '  </div>';
    html += '  <div class="metrics-nodes" id="metrics-nodes"></div>';
    html += '  <div class="metrics-toolbar-status mono" id="metrics-status"></div>';
    html += '</div>';
    html += '<div class="metrics-grid" id="metrics-grid">';
    for (var p = 0; p < this.PANELS.length; p++) {
      var panel = this.PANELS[p];
      html += '  <section class="metrics-panel" data-panel="' + this.esc(panel.key) + '">';
      html += '    <div class="metrics-panel-head">';
      html += '      <div class="metrics-panel-titles">';
      html += '        <h3 class="metrics-panel-title">' + this.esc(panel.title) + '</h3>';
      html += '        <span class="metrics-panel-hint">' + this.esc(panel.hint) + '</span>';
      html += '      </div>';
      html += '      <div class="metrics-legend mono" id="metrics-legend-' + this.esc(panel.key) + '"></div>';
      html += '    </div>';
      html += '    <div class="metrics-canvas-wrap">';
      html += '      <canvas class="metrics-canvas" id="metrics-canvas-' + this.esc(panel.key) + '"></canvas>';
      html += '      <div class="metrics-empty" id="metrics-empty-' + this.esc(panel.key) + '" style="display:none"></div>';
      html += '    </div>';
      html += '    <div class="metrics-panel-note" id="metrics-note-' + this.esc(panel.key) + '"></div>';
      html += '  </section>';
    }
    html += '</div>';
    html += '<div class="metrics-store mono" id="metrics-store"></div>';
    return html;
  },

  bind() {
    var self = this;
    var ranges = document.getElementById('metrics-ranges');
    if (ranges) {
      ranges.addEventListener('click', function (e) {
        var btn = e.target && e.target.closest ? e.target.closest('.metrics-range-pill') : null;
        if (!btn || !btn.dataset.range) return;
        self.state.range = btn.dataset.range;
        ranges.querySelectorAll('.metrics-range-pill').forEach(function (el) {
          el.classList.toggle('active', el.dataset.range === self.state.range);
        });
        self.load(true);
      });
    }
    var nodes = document.getElementById('metrics-nodes');
    if (nodes) {
      nodes.addEventListener('click', function (e) {
        var btn = e.target && e.target.closest ? e.target.closest('.metrics-node-pill') : null;
        if (!btn) return;
        var id = btn.dataset.nodeId || '';
        self.state.nodeId = (id === '' || id === self.localNodeId()) ? null : id;
        self.renderNodePills();
        self.load(true);
      });
    }
    if (!this.state.resizeBound) {
      this.state.resizeBound = true;
      window.addEventListener('resize', function () {
        if (self._resizeTimer) clearTimeout(self._resizeTimer);
        self._resizeTimer = setTimeout(function () { self.draw(); }, 150);
      });
    }
    this.renderNodePills();
  },

  // ======================================================================
  //  THE FLEET: one node's charts at a time, any node in the cluster
  // ======================================================================
  //
  // A peer's history is served BY THAT PEER. This node asks it for the same
  // window with ?node=<id> and passes the answer straight through
  // (metrics/api_routes.py), so the charts of a peer are that peer's own
  // measurements rather than anything this node inferred about it.

  localNodeId() {
    var s = this.app && this.app.state && this.app.state.status;
    return (s && s.node_id) || '';
  },

  fleetNodes() {
    var localId = this.localNodeId();
    var rows = (this.app && this.app.state && this.app.state.nodes) || [];
    var out = [];
    var seen = {};
    out.push({
      id: '',
      label: this.localNodeLabel(),
      local: true,
    });
    if (localId) seen[localId] = true;
    for (var i = 0; i < rows.length; i++) {
      var row = rows[i] || {};
      var id = row.node_id || '';
      if (!id || seen[id]) continue;
      seen[id] = true;
      out.push({ id: id, label: row.node_name || id, local: false });
    }
    return out;
  },

  localNodeLabel() {
    var s = this.app && this.app.state && this.app.state.status;
    var name = (s && (s.node_name || s.node_id)) || 'this node';
    return name + ' (this node)';
  },

  renderNodePills() {
    var mount = document.getElementById('metrics-nodes');
    if (!mount) return;
    var nodes = this.fleetNodes();
    // One node and nothing to choose between: the picker would be a label.
    if (nodes.length < 2) { mount.innerHTML = ''; return; }
    var selected = this.state.nodeId || '';
    var html = '<span class="metrics-nodes-label">NODE</span>';
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      var active = (selected === n.id) ? ' active' : '';
      html += '<button class="metrics-node-pill' + active + '" data-node-id="' + this.esc(n.id) + '">' +
              this.esc(n.label) + '</button>';
    }
    mount.innerHTML = html;
  },

  // ======================================================================
  //  FETCH
  // ======================================================================

  async fetchJSON(url) {
    try {
      var resp = await AINodeAuth.fetch(url);
      var body = null;
      try { body = await resp.json(); } catch (e) { body = null; }
      return { ok: resp.ok, status: resp.status, body: body };
    } catch (e) {
      return { ok: false, status: 0, body: null };
    }
  },

  async load(hard) {
    var D = this.data();
    if (!D) return;
    this.state.loading = true;
    if (hard) {
      this.state.payload = null;
      this.state.error = null;
      this.setStatus('loading…');
    }
    var qs = D.historyQuery(this.state.range, D.SERIES, this.state.nodeId);
    var history = await this.fetchJSON('/api/metrics/history?' + qs);
    // The live snapshot, for one thing the history cannot carry: whether this
    // node's "memory used" is NVML's reading or the engines' summed reservation.
    // Only for the node serving the page; a peer's snapshot is not ours to claim.
    if (!this.state.nodeId) {
      var snap = await this.fetchJSON('/api/metrics');
      this.state.snapshot = snap.ok ? snap.body : null;
    } else {
      this.state.snapshot = null;
    }
    this.state.loading = false;
    this.state.fetchedAt = Date.now();
    if (!history.body) {
      this.state.payload = null;
      this.state.error = history.status
        ? 'the node answered ' + history.status + ' for its history'
        : 'this node could not be reached for its history';
      this.paint();
      return;
    }
    if (!history.ok || history.body.error) {
      this.state.payload = history.body.series ? history.body : null;
      this.state.error = String(history.body.error || ('the node answered ' + history.status));
      this.paint();
      return;
    }
    this.state.payload = history.body;
    this.state.error = null;
    this.paint();
  },

  setStatus(text) {
    var el = document.getElementById('metrics-status');
    if (el) el.textContent = text || '';
  },

  // ======================================================================
  //  PAINT: legends, notes, the store block, then the canvases
  // ======================================================================

  paint() {
    var D = this.data();
    var payload = this.state.payload;
    var store = (payload && payload.store) || null;
    var retentionOff = store && store.enabled === false;

    var status = '';
    if (this.state.error) status = this.state.error;
    else if (retentionOff) status = 'retention is off on this node';
    else if (payload) {
      status = (payload.resolution === '1m' ? 'one minute roll-ups' : 'raw samples') +
               ' · step ' + Math.round(payload.step) + 's · ' + payload.points + ' points';
    }
    this.setStatus(status);
    this.renderStoreBlock(store, payload);
    this.draw();
  },

  renderStoreBlock(store, payload) {
    var el = document.getElementById('metrics-store');
    if (!el) return;
    if (!store) { el.innerHTML = ''; return; }
    if (store.enabled === false) {
      el.innerHTML = 'Retention is off on this node, so there is no history to draw. ' +
        'Turn it on with a <code>metrics</code> block in config.json ' +
        '(<code>"enabled": true</code>) and restart the node.';
      return;
    }
    var parts = [];
    if (store.degraded) {
      parts.push('<span class="metrics-store-warn">the store is degraded: the node kept serving and stopped recording</span>');
    }
    if (store.oldest_sample) {
      var reach = 'history reaches back to ' + this.esc(this.stamp(store.oldest_sample * 1000));
      // How much of the window on screen is before this node was recording. Said
      // here, once, because it is a fact about the node and the range rather than
      // about any one panel: the empty left hand side of every chart above is
      // this, and it is not a gap and not a fault.
      var D = this.data();
      var sampled = D.coverage(D.toPoints(payload, 'uptime_seconds'),
                               Number(store.oldest_sample) * 1000);
      if (sampled.slots && (sampled.beforeStart / sampled.slots) > 0.02) {
        reach += ' (' + Math.round((sampled.beforeStart / sampled.slots) * 100) +
                 '% of this window is before that, so the charts start where it does)';
      }
      parts.push(reach);
    }
    if (store.retention_hours) {
      parts.push(store.retention_hours + ' h raw, ' + store.retention_days + ' d rolled up');
    }
    if (store.interval_seconds) parts.push('sampled every ' + Math.round(store.interval_seconds) + 's');
    if (store.db_bytes) parts.push(this.bytes(store.db_bytes) + ' on disk');
    if (store.samples !== undefined && store.samples !== null) {
      parts.push(store.samples.toLocaleString() + ' raw rows, ' +
                 Number(store.downsampled || 0).toLocaleString() + ' rolled up');
    }
    el.innerHTML = parts.join(' · ');
  },

  stamp(ms) {
    try {
      var d = new Date(ms);
      return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch (e) { return ''; }
  },

  bytes(n) {
    var v = Number(n) || 0;
    if (v >= 1024 * 1024 * 1024) return (v / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
    if (v >= 1024 * 1024) return (v / (1024 * 1024)).toFixed(1) + ' MB';
    if (v >= 1024) return Math.round(v / 1024) + ' KB';
    return v + ' B';
  },

  // ======================================================================
  //  WHAT EACH PANEL DRAWS
  // ======================================================================
  //
  // One function per panel, from the payload to a list of lines. Counters become
  // rates here (see AINodeMetricsData.rate: a restart is a null, never a negative
  // or a zero), MB becomes GB, and nothing else is transformed.

  linesFor(key, payload, range) {
    var D = this.data();
    var palette = this.palette();
    var gap = D.gapLimitMs(payload, range);
    var series = function (name) { return D.toPoints(payload, name); };

    if (key === 'memory') {
      var used = D.scalePoints(series('gpu.memory_used_mb'), 1 / 1024);
      var total = D.scalePoints(series('gpu.memory_total_mb'), 1 / 1024);
      var host = D.scalePoints(series('gpu.system_memory_used_mb'), 1 / 1024);
      return [
        { name: 'used', color: palette.line1, points: used, digits: 1 },
        { name: 'total', color: palette.muted, points: total, digits: 0, dashed: true },
        { name: 'host RAM', color: palette.line2, points: host, digits: 1, faint: true },
      ];
    }
    if (key === 'util') {
      return [{ name: 'utilization', color: palette.line1, points: series('gpu.utilization_percent'), digits: 0 }];
    }
    if (key === 'temp') {
      return [{ name: 'temperature', color: palette.line3, points: series('gpu.temperature_c'), digits: 0 }];
    }
    if (key === 'requests') {
      return [
        { name: 'requests', color: palette.line1, points: D.rate(series('requests.total'), 60, gap), digits: 1 },
        { name: 'errors', color: palette.line4, points: D.rate(series('requests.errors'), 60, gap), digits: 1 },
      ];
    }
    if (key === 'latency') {
      return [
        { name: 'p50', color: palette.line1, points: series('requests.latency_ms.p50'), digits: 0 },
        { name: 'p95', color: palette.line2, points: series('requests.latency_ms.p95'), digits: 0 },
        { name: 'p99', color: palette.line3, points: series('requests.latency_ms.p99'), digits: 0 },
      ];
    }
    if (key === 'uptime') {
      return [{
        name: 'uptime', color: palette.line2,
        points: D.scalePoints(series('uptime_seconds'), 1 / 3600), digits: 1,
      }];
    }
    return [];
  },

  // What a panel says when it has nothing, or when what it has needs a word of
  // explanation. This is where "the driver does not expose it" is written down
  // instead of being drawn as a flat zero.
  noteFor(key, payload, lines) {
    var D = this.data();
    var measured = 0;
    for (var i = 0; i < lines.length; i++) measured += D.summary(lines[i].points).measured;
    var parts = this.panelNote(key, payload, lines, measured);
    // A panel with nothing on it has said all there is to say; the coverage
    // sentences below are about the shape of a line that exists.
    if (measured) parts = parts.concat(this.coverageNote(key, payload, lines));
    return parts.join(' ');
  },

  // The sentence that belongs to this panel and no other: why a series is absent,
  // what "used" means on this hardware, what a fall in the uptime line is.
  panelNote(key, payload, lines, measured) {
    var D = this.data();
    if (key === 'util' && !measured) {
      return ['This node reports no GPU utilization figure. On a unified memory part ' +
        '(GB10 / DGX Spark) the driver exposes no utilization counter at all, so there ' +
        'is nothing to draw: not zero, not idle, not measured. Memory and temperature ' +
        'above and below are real.'];
    }
    if (key === 'latency' && !measured) {
      return ['No request has been timed on this node in this window, so there are no ' +
        'percentiles. An untimed percentile is absent here rather than drawn at zero, ' +
        'which would read as instant answers.'];
    }
    if (key === 'requests' && !measured) {
      return ['No request has been counted in this window. A rate needs two samples of ' +
        'the counter, so the first point of the window is always absent.'];
    }
    if (key === 'temp' && !measured) {
      return ['This node reports no temperature. Nothing is drawn rather than a zero, which ' +
        'would read as a cold GPU.'];
    }
    if (key === 'uptime') {
      if (!measured) return ['This node recorded no uptime in this window.'];
      var last = D.summary(lines[0].points).last;
      // A fall in this series is the node coming back up, and it is the one
      // reading that explains a gap in every chart above.
      var restarts = 0;
      var previous = null;
      var points = lines[0].points;
      for (var k = 0; k < points.length; k++) {
        var value = points[k].v;
        if (value === null || value === undefined) { previous = null; continue; }
        if (previous !== null && value < previous) restarts += 1;
        previous = value;
      }
      if (restarts) {
        return ['This node restarted ' + restarts + (restarts === 1 ? ' time' : ' times') +
          ' in this window: the line falls back to zero at each one, and the gaps in the ' +
          'charts above line up with those falls.'];
      }
      return ['One run, ' + D.fmt(last, 1) + ' hours and counting. A fall in this line is a ' +
        'restart, which is what a gap in the charts above usually means.'];
    }
    if (key === 'memory') {
      if (!measured) return ['This node recorded no memory figures in this window.'];
      var gpu = (this.state.snapshot && this.state.snapshot.gpu) || null;
      var source = gpu && gpu.memory_used_source;
      if (source === 'engine_reservations') {
        return ['This node has unified memory, so "used" is what the loaded engines reserved, ' +
          'not a driver reading. Host RAM is the figure the operating system reports and is ' +
          'not VRAM: it counts page cache and every other process.'];
      }
      if (source === 'nvml') {
        return ['Used and total are the readings NVML gives, summed across every device on the ' +
          'node. Host RAM is the figure the operating system reports and is not VRAM.'];
      }
      return [];
    }
    return [];
  },

  // What is MISSING from a panel that has something on it. The two reasons a slot
  // can be empty are not the same fact:
  //
  //   * before the node's oldest sample it was not recording yet, which is not a
  //     gap and not a fault. A node up for ten minutes has fifty empty minutes in
  //     a one hour window, and "83% of this window was not measured" is a true
  //     sentence that tells the reader nothing.
  //   * after it, the sampler or the node was down, and that is worth naming.
  //
  // Both are counted off uptime_seconds, which the sampler writes on every tick it
  // runs, so it answers "was this node sampling" for every panel rather than "did
  // this particular figure exist yet".
  coverageNote(key, payload, lines) {
    var D = this.data();
    var store = (payload && payload.store) || {};
    var startedAt = (store.oldest_sample ? Number(store.oldest_sample) * 1000 : null);
    var sampled = D.coverage(D.toPoints(payload, 'uptime_seconds'), startedAt);
    var out = [];
    if (!sampled.slots) return out;

    // The "not recording yet" share is a fact about the NODE and the window, not
    // about one panel, so it is said once under the grid (renderStoreBlock) and
    // not six times down the page. What is left here is per panel.
    if ((sampled.gaps / sampled.slots) > 0.02) {
      out.push(Math.round((sampled.gaps / sampled.slots) * 100) + '% of the recorded window was ' +
        'not sampled (the node or its sampler was down). Those stretches are gaps in the lines, ' +
        'never zeros.');
    }
    // A figure that exists only once something happens: say that, rather than
    // letting the reader read its late start as a gap.
    var measured = 0;
    for (var i = 0; i < lines.length; i++) {
      measured = Math.max(measured, D.summary(lines[i].points).measured);
    }
    if (measured && sampled.measured && measured < sampled.measured * 0.9) {
      if (key === 'latency') {
        out.push('The percentiles start at the first request this node timed; before that there ' +
          'was nothing to take a percentile of.');
      } else if (key === 'requests') {
        out.push('A rate needs two samples of the counter, so it starts one tick after the ' +
          'history does.');
      }
    }
    return out;
  },

  // ======================================================================
  //  DRAWING
  // ======================================================================

  // The design tokens, read from the stylesheet rather than copied here, so the
  // charts follow a theme change instead of pinning last season's palette.
  palette() {
    var fallback = {
      line1: '#76B900', line2: '#00D4AA', line3: '#FFB800', line4: '#FF3333',
      text: '#888888', muted: '#555555', grid: '#1f1f1f', bg: '#111111',
    };
    var root = (typeof document !== 'undefined') ? document.documentElement : null;
    if (!root || typeof getComputedStyle !== 'function') return fallback;
    var css = getComputedStyle(root);
    var read = function (name, alt) {
      var value = '';
      try { value = (css.getPropertyValue(name) || '').trim(); } catch (e) { value = ''; }
      return value || alt;
    };
    return {
      line1: read('--nvidia-green', fallback.line1),
      line2: read('--cyan', fallback.line2),
      line3: read('--amber', fallback.line3),
      line4: read('--red', fallback.line4),
      text: read('--text-secondary', fallback.text),
      muted: read('--text-muted', fallback.muted),
      grid: read('--border', fallback.grid),
      bg: read('--bg-card', fallback.bg),
    };
  },

  draw() {
    if (!this.state.built) return;
    var D = this.data();
    if (!D) return;
    var payload = this.state.payload;
    var range = D.rangeFor(this.state.range);
    for (var i = 0; i < this.PANELS.length; i++) {
      var panel = this.PANELS[i];
      var lines = payload ? this.linesFor(panel.key, payload, range) : [];
      this.drawPanel(panel, lines, payload, range);
    }
  },

  drawPanel(panel, lines, payload, range) {
    var D = this.data();
    var canvas = document.getElementById('metrics-canvas-' + panel.key);
    var legend = document.getElementById('metrics-legend-' + panel.key);
    var empty = document.getElementById('metrics-empty-' + panel.key);
    var note = document.getElementById('metrics-note-' + panel.key);
    var palette = this.palette();

    var measured = 0;
    for (var i = 0; i < lines.length; i++) measured += D.summary(lines[i].points).measured;

    if (legend) legend.innerHTML = this.legendHtml(lines, panel);
    if (note) {
      var text = payload ? this.noteFor(panel.key, payload, lines) : '';
      note.textContent = text;
      note.style.display = text ? '' : 'none';
    }
    if (empty) {
      var reason = '';
      if (this.state.error) reason = this.state.error;
      else if (!payload) reason = 'no history yet';
      else if (!measured) reason = 'nothing measured in this window';
      empty.textContent = reason;
      empty.style.display = reason ? '' : 'none';
    }

    var setup = this.canvasSetup(canvas);
    if (!setup) return;
    var ctx = setup.ctx;
    var w = setup.width;
    var h = setup.height;
    ctx.clearRect(0, 0, w, h);

    var win = D.windowOf(payload, range);
    var pad = { left: 48, right: 10, top: 8, bottom: 20 };
    var plotW = Math.max(10, w - pad.left - pad.right);
    var plotH = Math.max(10, h - pad.top - pad.bottom);
    var axis = D.extent(lines.map(function (l) { return l.points; }), panel.axis);

    // The x axis is drawn even with nothing on it: an empty hour is a fact about
    // the node, and an axis with labels says which hour it was.
    this.drawTimeAxis(ctx, palette, win, pad, plotW, plotH, range);
    if (!axis || !measured) return;

    var yTicks = D.ticks(axis.min, axis.max, 4);
    ctx.save();
    ctx.font = '9px ' + this.monoFont();
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (var t = 0; t < yTicks.length; t++) {
      var value = yTicks[t];
      var y = pad.top + plotH - ((value - axis.min) / (axis.max - axis.min)) * plotH;
      ctx.strokeStyle = palette.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, Math.round(y) + 0.5);
      ctx.lineTo(pad.left + plotW, Math.round(y) + 0.5);
      ctx.stroke();
      ctx.fillStyle = palette.muted;
      ctx.fillText(D.fmt(value, this.tickDigits(yTicks)) + '', pad.left - 6, y);
    }
    ctx.restore();

    var X = function (ts) {
      return pad.left + ((ts - win.from) / Math.max(1, win.to - win.from)) * plotW;
    };
    var Y = function (v) {
      return pad.top + plotH - ((v - axis.min) / (axis.max - axis.min)) * plotH;
    };
    var gap = D.gapLimitMs(payload, range);

    for (var l = 0; l < lines.length; l++) {
      var line = lines[l];
      var runs = D.segments(line.points, gap);
      if (!runs.length) continue;
      ctx.save();
      ctx.strokeStyle = line.color;
      ctx.globalAlpha = line.faint ? 0.45 : 1;
      ctx.lineWidth = line.dashed ? 1 : 1.6;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      if (line.dashed && ctx.setLineDash) ctx.setLineDash([4, 4]);
      for (var r = 0; r < runs.length; r++) {
        var run = runs[r];
        if (run.length === 1) {
          // One measured sample between two holes is still a measurement.
          ctx.fillStyle = line.color;
          ctx.beginPath();
          ctx.arc(X(run[0].t), Y(run[0].v), 1.6, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }
        ctx.beginPath();
        for (var p = 0; p < run.length; p++) {
          var x = X(run[p].t);
          var y2 = Y(run[p].v);
          if (p === 0) ctx.moveTo(x, y2);
          else ctx.lineTo(x, y2);
        }
        ctx.stroke();
      }
      ctx.restore();
    }
  },

  tickDigits(values) {
    var span = 0;
    if (values.length > 1) span = Math.abs(values[1] - values[0]);
    if (span >= 10) return 0;
    if (span >= 1) return 1;
    return 2;
  },

  drawTimeAxis(ctx, palette, win, pad, plotW, plotH, range) {
    var D = this.data();
    var stamps = D.timeTicks(win.from, win.to, this.xTickCount(plotW));
    ctx.save();
    ctx.font = '9px ' + this.monoFont();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = palette.muted;
    ctx.strokeStyle = palette.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, Math.round(pad.top + plotH) + 0.5);
    ctx.lineTo(pad.left + plotW, Math.round(pad.top + plotH) + 0.5);
    ctx.stroke();
    for (var i = 0; i < stamps.length; i++) {
      var x = pad.left + ((stamps[i] - win.from) / Math.max(1, win.to - win.from)) * plotW;
      var label = this.axisLabel(stamps[i], range);
      if (i === 0) ctx.textAlign = 'left';
      else if (i === stamps.length - 1) ctx.textAlign = 'right';
      else ctx.textAlign = 'center';
      ctx.fillText(label, x, pad.top + plotH + 5);
    }
    ctx.restore();
  },

  xTickCount(plotW) {
    if (plotW < 240) return 3;
    if (plotW < 420) return 4;
    return 5;
  },

  axisLabel(ms, range) {
    var d = new Date(ms);
    try {
      if (range && range.key === '7d') {
        return d.toLocaleDateString([], { month: 'numeric', day: 'numeric' }) + ' ' +
               d.toLocaleTimeString([], { hour: '2-digit' });
      }
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return '';
    }
  },

  monoFont() {
    return 'JetBrains Mono, SF Mono, monospace';
  },

  legendHtml(lines, panel) {
    var D = this.data();
    var html = '';
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      var s = D.summary(line.points);
      var value = s.measured ? (D.fmt(s.last, line.digits) + (panel.unit || '')) : 'n/a';
      html += '<span class="metrics-legend-item' + (s.measured ? '' : ' absent') + '">' +
              '<i style="background:' + this.esc(line.color) + '"></i>' +
              this.esc(line.name) + ' <b>' + this.esc(value) + '</b></span>';
    }
    return html;
  },

  // Size the backing store to the device pixels, so a line is one pixel wide on
  // a retina display instead of a two pixel smear. Same shape as topology.js.
  canvasSetup(canvas) {
    if (!canvas || typeof canvas.getContext !== 'function') return null;
    var ctx = canvas.getContext('2d');
    if (!ctx) return null;
    var dpr = window.devicePixelRatio || 1;
    var width = canvas.clientWidth || canvas.parentNode && canvas.parentNode.clientWidth || 0;
    var height = canvas.clientHeight || 190;
    if (!width) return null;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    if (ctx.setTransform) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, width: width, height: height };
  },
};

window.AINodeMetrics = AINodeMetrics;
