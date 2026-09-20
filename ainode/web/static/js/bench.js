/* AINode Bench view - run the benchmark against whatever is loaded on the fleet.
 *
 * Self-contained on purpose. app.js owns one nav pill and one case arm; every
 * other line of the Bench view lives here, so this file and the chat view can be
 * worked on at the same time without touching each other.
 *
 * The model picker is fleet truth: it is built from /api/server/status's
 * loaded_models, the same source the Server view and the proxy's routing agree
 * on, so the list only ever offers instances that are actually serving. Nothing
 * in this view loads, unloads or restarts anything: a bench is inference only.
 */

const AINodeBench = {
  state: {
    built: false,
    instances: [],
    runs: [],
    activeId: null,
    watching: null,      // run id whose detail we are polling
    detail: null,
    poll: null,
    reportNonce: 0,
    submitting: false,
  },

  SECTIONS: [
    { key: 'single', title: 'Single stream', hint: 'TTFT and decode on a short prompt' },
    { key: 'prefill', title: 'Prefill scaling', hint: 'decode and TTFT against prompt length' },
    { key: 'sustained', title: 'Sustained generation', hint: 'does the rate hold over one long answer' },
    { key: 'concurrency', title: 'Concurrency', hint: 'aggregate throughput at N streams' },
    { key: 'reasoning', title: 'Reasoning tax', hint: 'same prompt, thinking on vs off' },
  ],

  DEPTH_PRESETS: [
    { label: 'Quick', value: '4000' },
    { label: 'Standard', value: '4000,16000,32000' },
    { label: 'Deep', value: '4000,16000,32000,64000,120000' },
  ],

  STREAM_PRESETS: [
    { label: 'Light', value: '1,4' },
    { label: 'Standard', value: '1,2,4,8' },
    { label: 'Full', value: '1,2,4,8,16' },
  ],

  // ======================================================================
  //  ENTRY POINT (called from app.js's refresh switch, once per poll tick)
  // ======================================================================

  render(app) {
    this.app = app || window.AINode || null;
    var mount = document.getElementById('bench-content');
    if (!mount) return;
    if (!this.state.built) {
      mount.innerHTML = this.formHtml() + this.runPanelHtml() + this.resultsHtml() +
                        this.reportHtml();
      this.bind();
      this.state.built = true;
      this.refreshInstances();
      this.refreshRuns();
      this.startPoll();
    } else {
      // A periodic tick must never rebuild the form out from under a half-filled
      // field, so it only refreshes what the server owns.
      this.refreshInstances();
      this.refreshRuns();
    }
  },

  esc(s) {
    if (this.app && typeof this.app.esc === 'function') return this.app.esc(s);
    var d = document.createElement('div');
    d.textContent = s === null || s === undefined ? '' : String(s);
    return d.innerHTML;
  },

  toast(msg, type) {
    if (this.app && typeof this.app.toast === 'function') this.app.toast(msg, type);
  },

  async json(url, options) {
    var resp = await AINodeAuth.fetch(url, options);
    var data = null;
    try { data = await resp.json(); } catch (e) { data = null; }
    return { ok: resp.ok, status: resp.status, data: data };
  },

  // ======================================================================
  //  MARKUP
  // ======================================================================

  formHtml() {
    var checks = this.SECTIONS.map(function (s) {
      var on = s.key !== 'sustained' ? ' checked' : '';
      return '<label class="bench-check" title="' + s.hint + '">' +
             '<input type="checkbox" class="bench-section" value="' + s.key + '"' + on + '>' +
             '<span>' + s.title + '</span></label>';
    }).join('');
    return '' +
      '<div class="view-header bench-header">' +
        '<h2>Bench</h2>' +
        '<p>Measure what a spec sheet does not, against whatever is already loaded on ' +
        'the fleet. Inference only: a run never loads, unloads or restarts anything.</p>' +
      '</div>' +
      '<div class="card bench-form-card">' +
        '<div class="card-header"><span class="card-title">New run</span>' +
          '<span class="bench-target" id="bench-target"></span></div>' +
        '<div class="bench-form-grid">' +
          '<div class="bench-field bench-field-wide">' +
            '<label class="form-label" for="bench-model">Loaded instance</label>' +
            '<select id="bench-model" class="form-select">' +
              '<option value="">loading fleet...</option></select>' +
          '</div>' +
          '<div class="bench-field">' +
            '<label class="form-label" for="bench-label">Label</label>' +
            '<input id="bench-label" class="form-input" maxlength="60" ' +
              'placeholder="what makes this run distinct">' +
          '</div>' +
          '<div class="bench-field">' +
            '<label class="form-label" for="bench-max-tokens">Max tokens</label>' +
            '<input id="bench-max-tokens" class="form-input" type="number" min="1" ' +
              'max="8192" value="200">' +
          '</div>' +
          '<div class="bench-field bench-field-full">' +
            '<label class="form-label">Sections</label>' +
            '<div class="bench-checks">' + checks + '</div>' +
          '</div>' +
          '<div class="bench-field">' +
            '<label class="form-label">Prompt depths (tokens)</label>' +
            '<div class="pill-group bench-presets" id="bench-depth-presets">' +
              this.DEPTH_PRESETS.map(function (p, i) {
                return '<button class="pill' + (i === 0 ? ' active' : '') +
                       '" data-value="' + p.value + '">' + p.label + '</button>';
              }).join('') +
            '</div>' +
            '<input id="bench-depths" class="form-input" value="4000">' +
          '</div>' +
          '<div class="bench-field">' +
            '<label class="form-label">Concurrency streams</label>' +
            '<div class="pill-group bench-presets" id="bench-stream-presets">' +
              this.STREAM_PRESETS.map(function (p, i) {
                return '<button class="pill' + (i === 0 ? ' active' : '') +
                       '" data-value="' + p.value + '">' + p.label + '</button>';
              }).join('') +
            '</div>' +
            '<input id="bench-streams" class="form-input" value="1,4">' +
          '</div>' +
          '<div class="bench-field">' +
            '<label class="form-label">Thinking</label>' +
            '<div class="pill-group" id="bench-think">' +
              '<button class="pill active" data-value="off">Off</button>' +
              '<button class="pill" data-value="default">Model default</button>' +
            '</div>' +
            '<div class="bench-hint">The reasoning section always measures both ' +
              'states; this sets the other four.</div>' +
          '</div>' +
        '</div>' +
        '<div class="bench-form-actions">' +
          '<button class="btn-nvidia" id="bench-run">RUN BENCH</button>' +
          '<span class="bench-hint" id="bench-form-hint"></span>' +
        '</div>' +
      '</div>';
  },

  runPanelHtml() {
    return '<div class="card bench-run-card" id="bench-run-card" style="display:none">' +
             '<div class="card-header"><span class="card-title">Live run</span>' +
               '<span id="bench-run-badge"></span></div>' +
             '<div id="bench-run-body"></div>' +
           '</div>';
  },

  resultsHtml() {
    return '<div class="card bench-results-card">' +
             '<div class="card-header"><span class="card-title">Results</span>' +
               '<span class="log-line-count" id="bench-results-dir"></span></div>' +
             '<div class="bench-table-wrap"><table class="table bench-table">' +
               '<thead><tr><th>Model</th><th>Placement</th><th>Single</th>' +
               '<th>Batched</th><th>When</th><th></th></tr></thead>' +
               '<tbody id="bench-results-body"></tbody></table></div>' +
           '</div>';
  },

  reportHtml() {
    return '<div class="card bench-report-card">' +
             '<div class="card-header"><span class="card-title">Report</span>' +
               '<span class="bench-report-actions">' +
                 '<button class="btn-ghost" id="bench-report-reload">Reload</button>' +
                 '<button class="btn-ghost" id="bench-report-open">Open</button>' +
               '</span></div>' +
             // Filled by reloadReport() through the auth wrapper, not by src.
             '<iframe id="bench-report-frame" class="bench-report-frame" ' +
               'title="AINode bench report"></iframe>' +
           '</div>';
  },

  // ======================================================================
  //  BINDING
  // ======================================================================

  bind() {
    var self = this;
    document.getElementById('bench-run').addEventListener('click', function () {
      self.submit();
    });
    document.getElementById('bench-model').addEventListener('change', function () {
      self.paintTarget();
    });
    document.getElementById('bench-report-reload').addEventListener('click', function () {
      self.reloadReport();
    });
    document.getElementById('bench-report-open').addEventListener('click', function () {
      self.openReport();
    });
    // The frame has no src: nothing but the wrapper may fetch the report.
    this.reloadReport();
    this.bindPills('bench-depth-presets', 'bench-depths');
    this.bindPills('bench-stream-presets', 'bench-streams');
    this.bindPills('bench-think', null);
    // Typing a custom sweep clears the preset highlight: the text field is the
    // value that gets submitted, so the pills must not claim otherwise.
    ['bench-depths', 'bench-streams'].forEach(function (id) {
      document.getElementById(id).addEventListener('input', function () {
        var group = id === 'bench-depths' ? 'bench-depth-presets' : 'bench-stream-presets';
        var val = this.value.replace(/\s/g, '');
        document.querySelectorAll('#' + group + ' .pill').forEach(function (p) {
          p.classList.toggle('active', p.dataset.value === val);
        });
      });
    });
  },

  bindPills(groupId, inputId) {
    var group = document.getElementById(groupId);
    if (!group) return;
    group.addEventListener('click', function (e) {
      var pill = e.target.closest('.pill');
      if (!pill) return;
      group.querySelectorAll('.pill').forEach(function (p) {
        p.classList.toggle('active', p === pill);
      });
      if (inputId) document.getElementById(inputId).value = pill.dataset.value;
    });
  },

  pillValue(groupId) {
    var active = document.querySelector('#' + groupId + ' .pill.active');
    return active ? active.dataset.value : null;
  },

  // ======================================================================
  //  FLEET-TRUE MODEL PICKER
  // ======================================================================

  async refreshInstances() {
    var res = await this.json('/api/server/status');
    var loaded = (res.data && res.data.loaded_models) || [];
    // Only chat-capable LLM instances can be benched; an embedding model has no
    // /v1/chat/completions to time.
    var usable = loaded.filter(function (m) {
      return m.id && (m.type || 'llm') === 'llm';
    });
    var key = usable.map(function (m) {
      return m.id + '@' + m.node_hostname + ':' + m.port + ':' + (m.ready ? 1 : 0);
    }).join('|');
    if (key === this._instanceKey) return;      // nothing moved on the fleet
    this._instanceKey = key;
    this.state.instances = usable;
    var select = document.getElementById('bench-model');
    if (!select) return;
    var previous = select.value;
    if (!usable.length) {
      select.innerHTML = '<option value="">no model is loaded on the fleet</option>';
      this.paintTarget();
      return;
    }
    select.innerHTML = usable.map(function (m) {
      var label = m.id + '  ·  ' + (m.node_hostname || '?') + ':' + m.port +
                  (m.ready ? '' : '  (not ready)');
      return '<option value="' + m.id + '"' + (m.ready ? '' : ' disabled') + '>' +
             label + '</option>';
    }).join('');
    if (previous && usable.some(function (m) { return m.id === previous; })) {
      select.value = previous;
    }
    this.paintTarget();
  },

  selectedInstance() {
    var id = (document.getElementById('bench-model') || {}).value;
    return this.state.instances.filter(function (m) { return m.id === id; })[0] || null;
  },

  paintTarget() {
    var el = document.getElementById('bench-target');
    if (!el) return;
    var inst = this.selectedInstance();
    if (!inst) {
      el.innerHTML = '<span class="badge badge-muted">no instance</span>';
      return;
    }
    el.innerHTML = '<span class="badge ' + (inst.ready ? 'badge-green' : 'badge-amber') +
                   '">' + (inst.ready ? 'READY' : 'NOT READY') + '</span>' +
                   '<span class="bench-target-node">' + this.esc(inst.node_hostname) +
                   ':' + inst.port + '</span>';
  },

  // ======================================================================
  //  SUBMIT
  // ======================================================================

  async submit() {
    if (this.state.submitting) return;
    var inst = this.selectedInstance();
    var hint = document.getElementById('bench-form-hint');
    if (!inst) {
      hint.textContent = 'Pick a loaded instance first.';
      return;
    }
    var sections = Array.prototype.map.call(
      document.querySelectorAll('.bench-section:checked'),
      function (el) { return el.value; });
    if (!sections.length) {
      hint.textContent = 'Tick at least one section.';
      return;
    }
    var body = {
      model: inst.id,
      sections: sections,
      depths: document.getElementById('bench-depths').value,
      streams: document.getElementById('bench-streams').value,
      no_think: this.pillValue('bench-think') === 'off',
      max_tokens: parseInt(document.getElementById('bench-max-tokens').value, 10) || 200,
      label: document.getElementById('bench-label').value.trim(),
    };
    this.state.submitting = true;
    hint.textContent = 'Starting…';
    var res = await this.json('/api/bench/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    this.state.submitting = false;
    if (!res.ok) {
      var msg = (res.data && res.data.error) || ('HTTP ' + res.status);
      hint.textContent = msg;
      this.toast(msg, 'error');
      if (res.data && res.data.running) {
        this.state.watching = res.data.running;
        this.pollDetail();
      }
      return;
    }
    hint.textContent = '';
    (res.data.warnings || []).forEach((w) => this.toast(w, 'warning'));
    this.state.watching = res.data.run_id;
    this.toast('Bench started on ' + (res.data.target || {}).node, 'success');
    this.pollDetail();
  },

  // ======================================================================
  //  LIVE RUN PANEL
  // ======================================================================

  isVisible() {
    var view = document.getElementById('view-bench');
    return !!view && view.style.display !== 'none';
  },

  startPoll() {
    var self = this;
    if (this.state.poll) return;
    this.state.poll = setInterval(function () {
      // The view's own timer, so it idles when the user navigates away rather
      // than needing a stop hook in app.js's navigate().
      if (!self.isVisible()) return;
      if (self.state.watching) self.pollDetail();
    }, 1200);
  },

  async pollDetail() {
    var id = this.state.watching;
    if (!id) return;
    var res = await this.json('/api/bench/runs/' + encodeURIComponent(id) + '?log=120');
    if (!res.ok) {
      this.state.watching = null;
      return;
    }
    this.state.detail = res.data;
    this.paintRun(res.data);
    if (['completed', 'failed', 'cancelled'].indexOf(res.data.status) >= 0) {
      this.state.watching = null;
      this.refreshRuns();
      if (res.data.status === 'completed') {
        this.reloadReport();
        this.toast('Bench finished: ' + (res.data.result_file || id), 'success');
      }
    }
  },

  paintRun(run) {
    var card = document.getElementById('bench-run-card');
    var badge = document.getElementById('bench-run-badge');
    var body = document.getElementById('bench-run-body');
    if (!card) return;
    card.style.display = '';
    var cls = { running: 'badge-green', pending: 'badge-amber', completed: 'badge-green',
                failed: 'badge-red', cancelled: 'badge-muted' }[run.status] || 'badge-muted';
    badge.innerHTML = '<span class="badge ' + cls + '">' + run.status.toUpperCase() +
                      '</span>';
    var p = run.progress || {};
    var live = run.status === 'running' || run.status === 'pending';
    var step = p.step_total ? (p.step + '/' + p.step_total +
               (p.step_label ? ' · ' + this.esc(p.step_label) : '')) : '';
    var logLines = (run.log || []).map((l) =>
      '<div class="log-line">' + this.esc(l) + '</div>').join('');
    body.innerHTML =
      '<div class="bench-run-head">' +
        '<div class="bench-run-model">' + this.esc(run.model) + '</div>' +
        '<div class="bench-run-meta">' + this.esc(run.node || '') +
          (run.endpoint ? ' · ' + this.esc(run.endpoint) : '') +
          (run.elapsed_seconds !== null && run.elapsed_seconds !== undefined
            ? ' · ' + run.elapsed_seconds + 's' : '') +
        '</div>' +
      '</div>' +
      '<div class="progress-header">' +
        '<span class="progress-epoch">' +
          (p.section_label ? this.esc(p.section_label) : 'starting') +
          (p.section_total ? ' (' + (p.section_index + 1) + '/' + p.section_total + ')' : '') +
          (step ? ' - ' + step : '') +
        '</span>' +
        '<span class="progress-pct">' + (p.percent || 0) + '%</span>' +
      '</div>' +
      '<div class="progress-bar"><div class="progress-fill green" style="width:' +
        (p.percent || 0) + '%"></div></div>' +
      (run.error ? '<div class="bench-error">' + this.esc(run.error) + '</div>' : '') +
      ((run.warnings || []).length
        ? '<div class="bench-warnings">' + run.warnings.map((w) =>
            '<div class="bench-warn-line">' + this.esc(w) + '</div>').join('') + '</div>'
        : '') +
      '<div class="training-log-viewer bench-log" id="bench-log">' +
        (logLines || '<div class="log-empty">no output yet</div>') + '</div>' +
      '<div class="bench-run-actions">' +
        (live ? '<button class="btn-danger" id="bench-cancel">CANCEL</button>' : '') +
        (run.result_file
          ? '<button class="btn-ghost" data-bench-download="' +
            this.esc(run.run_id) + '">Download JSON</button>' : '') +
        (live ? '' : '<button class="btn-ghost" id="bench-dismiss">Dismiss</button>') +
      '</div>';
    var log = document.getElementById('bench-log');
    if (log) log.scrollTop = log.scrollHeight;
    var cancel = document.getElementById('bench-cancel');
    if (cancel) {
      cancel.addEventListener('click', async () => {
        cancel.disabled = true;
        var res = await this.json('/api/bench/runs/' +
          encodeURIComponent(run.run_id) + '/cancel', { method: 'POST' });
        if (!res.ok) this.toast((res.data && res.data.error) || 'cancel failed', 'error');
      });
    }
    var dismiss = document.getElementById('bench-dismiss');
    if (dismiss) {
      dismiss.addEventListener('click', function () {
        card.style.display = 'none';
      });
    }
    this.bindDownloads(body);
  },

  // ======================================================================
  //  RESULTS TABLE
  // ======================================================================

  async refreshRuns() {
    var res = await this.json('/api/bench/runs');
    if (!res.ok || !res.data) return;
    this.state.runs = res.data.runs || [];
    this.state.activeId = res.data.running || null;
    var dir = document.getElementById('bench-results-dir');
    if (dir) dir.textContent = res.data.results_dir || '';
    // A run that is already going when the view opens (a page reload mid-run, or
    // a run started from another browser) should still be watched.
    if (this.state.activeId && !this.state.watching) {
      this.state.watching = this.state.activeId;
      this.pollDetail();
    }
    this.paintRuns();
  },

  paintRuns() {
    var body = document.getElementById('bench-results-body');
    if (!body) return;
    var rows = this.state.runs;
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="6" class="bench-empty">No runs yet. ' +
                       'Pick a loaded instance above and run one.</td></tr>';
      return;
    }
    body.innerHTML = rows.map((r) => this.rowHtml(r)).join('');
    var self = this;
    body.querySelectorAll('[data-bench-delete]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.remove(btn.dataset.benchDelete);
      });
    });
    body.querySelectorAll('[data-bench-watch]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.state.watching = btn.dataset.benchWatch;
        self.pollDetail();
      });
    });
    this.bindDownloads(body);
  },

  rowHtml(r) {
    var s = r.summary || {};
    var running = r.status === 'running' || r.status === 'pending';
    var placement = [s.node || r.node || '-'];
    if (s.tp && s.tp > 1) placement.push('TP=' + s.tp);
    if ((s.stacked_with || []).length) placement.push('+' + s.stacked_with.length + ' stacked');
    var single = s.single_tok_s
      ? s.single_tok_s.toFixed(1) + '<span class="bench-unit">tok/s</span>'
      : '<span class="bench-na">not measured</span>';
    var batched = s.conc_aggregate_tok_s
      ? s.conc_aggregate_tok_s.toFixed(1) + '<span class="bench-unit">tok/s @' +
        s.conc_streams + '</span>'
      : '<span class="bench-na">not measured</span>';
    var statusBadge = running
      ? '<span class="badge badge-amber">' + r.status.toUpperCase() + '</span>'
      : (r.status === 'failed'
          ? '<span class="badge badge-red">FAILED</span>'
          : (r.status === 'cancelled'
              ? '<span class="badge badge-muted">CANCELLED</span>' : ''));
    var label = r.label || s.label || '';
    return '<tr>' +
      '<td><div class="bench-cell-model">' + this.esc(s.model || r.model || '-') +
        '</div>' + (label ? '<div class="bench-cell-label">' + this.esc(label) +
        '</div>' : '') +
        (statusBadge ? '<div>' + statusBadge + '</div>' : '') + '</td>' +
      '<td class="bench-cell-dim">' + this.esc(placement.join(' · ')) + '</td>' +
      '<td class="bench-cell-num">' + single + '</td>' +
      '<td class="bench-cell-num">' + batched + '</td>' +
      '<td class="bench-cell-dim bench-cell-when">' + this.esc(this.when(r, s)) + '</td>' +
      '<td class="bench-cell-actions">' +
        (running
          ? '<button class="btn-ghost" data-bench-watch="' + this.esc(r.run_id) +
            '">Watch</button>'
          : (r.result_file
              ? '<button class="btn-ghost" data-bench-download="' +
                this.esc(r.run_id) + '">JSON</button>' : '')) +
        (running ? '' : '<button class="btn-ghost bench-del" title="Delete this run ' +
          'and its result file" data-bench-delete="' + this.esc(r.run_id) +
          '">&times;</button>') +
      '</td></tr>';
  },

  when(r, s) {
    // The stamp is the measurement's own UTC clock, so prefer it over a file
    // mtime that a copy would have rewritten.
    var stamp = s.stamp || '';
    var m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/.exec(stamp);
    if (m) {
      var d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]));
      return this.shortTime(d);
    }
    if (r.created_at) return this.shortTime(new Date(r.created_at * 1000));
    return '';
  },

  shortTime(d) {
    // Compact on purpose: the results table has six columns to fit before it
    // starts scrolling sideways, and the full locale string costs two of them.
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ', ' +
           d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit',
                                             hour12: false });
  },

  async remove(runId) {
    var res = await this.json('/api/bench/runs/' + encodeURIComponent(runId),
                              { method: 'DELETE' });
    if (!res.ok) {
      this.toast((res.data && res.data.error) || 'delete failed', 'error');
      return;
    }
    this.refreshRuns();
    this.reloadReport();
  },

  // An iframe src and an <a href> cannot carry an Authorization header, so with
  // auth on the report frame and the JSON links 401'd into nothing (#167). Both
  // go through the wrapper: the report is dropped in as srcdoc (it is one
  // self-contained page with no script and no CDN reference), a download is
  // handed to the browser as a blob.
  async reloadReport() {
    var frame = document.getElementById('bench-report-frame');
    if (!frame) return;
    // The report is rendered per request, so a cache-busting query is what makes
    // a new result show up.
    this.state.reportNonce += 1;
    var resp;
    try {
      resp = await AINodeAuth.fetch('/api/bench/report?v=' + this.state.reportNonce);
    } catch (e) {
      frame.srcdoc = this._frameNote('The node did not answer.');
      return;
    }
    if (!resp.ok) {
      frame.srcdoc = this._frameNote(resp.status === 401
        ? 'This node requires an API key. Config &gt; API access.'
        : 'Report unavailable (HTTP ' + resp.status + ').');
      return;
    }
    frame.srcdoc = await resp.text();
  },

  _frameNote(text) {
    return '<body style="margin:0;font:14px/1.5 system-ui,sans-serif;' +
           'background:#0d0d0d;color:#888;padding:16px">' + text + '</body>';
  },

  async openReport() {
    var resp = await AINodeAuth.fetch('/api/bench/report');
    if (!resp.ok) { this.toast('Report unavailable (HTTP ' + resp.status + ').', 'error'); return; }
    var url = URL.createObjectURL(new Blob([await resp.text()], { type: 'text/html' }));
    window.open(url, '_blank');
    // Revoked on a timer: revoking immediately races the new tab's load.
    setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
  },

  async downloadResult(runId) {
    var path = '/api/bench/results/' + encodeURIComponent(runId) + '.json';
    var resp = await AINodeAuth.fetch(path);
    if (!resp.ok) { this.toast('Result unavailable (HTTP ' + resp.status + ').', 'error'); return; }
    var url = URL.createObjectURL(new Blob([await resp.text()], { type: 'application/json' }));
    var a = document.createElement('a');
    a.href = url;
    a.download = runId + '.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
  },

  /** One click handler for both result tables and the run card. */
  bindDownloads(root) {
    var self = this;
    (root || document).querySelectorAll('[data-bench-download]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.downloadResult(btn.dataset.benchDownload);
      });
    });
  },
};

window.AINodeBench = AINodeBench;

/* Self-mount. app.js's periodic refresh skips a tick while the user is mid-click
 * (it must not rebuild the DOM under a drag), so the nav click that reveals this
 * view can land in that window and leave an empty pane for a few seconds. This
 * watcher builds the view the moment it becomes visible, and refreshes it on each
 * return, without another hook in app.js. */
document.addEventListener('DOMContentLoaded', function () {
  var wasVisible = false;
  setInterval(function () {
    if (!AINodeBench.isVisible()) {
      wasVisible = false;
      return;
    }
    if (!AINodeBench.state.built || !wasVisible) {
      AINodeBench.render(window.AINode);
    }
    wasVisible = true;
  }, 400);
});
