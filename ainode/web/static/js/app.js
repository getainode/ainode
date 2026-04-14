/* AINode Command Center — Single Page Application */

const AINode = {
  state: {
    status: null,
    nodes: [],
    metrics: null,
    messages: [],
    conversations: [],
    currentConversation: null,
    streaming: false,
    streamMetrics: { ttft: 0, tps: 0, tokens: 0 },
    topology: null,
    trainingJobs: [],
    trainingView: 'list',
    trainingDetailId: null,
    trainingLossData: [],
    pollInterval: null,
    metricsInterval: null,
    abortController: null,
    shardingStatus: null,
    currentView: 'dashboard',
    modelsFilter: 'all',
    modelsSearch: '',
    modelsSort: 'recommended',
  },

  // ========================================================================
  //  INITIALIZATION
  // ========================================================================

  init() {
    this.loadConversations();
    this.bindNav();
    this.bindChat();
    this.bindLaunchForm();
    this.initTopology();
    this.startPolling();
    this.renderConversationList();
  },

  initTopology() {
    var canvas = document.getElementById('topology-canvas');
    if (canvas && typeof Topology !== 'undefined') {
      this.state.topology = new Topology(canvas);
    }
  },

  // ========================================================================
  //  TOAST NOTIFICATION SYSTEM
  // ========================================================================

  toast(message, type) {
    type = type || 'info';
    var container = document.getElementById('toast-container');
    if (!container) return;
    var toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.innerHTML = '<span class="toast-message">' + this.esc(message) + '</span><button class="toast-close">&times;</button>';
    container.appendChild(toast);
    requestAnimationFrame(function () { toast.classList.add('toast-visible'); });
    var dismiss = function () {
      toast.classList.remove('toast-visible');
      toast.classList.add('toast-fade-out');
      setTimeout(function () { toast.remove(); }, 300);
    };
    toast.querySelector('.toast-close').addEventListener('click', dismiss);
    setTimeout(dismiss, 4000);
  },

  // ========================================================================
  //  CONVERSATION HISTORY (localStorage-backed)
  // ========================================================================

  loadConversations() {
    try {
      this.state.conversations = JSON.parse(localStorage.getItem('ainode_conversations') || '[]');
    } catch (e) {
      this.state.conversations = [];
    }
    // Restore last active conversation
    if (this.state.conversations.length > 0 && !this.state.currentConversation) {
      // Don't auto-load; let user pick
    }
  },

  saveConversations() {
    localStorage.setItem('ainode_conversations', JSON.stringify(this.state.conversations));
  },

  getCurrentConversation() {
    return this.state.conversations.find(function (c) { return c.id === AINode.state.currentConversation; }) || null;
  },

  newConversation() {
    var id = 'conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    var sel = document.getElementById('chat-model');
    var conv = { id: id, title: 'New Chat', messages: [], created_at: Date.now(), model: sel ? sel.value : '' };
    this.state.conversations.unshift(conv);
    this.state.currentConversation = id;
    this.state.messages = [];
    this.saveConversations();
    this.renderConversationList();
    this.renderChatMessages();
  },

  loadConversation(id) {
    var conv = this.state.conversations.find(function (c) { return c.id === id; });
    if (!conv) return;
    this.state.currentConversation = id;
    this.state.messages = conv.messages.slice();
    var select = document.getElementById('chat-model');
    if (select && conv.model) {
      for (var i = 0; i < select.options.length; i++) {
        if (select.options[i].value === conv.model) { select.value = conv.model; break; }
      }
    }
    this.renderConversationList();
    this.renderChatMessages();
  },

  deleteConversation(id) {
    this.state.conversations = this.state.conversations.filter(function (c) { return c.id !== id; });
    if (this.state.currentConversation === id) {
      this.state.currentConversation = null;
      this.state.messages = [];
      this.renderChatMessages();
    }
    this.saveConversations();
    this.renderConversationList();
    this.toast('Conversation deleted', 'info');
  },

  saveCurrentConversation() {
    var conv = this.getCurrentConversation();
    if (!conv) return;
    conv.messages = this.state.messages.slice();
    var sel = document.getElementById('chat-model');
    if (sel) conv.model = sel.value || conv.model;
    var firstUser = conv.messages.find(function (m) { return m.role === 'user'; });
    if (firstUser) conv.title = firstUser.content.slice(0, 30) + (firstUser.content.length > 30 ? '...' : '');
    this.saveConversations();
    this.renderConversationList();
  },

  renderConversationList() {
    var list = document.getElementById('conversation-list');
    if (!list) return;
    var self = this;
    var searchInput = document.getElementById('chat-search');
    var query = searchInput ? searchInput.value.toLowerCase().trim() : '';
    var filtered = this.state.conversations.filter(function (c) {
      if (!query) return true;
      return c.title.toLowerCase().indexOf(query) !== -1;
    });

    if (filtered.length === 0) {
      list.innerHTML = '<div class="conv-empty">' + (query ? 'No matches' : 'No conversations yet') + '</div>';
      return;
    }

    list.innerHTML = filtered.map(function (conv) {
      var active = conv.id === self.state.currentConversation ? ' active' : '';
      var dateStr = new Date(conv.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
      return '<div class="conv-item' + active + '" data-conv-id="' + self.esc(conv.id) + '">' +
        '<div class="conv-item-content">' +
        '<div class="conv-item-title">' + self.esc(conv.title) + '</div>' +
        '<div class="conv-item-date">' + dateStr + '</div>' +
        '</div>' +
        '<button class="conv-delete" data-delete-id="' + self.esc(conv.id) + '" title="Delete">&times;</button>' +
        '</div>';
    }).join('');

    list.querySelectorAll('.conv-item').forEach(function (el) {
      el.addEventListener('click', function (e) {
        if (e.target.closest('.conv-delete')) return;
        self.loadConversation(el.dataset.convId);
      });
    });
    list.querySelectorAll('.conv-delete').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        self.deleteConversation(btn.dataset.deleteId);
      });
    });
  },

  // ========================================================================
  //  NAVIGATION
  // ========================================================================

  bindNav() {
    var self = this;
    // Top nav pills: Dashboard, Downloads
    document.querySelectorAll('.nav-pill').forEach(function (el) {
      el.addEventListener('click', function () {
        var view = el.dataset.view;
        if (view) self.navigate(view);
      });
    });
  },

  navigate(view) {
    this.state.currentView = view;
    // Update nav pill active state
    document.querySelectorAll('.nav-pill').forEach(function (el) {
      el.classList.toggle('active', el.dataset.view === view);
    });
    // Show/hide views in center-stage
    document.querySelectorAll('#center-stage > .view').forEach(function (el) {
      el.style.display = 'none';
    });
    var target = document.getElementById('view-' + view);
    if (target) target.style.display = '';
    this.refresh();
  },

  // ========================================================================
  //  DATA FETCHING
  // ========================================================================

  async fetchJSON(url) {
    try {
      var resp = await fetch(url);
      if (!resp.ok) return null;
      return await resp.json();
    } catch (e) {
      return null;
    }
  },

  async refresh() {
    var results = await Promise.all([
      this.fetchJSON('/api/status'),
      this.fetchJSON('/api/nodes'),
      this.fetchJSON('/api/sharding/status'),
    ]);
    this.state.status = results[0];
    this.state.nodes = results[1]?.nodes || [];
    this.state.shardingStatus = results[2];

    this.updateTopBar();
    this.updateChatModelSelect();

    switch (this.state.currentView) {
      case 'dashboard':
        this.renderDashboard();
        break;
      case 'downloads':
        this.renderDownloads();
        break;
      case 'models':
        this.renderModels();
        break;
      case 'training':
        this.renderTraining();
        break;
    }

    // Always update right panel
    this.renderInstances();
    this.populateLaunchModels();
  },

  startPolling() {
    var self = this;
    // Status + nodes every 5s
    this.state.pollInterval = setInterval(function () { self.refresh(); }, 5000);
    // Metrics every 3s
    this.state.metricsInterval = setInterval(function () { self.pollMetrics(); }, 3000);
    // Initial fetch
    this.refresh();
  },

  async pollMetrics() {
    var data = await this.fetchJSON('/api/metrics');
    if (data) this.state.metrics = data;
  },

  // ========================================================================
  //  TOP BAR STATUS
  // ========================================================================

  updateTopBar() {
    var nodes = this.state.nodes;
    var s = this.state.status;
    var onlineCount = 0;
    if (s && s.engine_ready) onlineCount = 1;
    onlineCount = Math.max(onlineCount, nodes.filter(function (n) {
      return n.status === 'online' || n.status === 'serving' || n.engine_ready;
    }).length);

    var pill = document.querySelector('.top-bar-status');
    var countEl = document.getElementById('top-node-count');
    var labelEl = document.querySelector('.top-bar-status .node-label');

    if (!pill) return;

    if (onlineCount > 0) {
      pill.classList.remove('offline');
      pill.classList.add('online');
      if (countEl) countEl.textContent = onlineCount;
      if (labelEl) labelEl.textContent = onlineCount === 1 ? 'node online' : 'nodes online';
    } else {
      pill.classList.remove('online');
      pill.classList.add('offline');
      if (countEl) countEl.textContent = '0';
      if (labelEl) labelEl.textContent = 'offline';
    }
  },

  // ========================================================================
  //  DASHBOARD (center = topology canvas)
  // ========================================================================

  renderDashboard() {
    var nodes = this.state.nodes;
    var s = this.state.status;
    if (!s) return;

    // Update topology
    if (this.state.topology) {
      var topoNodes = nodes.length > 0 ? nodes : [{
        node_id: s.node_id || 'local',
        node_name: s.node_name || 'This Node',
        gpu_name: s.gpu?.name || 'GPU',
        gpu_memory_gb: s.gpu?.memory_gb || 0,
        model: s.model || 'none',
        status: s.engine_ready ? 'online' : 'starting',
      }];

      // Annotate with sharding info
      var shardInfo = this.state.shardingStatus && this.state.shardingStatus.active_sharding;
      if (shardInfo && shardInfo.shard_map) {
        topoNodes = topoNodes.map(function (n) {
          var shard = shardInfo.shard_map[n.node_id];
          if (shard) {
            n.shard_role = shard.role;
            n.shard_layers = shard.layers;
            n.shard_memory_gb = shard.estimated_memory_gb;
            n.sharded_model = shardInfo.model;
          }
          return n;
        });
      }
      this.state.topology.update(topoNodes);
    }
  },

  // ========================================================================
  //  RIGHT PANEL — INSTANCES
  // ========================================================================

  renderInstances() {
    var container = document.getElementById('instances-list');
    if (!container) return;
    var self = this;
    var s = this.state.status;
    var instances = [];

    // Collect from models_loaded
    if (s && s.models_loaded) {
      s.models_loaded.forEach(function (modelName) {
        instances.push({
          model: modelName,
          strategy: 'single',
          nodes: [s.node_id || 'local'],
          status: 'READY',
        });
      });
    }

    // Collect from sharding status
    var sharding = this.state.shardingStatus;
    if (sharding && sharding.active_sharding && sharding.active_sharding.model) {
      var sh = sharding.active_sharding;
      var shardNodes = sh.shard_map ? Object.keys(sh.shard_map) : [];
      // Avoid duplicating if already in models_loaded
      var already = instances.find(function (inst) { return inst.model === sh.model; });
      if (!already) {
        instances.push({
          model: sh.model,
          strategy: sh.strategy || 'pipeline',
          nodes: shardNodes,
          status: 'READY',
        });
      } else {
        already.strategy = sh.strategy || 'pipeline';
        already.nodes = shardNodes.length > 0 ? shardNodes : already.nodes;
      }
    }

    if (instances.length === 0) {
      container.innerHTML = '<div class="instances-empty">No running instances</div>';
      return;
    }

    container.innerHTML = instances.map(function (inst, idx) {
      var nodeList = inst.nodes.map(function (n) { return self.esc(n); }).join(', ');
      return '<div class="instance-card" data-idx="' + idx + '">' +
        '<div class="instance-model">' + self.esc(inst.model) + '</div>' +
        '<div class="instance-meta">' +
        '<span class="instance-strategy">' + self.esc(inst.strategy) + '</span>' +
        '<span class="instance-nodes">' + nodeList + '</span>' +
        '</div>' +
        '<div class="instance-footer">' +
        '<span class="instance-status ready">READY</span>' +
        '<button class="instance-delete" data-model="' + self.esc(inst.model) + '">DELETE</button>' +
        '</div>' +
        '</div>';
    }).join('');

    container.querySelectorAll('.instance-delete').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var model = btn.dataset.model;
        self.deleteInstance(model);
      });
    });
  },

  async deleteInstance(model) {
    try {
      var resp = await fetch('/api/models/unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: model }),
      });
      if (resp.ok) {
        this.toast('Instance stopped: ' + model, 'success');
        this.refresh();
      } else {
        var data = await resp.json().catch(function () { return {}; });
        this.toast(data.error || 'Failed to stop instance', 'error');
      }
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }
  },

  // ========================================================================
  //  RIGHT PANEL — LAUNCH INSTANCE
  // ========================================================================

  _launchModelsPopulated: false,

  bindLaunchForm() {
    var self = this;

    // Sharding pills
    var pillGroup = document.getElementById('sharding-pills');
    if (pillGroup) {
      pillGroup.querySelectorAll('.pill').forEach(function (pill) {
        pill.addEventListener('click', function () {
          pillGroup.querySelectorAll('.pill').forEach(function (p) { p.classList.remove('active'); });
          pill.classList.add('active');
        });
      });
    }

    // Node selector dots
    var nodeSelector = document.getElementById('node-selector');
    if (nodeSelector) {
      nodeSelector.querySelectorAll('.node-dot').forEach(function (dot) {
        dot.addEventListener('click', function () {
          nodeSelector.querySelectorAll('.node-dot').forEach(function (d) { d.classList.remove('active'); });
          dot.classList.add('active');
        });
      });
    }

    // Launch button
    var launchBtn = document.getElementById('launch-btn');
    if (launchBtn) {
      launchBtn.addEventListener('click', function () { self.launchInstance(); });
    }
  },

  populateLaunchModels() {
    var select = document.getElementById('launch-model');
    if (!select) return;
    var s = this.state.status;
    if (!s || !s.models_loaded) return;

    // Also fetch available models
    var self = this;
    if (!this._launchModelsPopulated) {
      this._launchModelsPopulated = true;
      this.fetchJSON('/api/models').then(function (data) {
        if (!data) return;
        var models = data.models || data.available || [];
        if (models.length === 0) return;
        var cv = select.value;
        select.innerHTML = '<option value="">-- SELECT MODEL --</option>' +
          models.map(function (m) {
            var id = typeof m === 'string' ? m : m.id || m.model_id || '';
            return '<option value="' + self.esc(id) + '">' + self.esc(id) + '</option>';
          }).join('');
        if (cv) select.value = cv;
      });
    }
  },

  async launchInstance() {
    var select = document.getElementById('launch-model');
    var model = select ? select.value : '';
    if (!model) { this.toast('Select a model first', 'error'); return; }

    var pillGroup = document.getElementById('sharding-pills');
    var strategy = 'pipeline';
    if (pillGroup) {
      var activePill = pillGroup.querySelector('.pill.active');
      if (activePill) strategy = activePill.dataset.value;
    }

    var nodeSelector = document.getElementById('node-selector');
    var minNodes = 1;
    if (nodeSelector) {
      var activeDot = nodeSelector.querySelector('.node-dot.active');
      if (activeDot) minNodes = parseInt(activeDot.dataset.value) || 1;
    }

    var launchBtn = document.getElementById('launch-btn');
    if (launchBtn) { launchBtn.disabled = true; launchBtn.textContent = 'LAUNCHING...'; }

    try {
      var endpoint, body;
      if (minNodes > 1) {
        // Sharded launch
        endpoint = '/api/sharding/launch';
        body = { model: model, strategy: strategy, min_nodes: minNodes };
      } else {
        // Single node launch
        endpoint = '/api/models/load';
        body = { model: model };
      }
      var resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var data = await resp.json();
      if (data.error) {
        this.toast(data.error, 'error');
      } else {
        this.toast('Launched: ' + model, 'success');
        this.refresh();
      }
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }

    if (launchBtn) { launchBtn.disabled = false; launchBtn.textContent = 'LAUNCH'; }
  },

  // ========================================================================
  //  CHAT — BINDING (Left Panel + Bottom Bar)
  // ========================================================================

  bindChat() {
    var self = this;
    var input = document.getElementById('chat-input');
    var send = document.getElementById('chat-send');

    if (send) {
      send.addEventListener('click', function () { self.handleSendClick(); });
    }
    if (input) {
      input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); self.handleSendClick(); }
      });
      input.addEventListener('input', function () {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 200) + 'px';
      });
    }

    // New Chat button
    var newChatBtn = document.getElementById('new-chat');
    if (newChatBtn) {
      newChatBtn.addEventListener('click', function () { self.newConversation(); });
    }

    // Search conversations
    var searchInput = document.getElementById('chat-search');
    if (searchInput) {
      searchInput.addEventListener('input', function () { self.renderConversationList(); });
    }
  },

  handleSendClick() {
    if (this.state.streaming) this.stopGeneration();
    else this.sendMessage();
  },

  stopGeneration() {
    if (this.state.abortController) {
      this.state.abortController.abort();
      this.state.abortController = null;
    }
    this.state.streaming = false;
    var sendBtn = document.getElementById('chat-send');
    if (sendBtn) { sendBtn.textContent = 'SEND'; sendBtn.classList.remove('streaming'); }
    this.toast('Generation stopped', 'info');
  },

  // ========================================================================
  //  CHAT — MODEL SELECT (Bottom Bar)
  // ========================================================================

  updateChatModelSelect() {
    var select = document.getElementById('chat-model');
    if (!select) return;
    var s = this.state.status;
    if (!s) return;
    var models = s.models_loaded || [];
    var cv = select.value;
    if (models.length === 0) {
      select.innerHTML = '<option>No models loaded</option>';
    } else {
      var self = this;
      select.innerHTML = models.map(function (m) {
        return '<option value="' + self.esc(m) + '">' + self.esc(m) + '</option>';
      }).join('');
      if (cv) {
        for (var i = 0; i < select.options.length; i++) {
          if (select.options[i].value === cv) { select.value = cv; break; }
        }
      }
    }
  },

  // ========================================================================
  //  CHAT — SEND / STREAM
  // ========================================================================

  async sendMessage() {
    var input = document.getElementById('chat-input');
    var content = input ? input.value.trim() : '';
    if (!content || this.state.streaming) return;
    input.value = '';
    input.style.height = 'auto';

    // Auto-create conversation if needed
    if (!this.state.currentConversation) this.newConversation();

    this.state.messages.push({ role: 'user', content: content });
    this.renderChatMessages();

    // Switch center-stage to show chat overlay
    this.showChatOverlay();

    var select = document.getElementById('chat-model');
    var model = select ? select.value : '';

    this.state.streaming = true;
    var sendBtn = document.getElementById('chat-send');
    if (sendBtn) { sendBtn.textContent = 'STOP'; sendBtn.classList.add('streaming'); }

    this.state.streamMetrics = { ttft: null, tps: 0, tokens: 0, startTime: performance.now(), firstTokenTime: 0 };
    this.updateStreamMetrics();

    var assistantMsg = { role: 'assistant', content: '' };
    this.state.messages.push(assistantMsg);
    this.state.abortController = new AbortController();

    try {
      var resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: model,
          messages: this.state.messages.slice(0, -1).map(function (m) { return { role: m.role, content: m.content }; }),
          stream: true,
        }),
        signal: this.state.abortController.signal,
      });

      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';

      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        var lines = buffer.split('\n');
        buffer = lines.pop();

        for (var li = 0; li < lines.length; li++) {
          var line = lines[li];
          if (!line.startsWith('data: ')) continue;
          var data = line.slice(6);
          if (data === '[DONE]') break;
          try {
            var json = JSON.parse(data);
            var delta = json.choices && json.choices[0] && json.choices[0].delta && json.choices[0].delta.content;
            if (delta) {
              if (this.state.streamMetrics.tokens === 0) {
                this.state.streamMetrics.firstTokenTime = performance.now();
                this.state.streamMetrics.ttft = Math.round(this.state.streamMetrics.firstTokenTime - this.state.streamMetrics.startTime);
              }
              this.state.streamMetrics.tokens++;
              var elapsed = (performance.now() - this.state.streamMetrics.firstTokenTime) / 1000;
              if (elapsed > 0) this.state.streamMetrics.tps = this.state.streamMetrics.tokens / elapsed;
              assistantMsg.content += delta;
              this.renderChatMessages();
              this.updateStreamMetrics();
            }
          } catch (e) { /* skip parse errors */ }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        if (!assistantMsg.content) this.state.messages.pop();
      } else {
        assistantMsg.content = 'Error: ' + err.message + '. Is the engine running?';
        this.toast('Engine not responding', 'error');
      }
    }

    this.state.streaming = false;
    this.state.abortController = null;
    if (sendBtn) { sendBtn.textContent = 'SEND'; sendBtn.classList.remove('streaming'); }
    this.renderChatMessages();
    this.saveCurrentConversation();
  },

  // ========================================================================
  //  CHAT — MESSAGE RENDERING (center-stage overlay)
  // ========================================================================

  showChatOverlay() {
    var stage = document.getElementById('center-stage');
    if (!stage) return;
    var overlay = document.getElementById('chat-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'chat-overlay';
      overlay.className = 'chat-overlay';
      overlay.innerHTML = '<div class="chat-metrics-bar" id="chat-metrics-bar">' +
        '<span id="chat-metric-ttft">TTFT: --</span>' +
        '<span id="chat-metric-tps">-- tok/s</span>' +
        '<span id="chat-metric-tokens">0 tokens</span>' +
        '</div>' +
        '<div class="chat-messages" id="chat-messages"></div>';
      stage.appendChild(overlay);
    }
    overlay.style.display = '';
  },

  hideChatOverlay() {
    var overlay = document.getElementById('chat-overlay');
    if (overlay) overlay.style.display = 'none';
  },

  updateStreamMetrics() {
    var m = this.state.streamMetrics;
    var t1 = document.getElementById('chat-metric-ttft');
    var t2 = document.getElementById('chat-metric-tps');
    var t3 = document.getElementById('chat-metric-tokens');
    if (t1) t1.textContent = m.ttft != null ? 'TTFT: ' + m.ttft + 'ms' : 'TTFT: --';
    if (t2) t2.textContent = m.tps > 0 ? m.tps.toFixed(1) + ' tok/s' : '-- tok/s';
    if (t3) t3.textContent = m.tokens + ' tokens';
  },

  renderChatMessages() {
    // Ensure overlay exists
    this.showChatOverlay();
    var container = document.getElementById('chat-messages');
    if (!container) return;
    var self = this;

    if (this.state.messages.length === 0) {
      this.hideChatOverlay();
      return;
    }

    container.innerHTML = this.state.messages.map(function (msg, i) {
      var cls = msg.role === 'user' ? 'chat-msg user' : 'chat-msg assistant';
      var html = '<div class="' + cls + '">' +
        '<div class="chat-msg-content">' + self.formatMarkdown(msg.content) + '</div>';
      if (msg.role === 'assistant' && msg.content) {
        html += '<button class="chat-copy-btn" data-msg-index="' + i + '" title="Copy">' +
          '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">' +
          '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>' +
          '<path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>' +
          '</svg></button>';
      }
      html += '</div>';
      return html;
    }).join('');

    container.scrollTop = container.scrollHeight;

    container.querySelectorAll('.chat-copy-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var msg = self.state.messages[parseInt(btn.dataset.msgIndex)];
        if (msg) {
          navigator.clipboard.writeText(msg.content).then(function () {
            self.toast('Copied to clipboard', 'success');
          }).catch(function () {
            self.toast('Failed to copy', 'error');
          });
        }
      });
    });

    // Show metrics bar during streaming
    var metricsBar = document.getElementById('chat-metrics-bar');
    if (metricsBar) metricsBar.style.display = this.state.streaming ? '' : 'none';
  },

  // ========================================================================
  //  DOWNLOADS VIEW (center-stage)
  // ========================================================================

  renderLiveCatalog(container, loaded, gpuMem, filterPills, source) {
    var self = this;
    var titles = {
      trending: '🔥 Trending Models',
      openrouter: '🚀 Most Used in Production',
      latest: '✨ Latest Releases',
    };
    var subtitles = {
      trending: 'Hot on HuggingFace right now',
      openrouter: 'Ranked by real API traffic on OpenRouter',
      latest: 'Newest text-generation models on HuggingFace',
    };

    // Clear if switching from another mode
    if (container.querySelector('#downloads-search') || container.querySelector('#hf-search-input')) {
      container.innerHTML = '';
    }

    var needsToolbar = !container.querySelector('#live-catalog-title');
    if (needsToolbar) {
      container.innerHTML =
        '<div class="downloads-header">' +
        '<h2 class="view-title" id="live-catalog-title">' + titles[source] + '</h2>' +
        '<div class="downloads-count" id="live-count">Loading...</div>' +
        '</div>' +
        '<div class="downloads-toolbar">' +
        '<div class="live-catalog-subtitle">' + subtitles[source] + '</div>' +
        '<div class="pill-group downloads-filters" id="downloads-filters">' + filterPills + '</div>' +
        '</div>' +
        '<div id="downloads-results"><div class="downloads-empty">Fetching live data...</div></div>';
    } else {
      var titleEl = container.querySelector('#live-catalog-title');
      if (titleEl) titleEl.textContent = titles[source];
      var subEl = container.querySelector('.live-catalog-subtitle');
      if (subEl) subEl.textContent = subtitles[source];
      var pillsEl = container.querySelector('#downloads-filters');
      if (pillsEl) pillsEl.innerHTML = filterPills;
    }

    var resultsContainer = container.querySelector('#downloads-results');
    var countEl = container.querySelector('#live-count');

    // Rebind filter pills
    container.querySelectorAll('.downloads-filter').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.state.modelsFilter = btn.dataset.filter;
        self.renderDownloads();
      });
    });

    fetch('/api/models/' + source)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var models = data.models || [];
        if (countEl) countEl.textContent = models.length + ' models';
        if (!resultsContainer) return;
        if (models.length === 0) {
          resultsContainer.innerHTML = '<div class="downloads-empty">No data available from ' + source + '.</div>';
          return;
        }
        var rows = models.map(function (m) {
          var isLoaded = loaded.includes(m.hf_repo);
          var sizeStr = m.size_gb > 0 ? '~' + Math.round(m.size_gb) + ' GB' : 'size unknown';
          var paramsStr = m.params_b ? m.params_b + 'B params' : '';
          var fits = gpuMem > 0 && m.size_gb > 0 && gpuMem >= (m.min_memory_gb || m.size_gb);
          var fitBadge = (gpuMem > 0 && m.size_gb > 0) ? (fits ?
            '<span class="fit-badge fits">Fits GPU</span>' :
            '<span class="fit-badge no-fit">Too large</span>') : '';
          var recBadge = m.recommended ? '<span class="fit-badge rec">Recommended</span>' : '';
          var quantBadge = m.quantization ? '<span class="fit-badge quant">' + self.esc(m.quantization.toUpperCase()) + '</span>' : '';
          var statusBadge = isLoaded ?
            '<span class="model-badge loaded">Loaded</span>' :
            '<span class="model-badge available">Available</span>';
          var ageStr = m.created_at ? self.relativeTime(m.created_at) : '';
          var downloadsStr = m.downloads ? self.formatNumber(m.downloads) + ' ⬇' : '';
          var likesStr = m.likes ? '❤ ' + self.formatNumber(m.likes) : '';
          var metaParts = [paramsStr, sizeStr, ageStr, downloadsStr, likesStr].filter(Boolean);
          var downloadBtn = !isLoaded ?
            '<button class="btn-sm downloads-download-btn" data-model-id="' + self.esc(m.hf_repo) + '">Download</button>' : '';
          return '<div class="download-card">' +
            '<div class="download-card-main">' +
            '<div class="download-card-info">' +
            '<div class="download-card-header">' +
            '<div class="download-card-name">' + self.esc(m.name || m.hf_repo) + '</div>' +
            '<div class="download-card-badges">' + recBadge + quantBadge + fitBadge + statusBadge + '</div>' +
            '</div>' +
            '<div class="download-card-repo">' + self.esc(m.hf_repo) + '</div>' +
            '<div class="download-card-desc">' + metaParts.join(' · ') + '</div>' +
            '</div>' +
            '<div class="download-card-actions">' + downloadBtn + '</div>' +
            '</div>' +
            '</div>';
        }).join('');
        resultsContainer.innerHTML = '<div class="downloads-grid">' + rows + '</div>';
      })
      .catch(function (err) {
        if (resultsContainer) resultsContainer.innerHTML = '<div class="downloads-empty">Failed to fetch: ' + self.esc(err.message || 'network error') + '</div>';
      });
  },

  relativeTime(isoString) {
    if (!isoString) return '';
    try {
      var then = new Date(isoString);
      var now = new Date();
      var diffMs = now - then;
      if (isNaN(diffMs) || diffMs < 0) return '';
      var seconds = Math.floor(diffMs / 1000);
      var minutes = Math.floor(seconds / 60);
      var hours = Math.floor(minutes / 60);
      var days = Math.floor(hours / 24);
      var months = Math.floor(days / 30);
      var years = Math.floor(days / 365);
      if (years > 0) return 'released ' + years + 'y ago';
      if (months > 0) return 'released ' + months + 'mo ago';
      if (days > 0) return 'released ' + days + 'd ago';
      if (hours > 0) return 'released ' + hours + 'h ago';
      if (minutes > 0) return 'released ' + minutes + 'm ago';
      return 'just now';
    } catch (e) {
      return '';
    }
  },

  renderHuggingFaceSearch(container, loaded, gpuMem, clusterNodeCount, totalClusterMem, filterPills, totalCount) {
    var self = this;
    var query = this.state.hfSearchQuery || '';
    // Always rebuild if we're entering HF mode from catalog mode
    var hfInput = container.querySelector('#hf-search-input');
    if (!hfInput) {
      var toolbarHtml = '<div class="downloads-header">' +
        '<h2 class="view-title">🤗 Hugging Face Hub</h2>' +
        '<div class="downloads-count" id="hf-count">Search millions of models</div>' +
        '</div>' +
        '<div class="downloads-toolbar">' +
        '<input type="text" id="hf-search-input" class="search-input" placeholder="Search HuggingFace (e.g. llama, qwen, mistral, phi)..." value="' + this.esc(query) + '" autofocus>' +
        '<div class="pill-group downloads-filters" id="downloads-filters">' + filterPills + '</div>' +
        '</div>' +
        '<div id="downloads-results"></div>';
      container.innerHTML = toolbarHtml;
    } else {
      var pillsEl = container.querySelector('#downloads-filters');
      if (pillsEl) pillsEl.innerHTML = filterPills;
    }

    var resultsContainer = container.querySelector('#downloads-results');
    var countEl = container.querySelector('#hf-count');

    // Rebind filter pills
    container.querySelectorAll('.downloads-filter').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.state.modelsFilter = btn.dataset.filter;
        self.renderDownloads();
      });
    });

    // Bind search input (debounced)
    var input = container.querySelector('#hf-search-input');
    if (input && !input.dataset.bound) {
      input.dataset.bound = '1';
      input.addEventListener('input', function () {
        self.state.hfSearchQuery = input.value;
        clearTimeout(self._hfSearchTimer);
        self._hfSearchTimer = setTimeout(function () {
          self.performHuggingFaceSearch(input.value, resultsContainer, countEl, loaded, gpuMem);
        }, 400);
      });
    }

    // Initial: if we have a query, rerun search; else show hint
    if (query) {
      this.performHuggingFaceSearch(query, resultsContainer, countEl, loaded, gpuMem);
    } else if (resultsContainer && !resultsContainer.innerHTML) {
      resultsContainer.innerHTML = '<div class="downloads-empty">Type a search query to find models on HuggingFace Hub.</div>';
    }
  },

  performHuggingFaceSearch(query, resultsContainer, countEl, loaded, gpuMem) {
    var self = this;
    if (!query || query.length < 2) {
      if (resultsContainer) resultsContainer.innerHTML = '<div class="downloads-empty">Type at least 2 characters to search.</div>';
      if (countEl) countEl.textContent = 'Search for any text-generation model';
      return;
    }

    if (resultsContainer) resultsContainer.innerHTML = '<div class="downloads-empty">Searching HuggingFace...</div>';

    fetch('/api/models/search?q=' + encodeURIComponent(query) + '&limit=50')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var models = data.models || [];
        if (countEl) countEl.textContent = models.length + ' results for "' + query + '"';
        if (models.length === 0) {
          if (resultsContainer) resultsContainer.innerHTML = '<div class="downloads-empty">No models found.</div>';
          return;
        }
        var rows = models.map(function (m) {
          var isLoaded = loaded.includes(m.hf_repo);
          var sizeStr = m.size_gb > 0 ? '~' + Math.round(m.size_gb) + ' GB' : 'size unknown';
          var fits = gpuMem > 0 && m.size_gb > 0 && gpuMem >= m.size_gb;
          var fitBadge = (gpuMem > 0 && m.size_gb > 0) ? (fits ?
            '<span class="fit-badge fits">Fits GPU</span>' :
            '<span class="fit-badge no-fit">Too large</span>') : '';
          var catalogBadge = m.in_catalog ? '<span class="fit-badge rec">In Catalog</span>' : '';
          var statusBadge = isLoaded ?
            '<span class="model-badge loaded">Loaded</span>' :
            '<span class="model-badge available">Available</span>';
          var downloadsStr = m.downloads ? self.formatNumber(m.downloads) + ' downloads' : '';
          var likesStr = m.likes ? '❤ ' + self.formatNumber(m.likes) : '';
          var statsLine = [downloadsStr, likesStr].filter(Boolean).join(' · ');
          var downloadBtn = !isLoaded ?
            '<button class="btn-nvidia btn-sm hf-download-btn" data-hf-repo="' + self.esc(m.hf_repo) + '" data-hf-slug="' + self.esc(m.id) + '">Download</button>' : '';
          return '<div class="download-card">' +
            '<div class="download-card-main">' +
            '<div class="download-card-info">' +
            '<div class="download-card-header">' +
            '<div class="download-card-name">' + self.esc(m.name) + '</div>' +
            '<div class="download-card-badges">' + catalogBadge + fitBadge + statusBadge + '</div>' +
            '</div>' +
            '<div class="download-card-repo">' + self.esc(m.hf_repo) + '</div>' +
            '<div class="download-card-desc">' + sizeStr + (statsLine ? ' &middot; ' + statsLine : '') + '</div>' +
            '</div>' +
            '<div class="download-card-actions">' + downloadBtn + '</div>' +
            '</div>' +
            '</div>';
        }).join('');
        if (resultsContainer) resultsContainer.innerHTML = '<div class="downloads-grid">' + rows + '</div>';
      })
      .catch(function (err) {
        if (resultsContainer) resultsContainer.innerHTML = '<div class="downloads-empty">Search failed: ' + err.message + '</div>';
      });
  },

  formatNumber(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
  },

  renderDownloads() {
    var container = document.getElementById('downloads-content');
    if (!container) return;
    var s = this.state.status;
    var loaded = s?.models_loaded || [];
    var self = this;
    var nodes = this.state.nodes;
    var gpuMem = s?.gpu?.memory_total_mb ? s.gpu.memory_total_mb / 1024 : 0;
    var totalClusterMem = nodes.reduce(function (sum, n) { return sum + (n.gpu_memory_gb || 0); }, 0);
    var clusterNodeCount = nodes.length;

    // Fetch catalog from API (42+ models) — lazy load once
    if (!this.state.catalog) {
      this.state.catalog = [];
      fetch('/api/models').then(function (r) { return r.json(); }).then(function (data) {
        self.state.catalog = (data.models || []).map(function (m) {
          return {
            id: m.hf_repo || m.id,
            slug: m.id,
            name: m.name,
            size: '~' + Math.round(m.size_gb) + ' GB',
            sizeGb: m.size_gb,
            desc: m.description,
            family: m.family || '',
            params: m.params_b ? m.params_b + 'B' : '',
            quantization: m.quantization,
            minMem: m.min_memory_gb || m.size_gb,
            recommended: m.recommended || false,
            created_at: m.created_at || '',
            downloads: m.downloads || 0,
            likes: m.likes || 0,
          };
        });
        self.renderDownloads();
      }).catch(function () { self.state.catalog = []; });
      container.innerHTML = '<div class="downloads-empty">Loading catalog...</div>';
      return;
    }

    var catalog = this.state.catalog;
    var query = (this.state.modelsSearch || '').toLowerCase();
    var totalCount = catalog.length;

    // Build filter pills now so HF branch can use them
    var allFilters = ['all', 'recommended', 'downloaded', 'trending', 'openrouter', 'latest', 'huggingface'];
    var filterLabels = {
      all: 'All',
      recommended: 'Recommended',
      downloaded: 'Downloaded',
      trending: '🔥 Trending',
      openrouter: '🚀 Most Used',
      latest: '✨ Latest',
      huggingface: '🤗 Search HF',
    };
    var filterPills = allFilters.map(function (f) {
      var active = self.state.modelsFilter === f ? ' active' : '';
      return '<button class="pill downloads-filter' + active + '" data-filter="' + f + '">' + filterLabels[f] + '</button>';
    }).join('');

    // Live-fetch filters — delegate to renderLiveCatalog
    if (['trending', 'openrouter', 'latest'].indexOf(this.state.modelsFilter) !== -1) {
      return this.renderLiveCatalog(container, loaded, gpuMem, filterPills, this.state.modelsFilter);
    }

    // HuggingFace live search mode
    if (this.state.modelsFilter === 'huggingface') {
      return this.renderHuggingFaceSearch(container, loaded, gpuMem, clusterNodeCount, totalClusterMem, filterPills, totalCount);
    }

    var models = catalog.filter(function (m) {
      if (query && m.id.toLowerCase().indexOf(query) === -1 &&
          m.desc.toLowerCase().indexOf(query) === -1 &&
          (m.family || '').toLowerCase().indexOf(query) === -1) return false;
      var isLoaded = loaded.includes(m.id);
      if (self.state.modelsFilter === 'downloaded' && !isLoaded) return false;
      if (self.state.modelsFilter === 'available' && isLoaded) return false;
      if (self.state.modelsFilter === 'recommended' && !m.recommended) return false;
      return true;
    });

    if (this.state.modelsSort === 'size-asc') models.sort(function (a, b) { return a.sizeGb - b.sizeGb; });
    else if (this.state.modelsSort === 'size-desc') models.sort(function (a, b) { return b.sizeGb - a.sizeGb; });
    else if (this.state.modelsSort === 'name') models.sort(function (a, b) { return a.id.localeCompare(b.id); });
    else {
      // Default: recommended first, then by size
      models.sort(function (a, b) {
        if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
        return a.sizeGb - b.sizeGb;
      });
    }

    var filteredCount = models.length;

    // If we're coming from HF mode, clear the container
    if (container.querySelector('#hf-search-input')) {
      container.innerHTML = '';
    }

    // Render toolbar once; only results re-render on search
    var needsToolbar = !container.querySelector('#downloads-search');
    var html = '';
    if (needsToolbar) {
      html += '<div class="downloads-header">' +
        '<h2 class="view-title">Model Catalog</h2>' +
        '<div class="downloads-count" id="downloads-count">' + filteredCount + ' of ' + totalCount + ' models</div>' +
        '</div>' +
        '<div class="downloads-toolbar">' +
        '<input type="text" id="downloads-search" class="search-input" placeholder="Search models, families, or descriptions..." value="' + this.esc(this.state.modelsSearch) + '">' +
        '<div class="pill-group downloads-filters" id="downloads-filters">' + filterPills + '</div>' +
        '</div>' +
        '<div id="downloads-results"></div>';
    } else {
      // Just update count + filter pills
      var countEl = container.querySelector('#downloads-count');
      if (countEl) countEl.textContent = filteredCount + ' of ' + totalCount + ' models';
      var pillsEl = container.querySelector('#downloads-filters');
      if (pillsEl) pillsEl.innerHTML = filterPills;
    }

    // Results list HTML (rebuilt on every render, but appended separately)
    var resultsRows = models.map(function (model) {
      var isLoaded = loaded.includes(model.id);
      var fits = gpuMem >= (model.minMem || model.sizeGb);
      var fitBadge = gpuMem > 0 ? (fits ?
        '<span class="fit-badge fits">Fits GPU</span>' :
        '<span class="fit-badge no-fit">Too large</span>') : '';
      var statusBadge = isLoaded ?
        '<span class="model-badge loaded">Loaded</span>' :
        '<span class="model-badge available">Available</span>';
      var recBadge = model.recommended ? '<span class="fit-badge rec">Recommended</span>' : '';
      var quantBadge = model.quantization ? '<span class="fit-badge quant">' + self.esc(model.quantization.toUpperCase()) + '</span>' : '';
      var paramsText = model.params ? model.params + ' params' : '';
      var ageText = model.created_at ? self.relativeTime(model.created_at) : '';
      var downloadsText = model.downloads ? self.formatNumber(model.downloads) + ' ⬇' : '';
      var descParts = [paramsText, model.size, ageText, downloadsText].filter(Boolean);
      var downloadBtn = !isLoaded ?
        '<button class="btn-nvidia btn-sm downloads-download-btn" data-model-id="' + self.esc(model.id) + '">Download</button>' : '';
      var shardBtn = '';
      if (!fits && clusterNodeCount > 1 && totalClusterMem >= model.sizeGb) {
        shardBtn = '<button class="btn-sm downloads-shard-btn" data-model-id="' + self.esc(model.id) + '">Shard Across Cluster</button>';
      }
      return '<div class="download-card" data-model-id="' + self.esc(model.id) + '">' +
        '<div class="download-card-main">' +
        '<div class="download-card-info">' +
        '<div class="download-card-header">' +
        '<div class="download-card-name">' + self.esc(model.name || model.id) + '</div>' +
        '<div class="download-card-badges">' + recBadge + quantBadge + fitBadge + statusBadge + '</div>' +
        '</div>' +
        '<div class="download-card-repo">' + self.esc(model.id) + '</div>' +
        '<div class="download-card-desc">' + descParts.join(' &middot; ') + (model.desc ? '<br><span class="download-card-tagline">' + self.esc(model.desc) + '</span>' : '') + '</div>' +
        '</div>' +
        '<div class="download-card-actions">' + downloadBtn + shardBtn + '</div>' +
        '</div>' +
        '</div>';
    }).join('');

    var resultsHtml = '<div class="downloads-grid">' + resultsRows + '</div>';
    if (models.length === 0) {
      resultsHtml = '<div class="downloads-empty">No models match your filters.</div>';
    }

    // First render: inject full toolbar + results
    if (needsToolbar) {
      container.innerHTML = html;
    }
    // Update only the results area so search input keeps focus
    var resultsContainer = container.querySelector('#downloads-results');
    if (resultsContainer) {
      resultsContainer.innerHTML = resultsHtml;
    }

    // Bind search (only once, on first render)
    var searchInput = document.getElementById('downloads-search');
    if (searchInput && !searchInput.dataset.bound) {
      searchInput.dataset.bound = '1';
      searchInput.addEventListener('input', function () {
        self.state.modelsSearch = searchInput.value;
        self.renderDownloads();
      });
    }

    // Bind filter pills (rebind each time since innerHTML was replaced)
    container.querySelectorAll('.downloads-filter').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.state.modelsFilter = btn.dataset.filter;
        self.renderDownloads();
      });
    });

    // Bind download buttons
    container.querySelectorAll('.downloads-download-btn').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var modelId = btn.dataset.modelId;
        // Find the catalog slug from the HF repo ID
        var modelEntry = (self.state.catalog || []).find(function (m) { return m.id === modelId; });
        var slug = modelEntry ? modelEntry.slug : modelId;
        btn.disabled = true;
        btn.textContent = 'Starting...';
        fetch('/api/models/' + encodeURIComponent(slug) + '/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        }).then(function (resp) {
          if (!resp.ok) {
            return resp.text().then(function (t) { throw new Error('HTTP ' + resp.status + ': ' + t.slice(0, 100)); });
          }
          return resp.json();
        }).then(function (data) {
          if (data.error) { self.toast(data.error, 'error'); btn.disabled = false; btn.textContent = 'Download'; return; }
          btn.textContent = 'Downloading...';
          var pollId = setInterval(function () {
            fetch('/api/models/' + encodeURIComponent(slug))
              .then(function (r) { return r.json(); })
              .then(function (st) {
                if (st.downloaded) {
                  clearInterval(pollId);
                  self.toast('Model downloaded: ' + modelId, 'success');
                  self.refresh();
                  btn.textContent = 'Downloaded';
                }
              }).catch(function () { clearInterval(pollId); btn.disabled = false; btn.textContent = 'Download'; });
          }, 2000);
        }).catch(function (err) { self.toast('Error: ' + err.message, 'error'); btn.disabled = false; btn.textContent = 'Download'; });
      });
    });

    // Bind shard buttons
    container.querySelectorAll('.downloads-shard-btn').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var modelId = btn.dataset.modelId;
        btn.disabled = true;
        btn.textContent = 'Planning...';
        fetch('/api/sharding/plan?model=' + encodeURIComponent(modelId))
          .then(function (r) { return r.json(); })
          .then(function (data) {
            if (data.error) { self.toast(data.error, 'error'); btn.disabled = false; btn.textContent = 'Shard Across Cluster'; return; }
            var plan = data.plan;
            var sm = plan.shard_map || {};
            var h = '<div class="shard-preview">';
            h += '<div class="shard-preview-title">Sharding Plan: ' + plan.strategy + '</div>';
            h += '<div class="shard-preview-meta">World size: ' + plan.world_size + ' | TP: ' + plan.tensor_parallel_size + ' | PP: ' + plan.pipeline_parallel_size + ' | Memory: ' + plan.total_memory_required_gb + ' GB</div>';
            Object.keys(sm).forEach(function (nid) {
              var s = sm[nid];
              h += '<div class="shard-node"><span class="shard-role ' + s.role + '">' + s.role.toUpperCase() + '</span> ' +
                self.esc(nid) + '<span class="shard-detail">Layers ' + (s.layers || 'all') + ' | ~' + s.estimated_memory_gb + ' GB</span></div>';
            });
            h += '<div class="shard-actions"><button class="btn-nvidia btn-sm shard-launch-btn" data-model-id="' + self.esc(modelId) + '">Launch Sharded</button>' +
              '<button class="btn-sm shard-cancel-btn">Cancel</button></div></div>';

            var card = btn.closest('.download-card');
            var existing = card.querySelector('.shard-preview');
            if (existing) existing.remove();
            var pe = document.createElement('div');
            pe.innerHTML = h;
            card.appendChild(pe.firstChild);
            btn.disabled = false;
            btn.textContent = 'Shard Across Cluster';

            card.querySelector('.shard-launch-btn').addEventListener('click', function (ev) {
              ev.stopPropagation();
              var lb = ev.target;
              lb.disabled = true;
              lb.textContent = 'Launching...';
              fetch('/api/sharding/launch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: modelId }),
              }).then(function (r) { return r.json(); }).then(function (res) {
                if (res.error) { self.toast(res.error, 'error'); lb.disabled = false; lb.textContent = 'Launch Sharded'; }
                else { self.toast('Sharded model launching: ' + modelId, 'success'); self.refresh(); }
              }).catch(function (err) { self.toast('Error: ' + err.message, 'error'); lb.disabled = false; lb.textContent = 'Launch Sharded'; });
            });

            card.querySelector('.shard-cancel-btn').addEventListener('click', function (ev) {
              ev.stopPropagation();
              card.querySelector('.shard-preview').remove();
            });
          }).catch(function (err) { self.toast('Error: ' + err.message, 'error'); btn.disabled = false; btn.textContent = 'Shard Across Cluster'; });
      });
    });
  },

  // ========================================================================
  //  MODELS VIEW (secondary, accessible from downloads or direct)
  // ========================================================================

  renderModels() {
    // Reuse downloads rendering for the models view
    this.renderDownloads();
  },

  // ========================================================================
  //  TRAINING VIEW
  // ========================================================================

  trainingModels: [
    { id: 'meta-llama/Llama-3.2-3B-Instruct', name: 'Llama 3.2 3B Instruct', size: '~6 GB' },
    { id: 'meta-llama/Llama-3.1-8B-Instruct', name: 'Llama 3.1 8B Instruct', size: '~16 GB' },
    { id: 'meta-llama/Llama-3.1-70B-Instruct-AWQ', name: 'Llama 3.1 70B AWQ', size: '~38 GB' },
    { id: 'Qwen/Qwen2.5-72B-Instruct', name: 'Qwen 2.5 72B Instruct', size: '~145 GB' },
    { id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', name: 'DeepSeek R1 7B', size: '~14 GB' },
    { id: 'mistralai/Mistral-7B-Instruct-v0.3', name: 'Mistral 7B v0.3', size: '~14 GB' },
    { id: 'microsoft/Phi-3-mini-4k-instruct', name: 'Phi-3 Mini 4K', size: '~7.5 GB' },
    { id: 'meta-llama/CodeLlama-34b-Instruct-hf', name: 'CodeLlama 34B', size: '~63 GB' },
  ],

  async renderTraining() {
    var container = document.getElementById('training-content');
    if (!container) return;
    var data = await this.fetchJSON('/api/training/jobs');
    this.state.trainingJobs = data?.jobs || [];
    switch (this.state.trainingView) {
      case 'list': this.renderTrainingList(container); break;
      case 'new': this.renderTrainingForm(container); break;
      case 'detail': await this.renderTrainingDetail(container); break;
    }
  },

  renderTrainingList(container) {
    var jobs = this.state.trainingJobs;
    var hasJobs = jobs.length > 0;
    var self = this;

    container.innerHTML =
      '<div class="training-header">' +
      '<div class="training-stat-row">' +
      '<div class="stat-card"><div class="stat-value">' + jobs.length + '</div><div class="stat-label">Total Jobs</div></div>' +
      '<div class="stat-card"><div class="stat-value">' + jobs.filter(function (j) { return j.status === 'running'; }).length + '</div><div class="stat-label">Running</div></div>' +
      '<div class="stat-card"><div class="stat-value">' + jobs.filter(function (j) { return j.status === 'completed'; }).length + '</div><div class="stat-label">Completed</div></div>' +
      '<div class="stat-card"><div class="stat-value">' + jobs.filter(function (j) { return j.status === 'pending'; }).length + '</div><div class="stat-label">Queued</div></div>' +
      '</div>' +
      '<button class="btn-nvidia" id="training-new-btn">+ New Training Job</button>' +
      '</div>' +
      (hasJobs ?
        '<div class="training-jobs-list">' +
        jobs.sort(function (a, b) { return (b.start_time || 0) - (a.start_time || 0); }).map(function (job) { return self.renderJobCard(job); }).join('') +
        '</div>' :
        '<div class="training-empty"><h3>No training jobs yet</h3><p>Create your first fine-tuning job to get started.</p></div>');

    var nb = document.getElementById('training-new-btn');
    if (nb) nb.addEventListener('click', function () { self.state.trainingView = 'new'; self.renderTraining(); });

    container.querySelectorAll('.training-job-card').forEach(function (card) {
      card.addEventListener('click', function () {
        self.state.trainingView = 'detail';
        self.state.trainingDetailId = card.dataset.jobId;
        self.state.trainingLossData = [];
        self.renderTraining();
      });
    });
  },

  renderJobCard(job) {
    var statusMap = {
      pending: { cls: 'status-pending', label: 'Pending' },
      running: { cls: 'status-running', label: 'Running' },
      completed: { cls: 'status-completed', label: 'Completed' },
      failed: { cls: 'status-failed', label: 'Failed' },
      cancelled: { cls: 'status-cancelled', label: 'Cancelled' },
    };
    var s = statusMap[job.status] || { cls: '', label: job.status };
    var modelShort = job.config?.base_model?.split('/').pop() || 'Unknown';
    var methodLabel = (job.config?.method || 'lora').toUpperCase();
    var started = job.start_time ? new Date(job.start_time * 1000).toLocaleString() : 'Not started';
    var elapsed = job.elapsed_seconds ? this.formatUptime(Math.round(job.elapsed_seconds)) : '--';

    var html = '<div class="training-job-card" data-job-id="' + this.esc(job.job_id) + '">' +
      '<div class="job-card-top">' +
      '<div><div class="job-card-model">' + this.esc(modelShort) + '</div>' +
      '<div class="job-card-meta"><span class="job-method-badge">' + methodLabel + '</span> ' + this.esc(job.job_id) + '</div></div>' +
      '<span class="job-status-badge ' + s.cls + '">' + s.label + '</span>' +
      '</div>';

    if (job.status === 'running') {
      html += '<div class="job-card-progress">' +
        '<div class="progress-bar"><div class="progress-fill" style="width:' + (job.progress || 0) + '%"></div></div>' +
        '<div class="job-progress-text">' + (job.progress || 0).toFixed(1) + '% &middot; Epoch ' + (job.current_epoch || 0) + '/' + (job.config?.num_epochs || '?') +
        (job.current_loss != null ? ' &middot; Loss: ' + job.current_loss.toFixed(4) : '') + '</div></div>';
    }

    html += '<div class="job-card-footer"><span>Started: ' + started + '</span><span>Duration: ' + elapsed + '</span></div></div>';
    return html;
  },

  renderTrainingForm(container) {
    var models = this.trainingModels;
    var optionsHtml = models.map(function (m) {
      return '<option value="' + AINode.esc(m.id) + '">' + AINode.esc(m.name) + ' (' + m.size + ')</option>';
    }).join('');

    container.innerHTML =
      '<div class="training-form-wrapper">' +
      '<div class="training-form-header">' +
      '<button class="btn-ghost" id="training-back-btn">&larr; Back to Jobs</button>' +
      '<h2>New Training Job</h2>' +
      '</div>' +
      '<form id="training-form" class="training-form">' +
      '<div class="form-section">' +
      '<h3 class="form-section-title">Model</h3>' +
      '<div class="form-group"><label class="form-label">Base Model</label><select id="train-model" class="form-select" required>' + optionsHtml + '</select></div>' +
      '<div class="form-group"><label class="form-label">Custom Model ID (optional)</label><input type="text" id="train-model-custom" class="form-input" placeholder="e.g. org/my-model"></div>' +
      '</div>' +
      '<div class="form-section">' +
      '<h3 class="form-section-title">Dataset</h3>' +
      '<div class="form-group"><label class="form-label">Dataset Path</label><input type="text" id="train-dataset" class="form-input" placeholder="my-dataset.jsonl" required>' +
      '<div class="form-hint">Place files in <code>~/.ainode/datasets/</code>. Supported: JSON, JSONL, CSV. Fields: <code>text</code>, or <code>instruction</code>+<code>output</code>, or <code>prompt</code>+<code>completion</code>.</div></div>' +
      '</div>' +
      '<div class="form-section">' +
      '<h3 class="form-section-title">Training Method</h3>' +
      '<div class="method-toggle">' +
      '<button type="button" class="method-btn active" data-method="lora"><strong>LoRA</strong><br><small>Recommended. Trains adapter weights only.</small></button>' +
      '<button type="button" class="method-btn" data-method="full"><strong>Full Fine-Tune</strong><br><small>Updates all model weights.</small></button>' +
      '</div>' +
      '<input type="hidden" id="train-method" value="lora">' +
      '</div>' +
      '<div class="form-section">' +
      '<h3 class="form-section-title collapsible" id="advanced-toggle">Advanced Settings <span class="chevron">&#9662;</span></h3>' +
      '<div class="advanced-settings collapsed" id="advanced-settings">' +
      '<div class="form-grid">' +
      '<div class="form-group"><label class="form-label">Epochs</label><input type="number" id="train-epochs" class="form-input" value="3" min="1" max="100"></div>' +
      '<div class="form-group"><label class="form-label">Batch Size</label><input type="number" id="train-batch" class="form-input" value="4" min="1" max="128"></div>' +
      '<div class="form-group"><label class="form-label">Learning Rate</label><input type="text" id="train-lr" class="form-input" value="2e-4"></div>' +
      '<div class="form-group"><label class="form-label">Max Seq Length</label><input type="number" id="train-seq-len" class="form-input" value="2048" min="128" max="32768" step="128"></div>' +
      '</div>' +
      '<div class="form-grid lora-settings" id="lora-settings">' +
      '<div class="form-group"><label class="form-label">LoRA Rank</label><input type="number" id="train-lora-rank" class="form-input" value="16" min="1" max="256"></div>' +
      '<div class="form-group"><label class="form-label">LoRA Alpha</label><input type="number" id="train-lora-alpha" class="form-input" value="32" min="1" max="512"></div>' +
      '</div>' +
      '</div>' +
      '</div>' +
      '<div class="form-actions">' +
      '<button type="button" class="btn-ghost" id="training-cancel-btn">Cancel</button>' +
      '<button type="submit" class="btn-nvidia" id="training-submit-btn">Start Training</button>' +
      '</div>' +
      '</form>' +
      '</div>';

    this.bindTrainingForm();
  },

  bindTrainingForm() {
    var self = this;
    var goBack = function () { self.state.trainingView = 'list'; self.renderTraining(); };

    var bb = document.getElementById('training-back-btn');
    if (bb) bb.addEventListener('click', goBack);
    var cb = document.getElementById('training-cancel-btn');
    if (cb) cb.addEventListener('click', goBack);

    document.querySelectorAll('.method-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.method-btn').forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        document.getElementById('train-method').value = btn.dataset.method;
        var ls = document.getElementById('lora-settings');
        if (ls) ls.style.display = btn.dataset.method === 'lora' ? '' : 'none';
      });
    });

    var at = document.getElementById('advanced-toggle');
    if (at) {
      at.addEventListener('click', function () {
        var s = document.getElementById('advanced-settings');
        if (s) s.classList.toggle('collapsed');
        at.classList.toggle('open');
      });
    }

    var f = document.getElementById('training-form');
    if (f) f.addEventListener('submit', function (e) { e.preventDefault(); self.submitTrainingJob(); });
  },

  async submitTrainingJob() {
    var submitBtn = document.getElementById('training-submit-btn');
    if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Submitting...'; }

    var customModel = document.getElementById('train-model-custom')?.value?.trim();
    var baseModel = customModel || document.getElementById('train-model')?.value;
    var method = document.getElementById('train-method')?.value || 'lora';

    var payload = {
      base_model: baseModel,
      dataset_path: document.getElementById('train-dataset')?.value?.trim(),
      method: method,
      num_epochs: parseInt(document.getElementById('train-epochs')?.value) || 3,
      batch_size: parseInt(document.getElementById('train-batch')?.value) || 4,
      learning_rate: parseFloat(document.getElementById('train-lr')?.value) || 2e-4,
      max_seq_length: parseInt(document.getElementById('train-seq-len')?.value) || 2048,
    };
    if (method === 'lora') {
      payload.lora_rank = parseInt(document.getElementById('train-lora-rank')?.value) || 16;
      payload.lora_alpha = parseInt(document.getElementById('train-lora-alpha')?.value) || 32;
    }

    try {
      var resp = await fetch('/api/training/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      var result = await resp.json();
      if (!resp.ok) {
        this.toast('Error: ' + (result.error || 'Failed to create job'), 'error');
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Start Training'; }
        return;
      }
      this.state.trainingView = 'detail';
      this.state.trainingDetailId = result.job_id;
      this.state.trainingLossData = [];
      this.renderTraining();
    } catch (err) {
      this.toast('Network error: ' + err.message, 'error');
      if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Start Training'; }
    }
  },

  async renderTrainingDetail(container) {
    var jobId = this.state.trainingDetailId;
    if (!jobId) { this.state.trainingView = 'list'; this.renderTraining(); return; }

    var jobData = await this.fetchJSON('/api/training/jobs/' + jobId);
    if (!jobData) { container.innerHTML = '<div class="training-empty"><h3>Job not found</h3></div>'; return; }

    var logsData = await this.fetchJSON('/api/training/jobs/' + jobId + '/logs?tail=200');
    var logs = logsData?.logs || [];

    // Collect loss data
    if (jobData.current_loss != null && jobData.progress > 0) {
      var lp = this.state.trainingLossData[this.state.trainingLossData.length - 1];
      if (!lp || lp.progress !== jobData.progress) {
        this.state.trainingLossData.push({ progress: jobData.progress, loss: jobData.current_loss, epoch: jobData.current_epoch });
      }
    }
    if (this.state.trainingLossData.length === 0) {
      for (var i = 0; i < logs.length; i++) {
        var marker = 'AINODE_PROGRESS:';
        var idx = logs[i].indexOf(marker);
        if (idx !== -1) {
          try {
            var p = JSON.parse(logs[i].slice(idx + marker.length));
            if (p.loss != null && p.progress != null) {
              this.state.trainingLossData.push({ progress: p.progress, loss: p.loss, epoch: p.epoch || 0 });
            }
          } catch (e) { /* skip */ }
        }
      }
    }

    var statusMap = {
      pending: { cls: 'status-pending', label: 'Pending' },
      running: { cls: 'status-running', label: 'Running' },
      completed: { cls: 'status-completed', label: 'Completed' },
      failed: { cls: 'status-failed', label: 'Failed' },
      cancelled: { cls: 'status-cancelled', label: 'Cancelled' },
    };
    var s = statusMap[jobData.status] || { cls: '', label: jobData.status };
    var cfg = jobData.config || {};
    var modelShort = cfg.base_model?.split('/').pop() || 'Unknown';
    var started = jobData.start_time ? new Date(jobData.start_time * 1000).toLocaleString() : 'Not started';
    var ended = jobData.end_time ? new Date(jobData.end_time * 1000).toLocaleString() : '--';
    var elapsed = jobData.elapsed_seconds ? this.formatUptime(Math.round(jobData.elapsed_seconds)) : '--';
    var isActive = jobData.status === 'running' || jobData.status === 'pending';

    var html = '<div class="training-detail">' +
      '<div class="training-form-header">' +
      '<button class="btn-ghost" id="training-back-btn">&larr; Back to Jobs</button>' +
      '<div class="training-detail-title"><h2>' + this.esc(modelShort) + '</h2><span class="job-status-badge ' + s.cls + '">' + s.label + '</span></div>' +
      '</div>';

    if (jobData.status === 'running') {
      html += '<div class="training-detail-progress">' +
        '<div class="progress-header"><span>' + (jobData.progress || 0).toFixed(1) + '%</span><span>Epoch ' + (jobData.current_epoch || 0) + ' / ' + (cfg.num_epochs || '?') + '</span></div>' +
        '<div class="progress-bar progress-lg"><div class="progress-fill" style="width:' + (jobData.progress || 0) + '%"></div></div>' +
        '<div class="progress-stats">' +
        (jobData.current_loss != null ? '<span>Loss: <strong>' + jobData.current_loss.toFixed(4) + '</strong></span>' : '') +
        '<span>Elapsed: <strong>' + elapsed + '</strong></span>' +
        '</div></div>';
    }

    html += '<div class="training-detail-grid">' +
      '<div class="training-config-card">' +
      '<h3>Configuration</h3>' +
      '<table class="config-table">' +
      '<tr><td>Model</td><td class="mono">' + this.esc(cfg.base_model || '') + '</td></tr>' +
      '<tr><td>Method</td><td>' + (cfg.method || 'lora').toUpperCase() + '</td></tr>' +
      '<tr><td>Dataset</td><td class="mono">' + this.esc(cfg.dataset_path || '') + '</td></tr>' +
      '<tr><td>Epochs</td><td>' + (cfg.num_epochs || '?') + '</td></tr>' +
      '<tr><td>Batch Size</td><td>' + (cfg.batch_size || '?') + '</td></tr>' +
      '<tr><td>Learning Rate</td><td>' + (cfg.learning_rate || '?') + '</td></tr>' +
      '<tr><td>Max Seq Length</td><td>' + (cfg.max_seq_length || '?') + '</td></tr>' +
      (cfg.method === 'lora' ? '<tr><td>LoRA Rank</td><td>' + (cfg.lora_rank || '?') + '</td></tr><tr><td>LoRA Alpha</td><td>' + (cfg.lora_alpha || '?') + '</td></tr>' : '') +
      '<tr><td>Job ID</td><td class="mono">' + this.esc(jobData.job_id) + '</td></tr>' +
      '<tr><td>Started</td><td>' + started + '</td></tr>' +
      '<tr><td>Ended</td><td>' + ended + '</td></tr>' +
      '<tr><td>Duration</td><td>' + elapsed + '</td></tr>' +
      '</table>' +
      (isActive ? '<div style="margin-top:16px"><button class="btn-danger" id="training-cancel-job-btn">Cancel Job</button></div>' : '') +
      '</div>' +
      '<div class="training-right-col">' +
      (this.state.trainingLossData.length > 1 ?
        '<div class="training-chart-card"><h3>Training Loss</h3><div class="loss-chart-container"><canvas id="loss-chart" width="460" height="200"></canvas></div></div>' : '') +
      '<div class="training-log-card"><h3>Logs <span class="log-line-count">' + (logsData?.total_lines || 0) + ' lines</span></h3>' +
      '<div class="training-log-viewer" id="training-log-viewer">' +
      (logs.length > 0 ? logs.map(function (l) { return '<div class="log-line">' + AINode.esc(l) + '</div>'; }).join('') : '<div class="log-empty">No logs yet</div>') +
      '</div></div>' +
      '</div></div></div>';

    container.innerHTML = html;

    var self = this;
    var bb = document.getElementById('training-back-btn');
    if (bb) bb.addEventListener('click', function () {
      self.state.trainingView = 'list';
      self.state.trainingDetailId = null;
      self.state.trainingLossData = [];
      self.renderTraining();
    });

    var cjb = document.getElementById('training-cancel-job-btn');
    if (cjb) cjb.addEventListener('click', async function () {
      if (!confirm('Cancel this training job?')) return;
      var resp = await fetch('/api/training/jobs/' + jobId, { method: 'DELETE' });
      if (resp.ok) self.renderTraining();
      else {
        var err = await resp.json().catch(function () { return {}; });
        self.toast(err.error || 'Failed to cancel job', 'error');
      }
    });

    var lv = document.getElementById('training-log-viewer');
    if (lv) lv.scrollTop = lv.scrollHeight;

    if (this.state.trainingLossData.length > 1) this.drawLossChart();
  },

  // ========================================================================
  //  TRAINING — LOSS CHART
  // ========================================================================

  drawLossChart() {
    var canvas = document.getElementById('loss-chart');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    var w = rect.width, h = rect.height;
    var pad = { top: 12, right: 16, bottom: 28, left: 48 };
    var plotW = w - pad.left - pad.right;
    var plotH = h - pad.top - pad.bottom;
    var data = this.state.trainingLossData;
    if (data.length < 2) return;

    var losses = data.map(function (d) { return d.loss; });
    var maxLoss = Math.max.apply(null, losses) * 1.05;
    var minLoss = Math.min.apply(null, losses) * 0.95;
    var maxProg = Math.max.apply(null, data.map(function (d) { return d.progress; }).concat([1]));

    var toX = function (prog) { return pad.left + (prog / maxProg) * plotW; };
    var toY = function (loss) { return pad.top + ((maxLoss - loss) / (maxLoss - minLoss || 1)) * plotH; };

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(45,55,72,0.5)';
    ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
      var y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
      var val = maxLoss - ((maxLoss - minLoss) / 4) * i;
      ctx.fillStyle = '#64748b';
      ctx.font = '10px Inter,sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(3), pad.left - 6, y + 3);
    }

    // X-axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter,sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('0%', pad.left, h - 6);
    ctx.fillText(maxProg.toFixed(0) + '%', w - pad.right, h - 6);

    // Loss line (green for command center theme)
    ctx.strokeStyle = '#76b900';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    data.forEach(function (d, i) {
      var x = toX(d.progress), y2 = toY(d.loss);
      if (i === 0) ctx.moveTo(x, y2); else ctx.lineTo(x, y2);
    });
    ctx.stroke();

    // Fill gradient
    var gradient = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
    gradient.addColorStop(0, 'rgba(118,185,0,0.15)');
    gradient.addColorStop(1, 'rgba(118,185,0,0)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    data.forEach(function (d, i) {
      var x = toX(d.progress), y2 = toY(d.loss);
      if (i === 0) ctx.moveTo(x, y2); else ctx.lineTo(x, y2);
    });
    ctx.lineTo(toX(data[data.length - 1].progress), h - pad.bottom);
    ctx.lineTo(toX(data[0].progress), h - pad.bottom);
    ctx.closePath();
    ctx.fill();

    // Endpoint dot
    var last = data[data.length - 1];
    ctx.fillStyle = '#76b900';
    ctx.beginPath();
    ctx.arc(toX(last.progress), toY(last.loss), 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0a0e17';
    ctx.lineWidth = 2;
    ctx.stroke();
  },

  // ========================================================================
  //  UTILITIES
  // ========================================================================

  esc(str) {
    var div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  },

  formatUptime(seconds) {
    if (seconds < 60) return seconds + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
    return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm';
  },

  formatMarkdown(text) {
    if (!text) return '';
    return text
      .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="code-block"><code>$2</code></pre>')
      .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
  },

  skeletonCards(n) {
    return Array(n).fill('<div class="skeleton-card"><div class="skeleton" style="height:48px;margin-bottom:8px"></div><div class="skeleton" style="height:14px;width:60%"></div></div>').join('');
  },

  gaugeColor(pct) {
    if (pct > 90) return '#ef4444';
    if (pct > 70) return '#f59e0b';
    return '#76b900';
  },

  tempColor(celsius) {
    if (celsius > 85) return '#ef4444';
    if (celsius > 70) return '#f59e0b';
    return '#76b900';
  },
};

// ========================================================================
//  BOOT
// ========================================================================

document.addEventListener('DOMContentLoaded', function () { AINode.init(); });
