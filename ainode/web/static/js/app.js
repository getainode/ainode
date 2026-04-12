/* AINode Dashboard — Single Page Application */

const AINode = {
  state: {
    currentView: 'dashboard',
    status: null,
    nodes: [],
    messages: [],
    streaming: false,
    pollInterval: null,
    trainingJobs: [],
    trainingView: 'list',
    trainingDetailId: null,
    trainingLossData: [],
    currentConversationId: null,
    conversations: [],
    abortController: null,
    historyVisible: true,
    streamMetrics: { ttft: null, tps: 0, tokenCount: 0, startTime: 0, firstTokenTime: 0 },
  },

  init() {
    this.loadConversations();
    this.bindNav();
    this.bindChat();
    this.navigate(window.location.hash.slice(1) || 'dashboard');
    this.startPolling();
  },

  // --- Toast Notification System ---
  toast(message, type) {
    type = type || 'info';
    var container = document.getElementById('toast-container');
    if (!container) return;
    var toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.innerHTML = '<span class="toast-message">' + this.esc(message) + '</span><button class="toast-close">&times;</button>';
    container.appendChild(toast);
    requestAnimationFrame(function() { toast.classList.add('toast-visible'); });
    var dismiss = function() {
      toast.classList.remove('toast-visible');
      toast.classList.add('toast-fade-out');
      setTimeout(function() { toast.remove(); }, 300);
    };
    toast.querySelector('.toast-close').addEventListener('click', dismiss);
    setTimeout(dismiss, 4000);
  },

  // --- Conversation History ---
  loadConversations() {
    try { this.state.conversations = JSON.parse(localStorage.getItem('ainode_conversations') || '[]'); }
    catch(e) { this.state.conversations = []; }
  },
  saveConversations() { localStorage.setItem('ainode_conversations', JSON.stringify(this.state.conversations)); },
  getCurrentConversation() { return this.state.conversations.find(c => c.id === this.state.currentConversationId) || null; },

  newConversation() {
    var id = 'conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    var sel = document.getElementById('chat-model');
    var conv = { id: id, title: 'New Chat', messages: [], created_at: Date.now(), model: sel ? sel.value : '' };
    this.state.conversations.unshift(conv);
    this.state.currentConversationId = id;
    this.state.messages = [];
    this.saveConversations();
    this.renderConversationList();
    this.renderMessages();
  },

  loadConversation(id) {
    var conv = this.state.conversations.find(c => c.id === id);
    if (!conv) return;
    this.state.currentConversationId = id;
    this.state.messages = conv.messages.slice();
    var select = document.getElementById('chat-model');
    if (select && conv.model) { for (var i = 0; i < select.options.length; i++) { if (select.options[i].value === conv.model) { select.value = conv.model; break; } } }
    this.renderConversationList();
    this.renderMessages();
  },

  deleteConversation(id) {
    this.state.conversations = this.state.conversations.filter(c => c.id !== id);
    if (this.state.currentConversationId === id) { this.state.currentConversationId = null; this.state.messages = []; this.renderMessages(); }
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
    var firstUser = conv.messages.find(m => m.role === 'user');
    if (firstUser) conv.title = firstUser.content.slice(0, 30) + (firstUser.content.length > 30 ? '...' : '');
    this.saveConversations();
    this.renderConversationList();
  },

  renderConversationList() {
    var list = document.getElementById('chat-history-list');
    if (!list) return;
    var self = this;
    if (this.state.conversations.length === 0) { list.innerHTML = '<div class="chat-history-empty">No conversations yet</div>'; return; }
    list.innerHTML = this.state.conversations.map(function(conv) {
      var active = conv.id === self.state.currentConversationId ? ' active' : '';
      var dateStr = new Date(conv.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
      return '<div class="chat-history-item' + active + '" data-conv-id="' + self.esc(conv.id) + '"><div class="chat-history-item-content"><div class="chat-history-item-title">' + self.esc(conv.title) + '</div><div class="chat-history-item-date">' + dateStr + '</div></div><button class="chat-history-delete" data-delete-id="' + self.esc(conv.id) + '" title="Delete">&times;</button></div>';
    }).join('');
    list.querySelectorAll('.chat-history-item').forEach(function(el) { el.addEventListener('click', function(e) { if (e.target.closest('.chat-history-delete')) return; self.loadConversation(el.dataset.convId); }); });
    list.querySelectorAll('.chat-history-delete').forEach(function(btn) { btn.addEventListener('click', function(e) { e.stopPropagation(); self.deleteConversation(btn.dataset.deleteId); }); });
  },

  toggleHistory() {
    var panel = document.getElementById('chat-history-panel');
    var openBtn = document.getElementById('chat-history-open');
    if (!panel) return;
    this.state.historyVisible = !this.state.historyVisible;
    panel.classList.toggle('collapsed', !this.state.historyVisible);
    if (openBtn) openBtn.style.display = this.state.historyVisible ? 'none' : '';
  },

  // --- Navigation ---
  bindNav() {
    var self = this;
    document.querySelectorAll('.nav-item').forEach(function(el) { el.addEventListener('click', function() { var view = el.dataset.view; if (view) self.navigate(view); }); });
  },

  navigate(view) {
    this.state.currentView = view;
    window.location.hash = view;
    document.querySelectorAll('.nav-item').forEach(function(el) { el.classList.toggle('active', el.dataset.view === view); });
    document.querySelectorAll('.view').forEach(function(el) { el.style.display = el.id === 'view-' + view ? 'block' : 'none'; });
    if (view === 'chat') { this.renderConversationList(); this.renderMessages(); }
    this.refresh();
  },

  async fetchJSON(url) { try { const resp = await fetch(url); if (!resp.ok) return null; return await resp.json(); } catch(e) { return null; } },

  async refresh() {
    const [status, nodes] = await Promise.all([this.fetchJSON('/api/status'), this.fetchJSON('/api/nodes')]);
    this.state.status = status;
    this.state.nodes = nodes?.nodes || [];
    switch (this.state.currentView) {
      case 'dashboard': this.renderDashboard(); break;
      case 'chat': this.renderChatHeader(); break;
      case 'models': this.renderModels(); break;
      case 'training': this.renderTraining(); break;
    }
  },

  startPolling() { var self = this; this.state.pollInterval = setInterval(function() { self.refresh(); }, 5000); },

  // --- Dashboard View ---
  renderDashboard() {
    var s = this.state.status, nodes = this.state.nodes;
    if (!s) { document.getElementById('dashboard-stats').innerHTML = this.skeletonCards(4); return; }
    var totalGPUs = nodes.reduce((sum, n) => sum + (n.gpu_count || 1), 0);
    var totalMem = nodes.reduce((sum, n) => sum + (n.gpu_memory_gb || 0), 0);
    var onlineNodes = nodes.filter(n => n.status === 'online').length;
    var modelsLoaded = s.models_loaded?.length || 0;
    document.getElementById('dashboard-stats').innerHTML = '<div class="card"><div class="stat-value">' + (onlineNodes || 1) + '</div><div class="stat-label">Nodes Online</div></div><div class="card"><div class="stat-value">' + (totalGPUs || 1) + '</div><div class="stat-label">GPUs</div></div><div class="card"><div class="stat-value">' + (totalMem || (s.gpu?.memory_gb || 0)) + ' GB</div><div class="stat-label">Total Memory</div></div><div class="card"><div class="stat-value">' + modelsLoaded + '</div><div class="stat-label">Models Loaded</div></div>';

    // Initialize topology visualization
    if (!this.topology) {
      var canvas = document.getElementById('topology-canvas');
      if (canvas && typeof Topology !== 'undefined') {
        this.topology = new Topology(canvas);
      }
    }
    if (this.topology) {
      var topoNodes = nodes.length > 0 ? nodes : [{
        node_id: s.node_id || 'local',
        node_name: s.node_name || 'This Node',
        gpu_name: s.gpu?.name || 'GPU',
        gpu_memory_gb: s.gpu?.memory_gb || 0,
        model: s.model || 'none',
        status: s.engine_ready ? 'online' : 'starting'
      }];
      this.topology.update(topoNodes);
    }

    this.renderNodes(nodes, s);
    this.renderClusterHealth(s);
    this.renderMonitoring();

    // Set up metrics polling (separate from main polling)
    if (!this._metricsInterval) {
      var self = this;
      this._metricsInterval = setInterval(function() {
        if (self.state.currentView === 'dashboard') self.renderMonitoring();
      }, 3000);
    }
  },

  renderNodes(nodes, status) {
    var container = document.getElementById('dashboard-nodes');
    if (!container) return;
    if (nodes.length === 0 && status) { nodes = [{ node_id: status.node_id || 'local', node_name: status.node_name || 'This Node', gpu_name: status.gpu?.name || 'Unknown GPU', gpu_memory_gb: status.gpu?.memory_gb || 0, unified_memory: status.gpu?.unified_memory || false, model: status.model || 'none', status: status.engine_ready ? 'online' : 'starting', is_leader: true }]; }
    var self = this;
    container.innerHTML = nodes.map(function(node) {
      var sc = node.status === 'online' ? 'status-online' : node.status === 'starting' ? 'status-starting' : 'status-offline';
      var lc = node.is_leader ? 'is-leader' : '';
      var ml = node.unified_memory ? 'unified' : 'VRAM';
      var mp = node.gpu_memory_used_pct || 0;
      var bc = mp > 90 ? 'red' : mp > 70 ? 'yellow' : 'green';
      return '<div class="node-card ' + lc + '"><div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:12px"><div><div class="node-name">' + self.esc(node.node_name || node.node_id) + '</div><div class="node-id">' + self.esc(node.node_id) + '</div></div><div class="status ' + sc + '"><span class="status-dot"></span>' + (node.status || 'unknown') + '</div></div><div class="node-gpu">' + self.esc(node.gpu_name || 'Unknown') + ' &middot; ' + (node.gpu_memory_gb || '?') + ' GB ' + ml + '</div><div class="node-model">' + self.esc(node.model || 'no model') + '</div><div class="progress-bar"><div class="progress-fill ' + bc + '" style="width:' + mp + '%"></div></div><div style="font-size:11px;color:var(--text-muted);margin-top:4px">Memory: ' + mp + '% used</div></div>';
    }).join('');
  },

  renderClusterHealth(status) {
    var container = document.getElementById('dashboard-health');
    if (!container) return;
    var uptime = status.uptime_seconds ? this.formatUptime(status.uptime_seconds) : 'N/A';
    container.innerHTML = '<div class="card"><div class="card-header"><div class="card-title">Engine Status</div></div><table class="table"><tr><td>Engine</td><td>' + (status.engine_ready ? '<span class="status status-online"><span class="status-dot"></span>Ready</span>' : '<span class="status status-starting"><span class="status-dot"></span>Starting</span>') + '</td></tr><tr><td>Model</td><td style="font-family:var(--font-mono)">' + this.esc(status.model || 'none') + '</td></tr><tr><td>API Port</td><td style="font-family:var(--font-mono)">' + (status.api_port || 8000) + '</td></tr><tr><td>Uptime</td><td>' + uptime + '</td></tr><tr><td>Version</td><td>' + this.esc(status.version || 'unknown') + '</td></tr></table></div>';
  },

  // --- Chat View ---
  bindChat() {
    var input = document.getElementById('chat-input'), send = document.getElementById('chat-send');
    if (!input || !send) return;
    var self = this;
    send.addEventListener('click', function() { self.handleSendClick(); });
    input.addEventListener('keydown', function(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); self.handleSendClick(); } });
    input.addEventListener('input', function() { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 200) + 'px'; });
    var nb = document.getElementById('chat-new-btn'); if (nb) nb.addEventListener('click', function() { self.newConversation(); });
    var tb = document.getElementById('chat-history-toggle'); if (tb) tb.addEventListener('click', function() { self.toggleHistory(); });
    var ob = document.getElementById('chat-history-open'); if (ob) ob.addEventListener('click', function() { self.toggleHistory(); });
  },

  handleSendClick() { if (this.state.streaming) this.stopGeneration(); else this.sendMessage(); },

  stopGeneration() {
    if (this.state.abortController) { this.state.abortController.abort(); this.state.abortController = null; }
    this.state.streaming = false;
    var sendBtn = document.getElementById('chat-send');
    if (sendBtn) { sendBtn.textContent = 'Send'; sendBtn.classList.remove('chat-stop'); sendBtn.disabled = false; }
    this.hideMetrics();
    this.toast('Generation stopped', 'info');
  },

  renderChatHeader() {
    var s = this.state.status, select = document.getElementById('chat-model');
    if (!select || !s) return;
    var models = s.models_loaded || [], cv = select.value;
    if (models.length === 0) { select.innerHTML = '<option>No models loaded</option>'; }
    else { var self = this; select.innerHTML = models.map(function(m) { return '<option value="' + self.esc(m) + '">' + self.esc(m) + '</option>'; }).join(''); if (cv) { for (var i = 0; i < select.options.length; i++) { if (select.options[i].value === cv) { select.value = cv; break; } } } }
  },

  showMetrics() { var b = document.getElementById('chat-metrics-bar'); if (b) b.style.display = ''; },
  hideMetrics() { var b = document.getElementById('chat-metrics-bar'); if (b) b.style.display = 'none'; },
  updateMetrics() {
    var m = this.state.streamMetrics;
    var t1 = document.getElementById('chat-metric-ttft'), t2 = document.getElementById('chat-metric-tps'), t3 = document.getElementById('chat-metric-tokens');
    if (t1) t1.textContent = m.ttft != null ? 'TTFT: ' + m.ttft + 'ms' : 'TTFT: --';
    if (t2) t2.textContent = m.tps > 0 ? m.tps.toFixed(1) + ' tok/s' : '-- tok/s';
    if (t3) t3.textContent = m.tokenCount + ' tokens';
  },

  async sendMessage() {
    var input = document.getElementById('chat-input'), content = input.value.trim();
    if (!content || this.state.streaming) return;
    input.value = ''; input.style.height = 'auto';
    if (!this.state.currentConversationId) this.newConversation();
    this.state.messages.push({ role: 'user', content: content });
    this.renderMessages();
    var select = document.getElementById('chat-model'), model = select ? select.value : '';
    this.state.streaming = true;
    var sendBtn = document.getElementById('chat-send');
    if (sendBtn) { sendBtn.textContent = 'Stop'; sendBtn.classList.add('chat-stop'); sendBtn.disabled = false; }
    this.state.streamMetrics = { ttft: null, tps: 0, tokenCount: 0, startTime: performance.now(), firstTokenTime: 0 };
    this.showMetrics(); this.updateMetrics();
    var assistantMsg = { role: 'assistant', content: '' };
    this.state.messages.push(assistantMsg);
    this.state.abortController = new AbortController();
    try {
      var resp = await fetch('/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: model, messages: this.state.messages.slice(0, -1).map(function(m) { return { role: m.role, content: m.content }; }), stream: true }), signal: this.state.abortController.signal });
      var reader = resp.body.getReader(), decoder = new TextDecoder(), buffer = '';
      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        var lines = buffer.split('\n'); buffer = lines.pop();
        for (var li = 0; li < lines.length; li++) {
          var line = lines[li];
          if (!line.startsWith('data: ')) continue;
          var data = line.slice(6);
          if (data === '[DONE]') break;
          try {
            var json = JSON.parse(data), delta = json.choices && json.choices[0] && json.choices[0].delta && json.choices[0].delta.content;
            if (delta) {
              if (this.state.streamMetrics.tokenCount === 0) { this.state.streamMetrics.firstTokenTime = performance.now(); this.state.streamMetrics.ttft = Math.round(this.state.streamMetrics.firstTokenTime - this.state.streamMetrics.startTime); }
              this.state.streamMetrics.tokenCount++;
              var elapsed = (performance.now() - this.state.streamMetrics.firstTokenTime) / 1000;
              if (elapsed > 0) this.state.streamMetrics.tps = this.state.streamMetrics.tokenCount / elapsed;
              assistantMsg.content += delta;
              this.renderMessages(); this.updateMetrics();
            }
          } catch(e) {}
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') { if (!assistantMsg.content) this.state.messages.pop(); }
      else { assistantMsg.content = 'Error: ' + err.message + '. Is the engine running?'; this.toast('Engine not responding', 'error'); }
    }
    this.state.streaming = false; this.state.abortController = null;
    if (sendBtn) { sendBtn.textContent = 'Send'; sendBtn.classList.remove('chat-stop'); sendBtn.disabled = false; }
    this.renderMessages(); this.saveCurrentConversation();
  },

  renderMessages() {
    var container = document.getElementById('chat-messages');
    if (!container) return;
    var self = this;
    if (this.state.messages.length === 0) {
      container.innerHTML = '<div class="chat-empty-state"><div class="chat-empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg></div><h3 class="chat-empty-title">Start a conversation</h3><p class="chat-empty-desc">Ask your local model anything. Try one of these:</p><div class="chat-suggested-prompts"><button class="chat-prompt-chip" data-prompt="Explain how transformer attention works">Explain how transformer attention works</button><button class="chat-prompt-chip" data-prompt="Write a Python script to monitor GPU usage">Write a Python script to monitor GPU usage</button><button class="chat-prompt-chip" data-prompt="What are the best practices for fine-tuning LLMs?">Best practices for fine-tuning LLMs</button><button class="chat-prompt-chip" data-prompt="Compare LoRA vs full fine-tuning">Compare LoRA vs full fine-tuning</button></div></div>';
      container.querySelectorAll('.chat-prompt-chip').forEach(function(chip) { chip.addEventListener('click', function() { var inp = document.getElementById('chat-input'); if (inp) { inp.value = chip.dataset.prompt; inp.focus(); } }); });
      return;
    }
    container.innerHTML = this.state.messages.map(function(msg, i) {
      var html = '<div class="chat-message ' + msg.role + '">' + self.formatMarkdown(msg.content);
      if (msg.role === 'assistant' && msg.content) html += '<button class="chat-copy-btn" data-msg-index="' + i + '" title="Copy to clipboard"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg></button>';
      return html + '</div>';
    }).join('');
    container.scrollTop = container.scrollHeight;
    container.querySelectorAll('.chat-copy-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var msg = self.state.messages[parseInt(btn.dataset.msgIndex)];
        if (msg) navigator.clipboard.writeText(msg.content).then(function() { self.toast('Message copied', 'success'); }).catch(function() { self.toast('Failed to copy', 'error'); });
      });
    });
  },

  // --- Models View ---
  modelsFilter: 'all',
  modelsSearch: '',
  modelsSort: 'recommended',

  renderModels() {
    var container = document.getElementById('models-list');
    if (!container) return;
    var s = this.state.status, loaded = s?.models_loaded || [], self = this;
    var gpuMem = s?.gpu?.memory_total_mb ? s.gpu.memory_total_mb / 1024 : 0;
    var recommended = [
      { id: 'meta-llama/Llama-3.2-3B-Instruct', size: '~6 GB', sizeGb: 6, desc: 'Quick start, fast inference', recommended: true },
      { id: 'meta-llama/Llama-3.1-8B-Instruct', size: '~16 GB', sizeGb: 16, desc: 'Recommended for most tasks', recommended: true },
      { id: 'meta-llama/Llama-3.1-70B-Instruct-AWQ', size: '~35 GB', sizeGb: 35, desc: 'High quality, needs 40+ GB', recommended: false },
      { id: 'Qwen/Qwen2.5-72B-Instruct', size: '~40 GB', sizeGb: 40, desc: 'Great for coding + multilingual', recommended: false },
      { id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', size: '~14 GB', sizeGb: 14, desc: 'Reasoning specialist', recommended: true },
      { id: 'mistralai/Mistral-7B-Instruct-v0.3', size: '~14 GB', sizeGb: 14, desc: 'Fast, general purpose', recommended: false },
    ];

    // Apply search filter
    var query = (this.modelsSearch || '').toLowerCase();
    var models = recommended.filter(function(m) {
      if (query && m.id.toLowerCase().indexOf(query) === -1 && m.desc.toLowerCase().indexOf(query) === -1) return false;
      var isLoaded = loaded.includes(m.id);
      if (self.modelsFilter === 'downloaded' && !isLoaded) return false;
      if (self.modelsFilter === 'available' && isLoaded) return false;
      if (self.modelsFilter === 'recommended' && !m.recommended) return false;
      return true;
    });

    // Apply sort
    if (this.modelsSort === 'size-asc') models.sort(function(a, b) { return a.sizeGb - b.sizeGb; });
    else if (this.modelsSort === 'size-desc') models.sort(function(a, b) { return b.sizeGb - a.sizeGb; });
    else if (this.modelsSort === 'name') models.sort(function(a, b) { return a.id.localeCompare(b.id); });

    var filterPills = ['all', 'downloaded', 'available', 'recommended'].map(function(f) {
      var active = self.modelsFilter === f ? ' btn-primary' : ' btn-ghost';
      return '<button class="btn btn-sm models-filter-pill' + active + '" data-filter="' + f + '">' + f.charAt(0).toUpperCase() + f.slice(1) + '</button>';
    }).join('');

    var sortSelect = '<select class="form-select" id="models-sort" style="width:auto;font-size:12px;padding:4px 8px"><option value="recommended"' + (this.modelsSort === 'recommended' ? ' selected' : '') + '>Recommended</option><option value="size-asc"' + (this.modelsSort === 'size-asc' ? ' selected' : '') + '>Size (small first)</option><option value="size-desc"' + (this.modelsSort === 'size-desc' ? ' selected' : '') + '>Size (large first)</option><option value="name"' + (this.modelsSort === 'name' ? ' selected' : '') + '>Name</option></select>';

    var html = '<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px">' +
      '<input type="text" id="models-search" class="form-input" placeholder="Search models..." value="' + this.esc(this.modelsSearch) + '" style="flex:1;min-width:200px">' +
      sortSelect + '</div>' +
      '<div style="display:flex;gap:8px;margin-bottom:16px">' + filterPills + '</div>';

    html += models.map(function(model) {
      var isLoaded = loaded.includes(model.id);
      var fits = gpuMem >= model.sizeGb;
      var fitBadge = gpuMem > 0 ? (fits ? '<span class="model-badge" style="background:rgba(34,197,94,0.15);color:#22c55e">Fits GPU</span>' : '<span class="model-badge" style="background:rgba(239,68,68,0.15);color:#ef4444">Too large</span>') : '';
      var downloadBtn = !isLoaded ? '<button class="btn btn-sm btn-primary models-download-btn" data-model-id="' + self.esc(model.id) + '">Download</button>' : '';
      var statusBadge = isLoaded ? '<span class="model-badge loaded">Loaded</span>' : '<span class="model-badge available">Available</span>';
      return '<div class="model-card" data-model-id="' + self.esc(model.id) + '"><div class="model-info"><div class="model-name">' + self.esc(model.id) + '</div><div class="model-meta">' + model.size + ' &middot; ' + self.esc(model.desc) + '</div></div><div class="model-status" style="display:flex;gap:8px;align-items:center">' + fitBadge + statusBadge + downloadBtn + '</div></div>';
    }).join('');

    if (models.length === 0) {
      html += '<div style="text-align:center;padding:24px;color:var(--text-muted)">No models match your filters.</div>';
    }

    container.innerHTML = html;

    // Bind search
    var searchInput = document.getElementById('models-search');
    if (searchInput) searchInput.addEventListener('input', function() { self.modelsSearch = searchInput.value; self.renderModels(); });

    // Bind sort
    var sortSel = document.getElementById('models-sort');
    if (sortSel) sortSel.addEventListener('change', function() { self.modelsSort = sortSel.value; self.renderModels(); });

    // Bind filter pills
    container.querySelectorAll('.models-filter-pill').forEach(function(btn) {
      btn.addEventListener('click', function() { self.modelsFilter = btn.dataset.filter; self.renderModels(); });
    });

    // Bind download buttons
    container.querySelectorAll('.models-download-btn').forEach(function(btn) {
      btn.addEventListener('click', function(e) {
        e.stopPropagation();
        var modelId = btn.dataset.modelId;
        btn.disabled = true;
        btn.textContent = 'Starting...';
        fetch('/api/models/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_id: modelId })
        }).then(function(resp) { return resp.json(); }).then(function(data) {
          if (data.error) { self.toast(data.error, 'error'); btn.disabled = false; btn.textContent = 'Download'; return; }
          btn.textContent = 'Downloading...';
          // Poll download progress
          var pollId = setInterval(function() {
            fetch('/api/models/download/status?model_id=' + encodeURIComponent(modelId)).then(function(r) { return r.json(); }).then(function(st) {
              if (st.status === 'completed') { clearInterval(pollId); self.toast('Model downloaded: ' + modelId, 'success'); self.refresh(); }
              else if (st.status === 'failed') { clearInterval(pollId); self.toast('Download failed', 'error'); btn.disabled = false; btn.textContent = 'Download'; }
              else if (st.progress != null) { btn.textContent = Math.round(st.progress) + '%'; }
            }).catch(function() { clearInterval(pollId); btn.disabled = false; btn.textContent = 'Download'; });
          }, 2000);
        }).catch(function(err) { self.toast('Error: ' + err.message, 'error'); btn.disabled = false; btn.textContent = 'Download'; });
      });
    });

    // Bind expandable model details
    container.querySelectorAll('.model-card').forEach(function(card) {
      card.addEventListener('click', function(e) {
        if (e.target.closest('.models-download-btn')) return;
        var existing = card.querySelector('.model-details-expanded');
        if (existing) { existing.remove(); return; }
        var mid = card.dataset.modelId;
        var model = recommended.find(function(m) { return m.id === mid; });
        if (!model) return;
        var detail = document.createElement('div');
        detail.className = 'model-details-expanded';
        detail.style.cssText = 'padding:12px 0 0;border-top:1px solid rgba(255,255,255,0.08);margin-top:12px;font-size:13px;color:var(--text-muted)';
        detail.innerHTML = '<div><strong>Model ID:</strong> ' + self.esc(model.id) + '</div><div><strong>Memory Required:</strong> ' + model.size + '</div><div><strong>Description:</strong> ' + self.esc(model.desc) + '</div>' + (gpuMem > 0 ? '<div><strong>GPU Fit:</strong> ' + (gpuMem >= model.sizeGb ? 'Yes (' + Math.round(gpuMem) + ' GB available)' : 'No (need ' + model.sizeGb + ' GB, have ' + Math.round(gpuMem) + ' GB)') + '</div>' : '');
        card.appendChild(detail);
      });
    });
  },

  // --- Training View ---
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
    var jobs = this.state.trainingJobs, hasJobs = jobs.length > 0, self = this;
    container.innerHTML = '<div class="training-header"><div class="training-stat-row"><div class="card training-stat-card"><div class="stat-value">' + jobs.length + '</div><div class="stat-label">Total Jobs</div></div><div class="card training-stat-card"><div class="stat-value">' + jobs.filter(j => j.status === 'running').length + '</div><div class="stat-label">Running</div></div><div class="card training-stat-card"><div class="stat-value">' + jobs.filter(j => j.status === 'completed').length + '</div><div class="stat-label">Completed</div></div><div class="card training-stat-card"><div class="stat-value">' + jobs.filter(j => j.status === 'pending').length + '</div><div class="stat-label">Queued</div></div></div><button class="btn btn-primary" id="training-new-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>New Training Job</button></div>' + (hasJobs ? '<div class="training-jobs-list">' + jobs.sort((a, b) => (b.start_time || 0) - (a.start_time || 0)).map(job => this.renderJobCard(job)).join('') + '</div>' : '<div class="training-empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="56" height="56"><path d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"/></svg><h3>No training jobs yet</h3><p>Create your first fine-tuning job to get started.</p></div>');
    var nb = document.getElementById('training-new-btn'); if (nb) nb.addEventListener('click', function() { self.state.trainingView = 'new'; self.renderTraining(); });
    container.querySelectorAll('.training-job-card').forEach(function(card) { card.addEventListener('click', function() { self.state.trainingView = 'detail'; self.state.trainingDetailId = card.dataset.jobId; self.state.trainingLossData = []; self.renderTraining(); }); });
  },

  renderJobCard(job) {
    var statusMap = { pending: { cls: 'status-pending', label: 'Pending' }, running: { cls: 'status-running', label: 'Running' }, completed: { cls: 'status-completed', label: 'Completed' }, failed: { cls: 'status-failed', label: 'Failed' }, cancelled: { cls: 'status-cancelled', label: 'Cancelled' } };
    var s = statusMap[job.status] || { cls: '', label: job.status };
    var modelShort = job.config?.base_model?.split('/').pop() || 'Unknown';
    var methodLabel = (job.config?.method || 'lora').toUpperCase();
    var started = job.start_time ? new Date(job.start_time * 1000).toLocaleString() : 'Not started';
    var elapsed = job.elapsed_seconds ? this.formatUptime(Math.round(job.elapsed_seconds)) : '--';
    var html = '<div class="training-job-card" data-job-id="' + this.esc(job.job_id) + '"><div class="job-card-top"><div class="job-card-left"><div class="job-card-model">' + this.esc(modelShort) + '</div><div class="job-card-meta"><span class="job-method-badge">' + methodLabel + '</span><span class="job-card-id">' + this.esc(job.job_id) + '</span></div></div><div class="job-card-right"><span class="job-status-badge ' + s.cls + '">' + s.label + '</span></div></div>';
    if (job.status === 'running') html += '<div class="job-card-progress"><div class="progress-bar training-progress-bar"><div class="progress-fill accent training-progress-animated" style="width:' + (job.progress || 0) + '%"></div></div><div class="job-progress-text">' + (job.progress || 0).toFixed(1) + '% &middot; Epoch ' + (job.current_epoch || 0) + '/' + (job.config?.num_epochs || '?') + (job.current_loss != null ? ' &middot; Loss: ' + job.current_loss.toFixed(4) : '') + '</div></div>';
    html += '<div class="job-card-footer"><span>Started: ' + started + '</span><span>Duration: ' + elapsed + '</span></div></div>';
    return html;
  },

  renderTrainingForm(container) {
    var models = this.trainingModels;
    var optionsHtml = models.map(m => '<option value="' + this.esc(m.id) + '">' + this.esc(m.name) + ' (' + m.size + ')</option>').join('');
    container.innerHTML = '<div class="training-form-wrapper"><div class="training-form-header"><button class="btn btn-ghost" id="training-back-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polyline points="15 18 9 12 15 6"/></svg>Back to Jobs</button><h2 class="training-form-title">New Training Job</h2></div><form id="training-form" class="training-form"><div class="form-section"><div class="form-section-title">Model</div><div class="form-group"><label class="form-label" for="train-model">Base Model</label><select id="train-model" class="form-select" required>' + optionsHtml + '</select></div><div class="form-group"><label class="form-label" for="train-model-custom">Custom Model ID (optional)</label><input type="text" id="train-model-custom" class="form-input" placeholder="e.g. org/my-model"></div></div><div class="form-section"><div class="form-section-title">Dataset</div><div class="form-group"><label class="form-label" for="train-dataset">Dataset Path</label><input type="text" id="train-dataset" class="form-input" placeholder="my-dataset.jsonl" required><div class="form-hint">Place files in <code>~/.ainode/datasets/</code> and use the filename, or provide a full path under that directory. Supported formats: JSON, JSONL, CSV. Dataset fields: <code>text</code>, or <code>instruction</code>+<code>output</code>, or <code>prompt</code>+<code>completion</code>. You can also use a HuggingFace dataset name (e.g. <code>tatsu-lab/alpaca</code>).</div></div></div><div class="form-section"><div class="form-section-title">Training Method</div><div class="form-group"><div class="method-toggle"><button type="button" class="method-btn active" data-method="lora"><div class="method-btn-title">LoRA</div><div class="method-btn-desc">Recommended. Trains adapter weights only.</div></button><button type="button" class="method-btn" data-method="full"><div class="method-btn-title">Full Fine-Tune</div><div class="method-btn-desc">Updates all model weights.</div></button></div><input type="hidden" id="train-method" value="lora"></div></div><div class="form-section"><div class="form-section-title collapsible" id="advanced-toggle"><span>Advanced Settings</span><svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polyline points="6 9 12 15 18 9"/></svg></div><div class="advanced-settings collapsed" id="advanced-settings"><div class="form-grid"><div class="form-group"><label class="form-label">Epochs</label><input type="number" id="train-epochs" class="form-input" value="3" min="1" max="100"></div><div class="form-group"><label class="form-label">Batch Size</label><input type="number" id="train-batch" class="form-input" value="4" min="1" max="128"></div><div class="form-group"><label class="form-label">Learning Rate</label><input type="text" id="train-lr" class="form-input" value="2e-4"></div><div class="form-group"><label class="form-label">Max Seq Length</label><input type="number" id="train-seq-len" class="form-input" value="2048" min="128" max="32768" step="128"></div></div><div class="form-grid lora-settings" id="lora-settings"><div class="form-group"><label class="form-label">LoRA Rank</label><input type="number" id="train-lora-rank" class="form-input" value="16" min="1" max="256"></div><div class="form-group"><label class="form-label">LoRA Alpha</label><input type="number" id="train-lora-alpha" class="form-input" value="32" min="1" max="512"></div></div></div></div><div class="form-actions"><button type="button" class="btn btn-ghost" id="training-cancel-btn">Cancel</button><button type="submit" class="btn btn-primary btn-lg" id="training-submit-btn">Start Training</button></div></form></div>';
    this.bindTrainingForm();
  },

  bindTrainingForm() {
    var self = this, goBack = function() { self.state.trainingView = 'list'; self.renderTraining(); };
    var bb = document.getElementById('training-back-btn'); if (bb) bb.addEventListener('click', goBack);
    var cb = document.getElementById('training-cancel-btn'); if (cb) cb.addEventListener('click', goBack);
    document.querySelectorAll('.method-btn').forEach(function(btn) { btn.addEventListener('click', function() { document.querySelectorAll('.method-btn').forEach(function(b) { b.classList.remove('active'); }); btn.classList.add('active'); document.getElementById('train-method').value = btn.dataset.method; var ls = document.getElementById('lora-settings'); if (ls) ls.style.display = btn.dataset.method === 'lora' ? '' : 'none'; }); });
    var at = document.getElementById('advanced-toggle'); if (at) at.addEventListener('click', function() { var s = document.getElementById('advanced-settings'), t = document.getElementById('advanced-toggle'); if (s) { s.classList.toggle('collapsed'); t.classList.toggle('open'); } });
    var f = document.getElementById('training-form'); if (f) f.addEventListener('submit', function(e) { e.preventDefault(); self.submitTrainingJob(); });
  },

  async submitTrainingJob() {
    var submitBtn = document.getElementById('training-submit-btn');
    if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Submitting...'; }
    var customModel = document.getElementById('train-model-custom')?.value?.trim();
    var baseModel = customModel || document.getElementById('train-model')?.value;
    var method = document.getElementById('train-method')?.value || 'lora';
    var payload = { base_model: baseModel, dataset_path: document.getElementById('train-dataset')?.value?.trim(), method: method, num_epochs: parseInt(document.getElementById('train-epochs')?.value) || 3, batch_size: parseInt(document.getElementById('train-batch')?.value) || 4, learning_rate: parseFloat(document.getElementById('train-lr')?.value) || 2e-4, max_seq_length: parseInt(document.getElementById('train-seq-len')?.value) || 2048 };
    if (method === 'lora') { payload.lora_rank = parseInt(document.getElementById('train-lora-rank')?.value) || 16; payload.lora_alpha = parseInt(document.getElementById('train-lora-alpha')?.value) || 32; }
    try {
      var resp = await fetch('/api/training/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      var result = await resp.json();
      if (!resp.ok) { this.toast('Error: ' + (result.error || 'Failed to create job'), 'error'); if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Start Training'; } return; }
      this.state.trainingView = 'detail'; this.state.trainingDetailId = result.job_id; this.state.trainingLossData = []; this.renderTraining();
    } catch (err) { this.toast('Network error: ' + err.message, 'error'); if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Start Training'; } }
  },

  async renderTrainingDetail(container) {
    var jobId = this.state.trainingDetailId;
    if (!jobId) { this.state.trainingView = 'list'; this.renderTraining(); return; }
    var jobData = await this.fetchJSON('/api/training/jobs/' + jobId);
    if (!jobData) { container.innerHTML = '<div class="training-empty"><h3>Job not found</h3></div>'; return; }
    var logsData = await this.fetchJSON('/api/training/jobs/' + jobId + '/logs?tail=200');
    var logs = logsData?.logs || [];
    if (jobData.current_loss != null && jobData.progress > 0) { var lp = this.state.trainingLossData[this.state.trainingLossData.length - 1]; if (!lp || lp.progress !== jobData.progress) this.state.trainingLossData.push({ progress: jobData.progress, loss: jobData.current_loss, epoch: jobData.current_epoch }); }
    if (this.state.trainingLossData.length === 0) { for (var i = 0; i < logs.length; i++) { var marker = 'AINODE_PROGRESS:', idx = logs[i].indexOf(marker); if (idx !== -1) { try { var p = JSON.parse(logs[i].slice(idx + marker.length)); if (p.loss != null && p.progress != null) this.state.trainingLossData.push({ progress: p.progress, loss: p.loss, epoch: p.epoch || 0 }); } catch(e) {} } } }
    var statusMap = { pending: { cls: 'status-pending', label: 'Pending' }, running: { cls: 'status-running', label: 'Running' }, completed: { cls: 'status-completed', label: 'Completed' }, failed: { cls: 'status-failed', label: 'Failed' }, cancelled: { cls: 'status-cancelled', label: 'Cancelled' } };
    var s = statusMap[jobData.status] || { cls: '', label: jobData.status }, cfg = jobData.config || {};
    var modelShort = cfg.base_model?.split('/').pop() || 'Unknown';
    var started = jobData.start_time ? new Date(jobData.start_time * 1000).toLocaleString() : 'Not started';
    var ended = jobData.end_time ? new Date(jobData.end_time * 1000).toLocaleString() : '--';
    var elapsed = jobData.elapsed_seconds ? this.formatUptime(Math.round(jobData.elapsed_seconds)) : '--';
    var isActive = jobData.status === 'running' || jobData.status === 'pending';
    var html = '<div class="training-detail"><div class="training-form-header"><button class="btn btn-ghost" id="training-back-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polyline points="15 18 9 12 15 6"/></svg>Back to Jobs</button><div class="training-detail-title"><h2>' + this.esc(modelShort) + '</h2><span class="job-status-badge ' + s.cls + '">' + s.label + '</span></div></div>';
    if (jobData.status === 'running') html += '<div class="training-detail-progress card"><div class="progress-header"><span class="progress-pct">' + (jobData.progress || 0).toFixed(1) + '%</span><span class="progress-epoch">Epoch ' + (jobData.current_epoch || 0) + ' / ' + (cfg.num_epochs || '?') + '</span></div><div class="progress-bar training-progress-bar-lg"><div class="progress-fill accent training-progress-animated" style="width:' + (jobData.progress || 0) + '%"></div></div><div class="progress-stats">' + (jobData.current_loss != null ? '<span>Loss: <strong>' + jobData.current_loss.toFixed(4) + '</strong></span>' : '') + '<span>Elapsed: <strong>' + elapsed + '</strong></span></div></div>';
    html += '<div class="training-detail-grid"><div class="card training-config-card"><div class="card-header"><div class="card-title">Configuration</div></div><table class="table"><tr><td>Model</td><td style="font-family:var(--font-mono)">' + this.esc(cfg.base_model || '') + '</td></tr><tr><td>Method</td><td>' + (cfg.method || 'lora').toUpperCase() + '</td></tr><tr><td>Dataset</td><td style="font-family:var(--font-mono);word-break:break-all">' + this.esc(cfg.dataset_path || '') + '</td></tr><tr><td>Epochs</td><td>' + (cfg.num_epochs || '?') + '</td></tr><tr><td>Batch Size</td><td>' + (cfg.batch_size || '?') + '</td></tr><tr><td>Learning Rate</td><td>' + (cfg.learning_rate || '?') + '</td></tr><tr><td>Max Seq Length</td><td>' + (cfg.max_seq_length || '?') + '</td></tr>' + (cfg.method === 'lora' ? '<tr><td>LoRA Rank</td><td>' + (cfg.lora_rank || '?') + '</td></tr><tr><td>LoRA Alpha</td><td>' + (cfg.lora_alpha || '?') + '</td></tr>' : '') + '<tr><td>Job ID</td><td style="font-family:var(--font-mono)">' + this.esc(jobData.job_id) + '</td></tr><tr><td>Started</td><td>' + started + '</td></tr><tr><td>Ended</td><td>' + ended + '</td></tr><tr><td>Duration</td><td>' + elapsed + '</td></tr></table>' + (isActive ? '<div style="margin-top:16px"><button class="btn btn-danger" id="training-cancel-job-btn">Cancel Job</button></div>' : '') + '</div><div class="training-right-col">' + (this.state.trainingLossData.length > 1 ? '<div class="card training-chart-card"><div class="card-header"><div class="card-title">Training Loss</div></div><div class="loss-chart-container"><canvas id="loss-chart" width="460" height="200"></canvas></div></div>' : '') + '<div class="card training-log-card"><div class="card-header"><div class="card-title">Logs</div><span class="log-line-count">' + (logsData?.total_lines || 0) + ' lines</span></div><div class="training-log-viewer" id="training-log-viewer">' + (logs.length > 0 ? logs.map(l => '<div class="log-line">' + this.esc(l) + '</div>').join('') : '<div class="log-empty">No logs yet</div>') + '</div></div></div></div></div>';
    container.innerHTML = html;
    var self = this;
    var bb = document.getElementById('training-back-btn'); if (bb) bb.addEventListener('click', function() { self.state.trainingView = 'list'; self.state.trainingDetailId = null; self.state.trainingLossData = []; self.renderTraining(); });
    var cjb = document.getElementById('training-cancel-job-btn'); if (cjb) cjb.addEventListener('click', async function() { if (!confirm('Cancel this training job?')) return; var resp = await fetch('/api/training/jobs/' + jobId, { method: 'DELETE' }); if (resp.ok) self.renderTraining(); else { var err = await resp.json().catch(function() { return {}; }); self.toast(err.error || 'Failed to cancel job', 'error'); } });
    var lv = document.getElementById('training-log-viewer'); if (lv) lv.scrollTop = lv.scrollHeight;
    if (this.state.trainingLossData.length > 1) this.drawLossChart();
  },

  drawLossChart() {
    var canvas = document.getElementById('loss-chart'); if (!canvas) return;
    var ctx = canvas.getContext('2d'), dpr = window.devicePixelRatio || 1, rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr; canvas.height = rect.height * dpr; ctx.scale(dpr, dpr);
    var w = rect.width, h = rect.height, pad = { top: 12, right: 16, bottom: 28, left: 48 };
    var plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom, data = this.state.trainingLossData;
    if (data.length < 2) return;
    var losses = data.map(d => d.loss), maxLoss = Math.max(...losses) * 1.05, minLoss = Math.min(...losses) * 0.95;
    var maxProg = Math.max(...data.map(d => d.progress), 1);
    var toX = (prog) => pad.left + (prog / maxProg) * plotW, toY = (loss) => pad.top + ((maxLoss - loss) / (maxLoss - minLoss || 1)) * plotH;
    ctx.clearRect(0, 0, w, h); ctx.strokeStyle = 'rgba(45,55,72,0.5)'; ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) { var y = pad.top + (plotH / 4) * i; ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke(); var val = maxLoss - ((maxLoss - minLoss) / 4) * i; ctx.fillStyle = '#64748b'; ctx.font = '10px Inter,sans-serif'; ctx.textAlign = 'right'; ctx.fillText(val.toFixed(3), pad.left - 6, y + 3); }
    ctx.fillStyle = '#64748b'; ctx.font = '10px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.fillText('0%', pad.left, h - 6); ctx.fillText(maxProg.toFixed(0) + '%', w - pad.right, h - 6);
    ctx.strokeStyle = '#4a90d9'; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.beginPath();
    data.forEach((d, i) => { var x = toX(d.progress), y = toY(d.loss); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
    var gradient = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom); gradient.addColorStop(0, 'rgba(74,144,217,0.15)'); gradient.addColorStop(1, 'rgba(74,144,217,0)');
    ctx.fillStyle = gradient; ctx.beginPath(); data.forEach((d, i) => { var x = toX(d.progress), y = toY(d.loss); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.lineTo(toX(data[data.length - 1].progress), h - pad.bottom); ctx.lineTo(toX(data[0].progress), h - pad.bottom); ctx.closePath(); ctx.fill();
    var last = data[data.length - 1]; ctx.fillStyle = '#4a90d9'; ctx.beginPath(); ctx.arc(toX(last.progress), toY(last.loss), 4, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = '#0a0e17'; ctx.lineWidth = 2; ctx.stroke();
  },

  // --- Monitoring Gauges ---
  async renderMonitoring() {
    var container = document.getElementById('dashboard-monitoring');
    if (!container) return;
    var data = await this.fetchJSON('/api/metrics');
    if (!data) return;
    var gpu = data.gpu || {};
    var req = data.requests || {};
    var gpuUtil = gpu.utilization_percent || 0;
    var memUsed = gpu.memory_used_mb || 0;
    var memTotal = gpu.memory_total_mb || 1;
    var temp = gpu.temperature_c || 0;
    var memPct = Math.round((memUsed / memTotal) * 100);

    container.innerHTML =
      '<div class="card" style="text-align:center;padding:16px">' + this.renderArcGauge('gpu-util', 'GPU Utilization', gpuUtil, 100, '%') + '</div>' +
      '<div class="card" style="text-align:center;padding:16px">' + this.renderArcGauge('mem-ring', 'Memory', memPct, 100, '%') + '<div style="font-size:11px;color:var(--text-muted);margin-top:4px">' + Math.round(memUsed / 1024) + ' / ' + Math.round(memTotal / 1024) + ' GB</div></div>' +
      '<div class="card" style="text-align:center;padding:16px">' + this.renderArcGauge('temp-bar', 'Temperature', temp, 100, 'C') + '</div>' +
      '<div class="card" style="padding:16px"><div style="font-weight:600;margin-bottom:8px">Requests</div>' +
      '<div style="font-size:13px;color:var(--text-muted)">Total: <strong>' + (req.total || 0) + '</strong></div>' +
      '<div style="font-size:13px;color:var(--text-muted)">Errors: <strong>' + (req.errors || 0) + '</strong></div>' +
      '<div style="font-size:13px;color:var(--text-muted)">p50: <strong>' + ((req.latency_ms || {}).p50 || 0) + ' ms</strong></div>' +
      '<div style="font-size:13px;color:var(--text-muted)">p95: <strong>' + ((req.latency_ms || {}).p95 || 0) + ' ms</strong></div>' +
      '<div style="font-size:13px;color:var(--text-muted)">tok/s: <strong>' + (req.tokens_per_second || 0) + '</strong></div></div>';
  },

  renderArcGauge(id, label, value, max, unit) {
    var pct = Math.min(value / max, 1);
    var r = 40, cx = 50, cy = 50, sw = 8;
    var circ = 2 * Math.PI * r;
    var dashOffset = circ * (1 - pct * 0.75);
    var color = unit === 'C' ? this.tempColor(value) : this.gaugeColor(pct * 100);
    return '<svg viewBox="0 0 100 100" width="100" height="100">' +
      '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="' + sw + '" stroke-dasharray="' + (circ * 0.75) + ' ' + (circ * 0.25) + '" stroke-linecap="round" transform="rotate(135 ' + cx + ' ' + cy + ')"/>' +
      '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="none" stroke="' + color + '" stroke-width="' + sw + '" stroke-dasharray="' + (circ * 0.75 - dashOffset) + ' ' + (dashOffset + circ * 0.25) + '" stroke-linecap="round" transform="rotate(135 ' + cx + ' ' + cy + ')"/>' +
      '<text x="' + cx + '" y="' + (cy + 4) + '" text-anchor="middle" fill="white" font-size="16" font-weight="600">' + Math.round(value) + '</text>' +
      '<text x="' + cx + '" y="' + (cy + 16) + '" text-anchor="middle" fill="#64748b" font-size="9">' + unit + '</text>' +
      '</svg>' +
      '<div style="font-size:11px;color:var(--text-muted);margin-top:2px">' + label + '</div>';
  },

  gaugeColor(pct) {
    if (pct > 90) return '#ef4444';
    if (pct > 70) return '#f59e0b';
    return '#22c55e';
  },

  tempColor(celsius) {
    if (celsius > 85) return '#ef4444';
    if (celsius > 70) return '#f59e0b';
    return '#22c55e';
  },

  esc(str) { var div = document.createElement('div'); div.textContent = str || ''; return div.innerHTML; },
  formatUptime(seconds) { if (seconds < 60) return seconds + 's'; if (seconds < 3600) return Math.floor(seconds / 60) + 'm'; return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm'; },
  formatMarkdown(text) { if (!text) return ''; return text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>').replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>'); },
  skeletonCards(n) { return Array(n).fill('<div class="card"><div class="skeleton" style="height:48px;margin-bottom:8px"></div><div class="skeleton" style="height:14px;width:60%"></div></div>').join(''); },
};

document.addEventListener('DOMContentLoaded', function() { AINode.init(); });
