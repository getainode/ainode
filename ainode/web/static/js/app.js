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
    trainingView: 'list',        // legacy — kept for detail/form routing
    trainingTab: 'overview',     // new: overview | datasets | runs | templates | benchmarks
    trainingDetailId: null,
    trainingLossData: [],
    trainingStats: null,
    trainingTemplates: [],
    datasets: [],
    runsFilter: 'all',
    expandedDatasetId: null,
    wizardState: null,
    pollInterval: null,
    metricsInterval: null,
    abortController: null,
    shardingStatus: null,
    currentView: 'dashboard',
    modelsFilter: 'all',
    modelsSearch: '',
    modelsSort: 'recommended',
    configSection: 'credentials',
    configData: {
      secrets: null,
      cluster: null,
      config: null,
    },
    configRevealed: {},
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
    // Restore any in-flight downloads from before page refresh
    this.loadActiveDownloads();
    // Also ask the server if there are active jobs we missed
    setTimeout(() => this.reconcileActiveDownloads(), 500);
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
    var prevView = this.state.currentView;
    this.state.currentView = view;
    // Reset downloads-view init flag when leaving the view, so we do a full render next time
    if (prevView === 'downloads' && view !== 'downloads') {
      this._downloadsViewInitialized = false;
    }
    if (view === 'downloads' && prevView !== 'downloads') {
      this._downloadsViewInitialized = false;  // force full render on entry
    }
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
    // Context-switch left panel: training shows training sidebar, else chat sidebar.
    this.updateLeftPanelContext(view);
    // Context-switch right panel: server view swaps to Model Info
    this.updateRightPanelContext(view);
    // Start/stop Server log polling
    if (view === 'server') {
      this.startServerLogPolling();
    } else {
      this.stopServerLogPolling();
    }
    this.refresh();
  },

  updateRightPanelContext(view) {
    var def = document.getElementById('right-panel-default');
    var srv = document.getElementById('right-panel-server');
    if (!def || !srv) return;
    if (view === 'server') {
      def.style.display = 'none';
      srv.style.display = '';
    } else {
      def.style.display = '';
      srv.style.display = 'none';
    }
  },

  updateLeftPanelContext(view) {
    var chatSide = document.getElementById('left-panel-chat');
    var trainSide = document.getElementById('left-panel-training');
    if (!chatSide || !trainSide) return;
    if (view === 'training') {
      chatSide.style.display = 'none';
      trainSide.style.display = '';
      this.renderTrainingSidebar();
    } else {
      trainSide.style.display = 'none';
      chatSide.style.display = '';
    }
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
      this.fetchJSON('/api/cluster/resources'),
    ]);
    this.state.status = results[0];
    this.state.nodes = results[1]?.nodes || [];
    this.state.shardingStatus = results[2];
    this.state.clusterResources = results[3];

    this.updateTopBar();
    this.updateClusterHero();
    this.updateChatModelSelect();

    // Preserve scroll position of the main content area across periodic
    // re-renders — the 5s poll was rebuilding innerHTML and bouncing the
    // user back to the top of long lists.
    var mainEl = document.querySelector('.main-content') || document.querySelector('#main') || document.scrollingElement;
    var savedScroll = mainEl ? mainEl.scrollTop : 0;

    switch (this.state.currentView) {
      case 'dashboard':
        this.renderDashboard();
        break;
      case 'downloads':
        // Don't rebuild the downloads view during periodic refresh — just update the queue
        // in place. Only do a full render when the user lands on the view.
        if (this._downloadsViewInitialized) {
          this.renderDownloadsQueue();
          this.updateNavDownloadBadge();
        } else {
          this._downloadsViewInitialized = true;
          this.renderDownloads();
        }
        break;
      case 'models':
        this.renderModels();
        break;
      case 'training':
        this.renderTraining();
        break;
      case 'config':
        this.renderConfig();
        break;
      case 'server':
        this.renderServer();
        break;
    }

    // Always update right panel
    this.renderInstances();
    this.populateLaunchModels();

    // Restore scroll
    if (mainEl && savedScroll) {
      try { mainEl.scrollTop = savedScroll; } catch (_) {}
    }
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
  //  CLUSTER HERO PILL (aggregated VRAM/GPUs)
  // ========================================================================

  updateClusterHero() {
    var r = this.state.clusterResources;
    var pill = document.getElementById('cluster-hero-pill');
    if (!pill) return;
    if (!r || !r.total_nodes) { pill.style.display = 'none'; return; }
    pill.style.display = '';
    var nodesEl = document.getElementById('hero-nodes');
    var vramEl = document.getElementById('hero-vram');
    var gpusEl = document.getElementById('hero-gpus');
    if (nodesEl) nodesEl.textContent = r.total_nodes + (r.total_nodes === 1 ? ' node' : ' nodes');
    if (vramEl) vramEl.textContent = Math.round(r.total_vram_gb) + ' GB VRAM';
    if (gpusEl) gpusEl.textContent = r.total_gpus + (r.total_gpus === 1 ? ' GPU' : ' GPUs');

    // Also surface on the Server view top bar (if the element exists).
    var srvEl = document.getElementById('server-cluster-summary');
    if (srvEl) {
      srvEl.textContent = 'Cluster: ' + r.total_nodes + ' nodes · '
        + Math.round(r.total_vram_gb) + ' GB VRAM · ' + r.total_gpus + ' GPUs';
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
    var self = this;
    var s = this.state.status;
    var loaded = (s && s.models_loaded) || [];

    this.fetchJSON('/api/models').then(function (data) {
      if (!data) return;
      var all = data.models || [];
      // Only show models that are downloaded (or currently loaded).
      var ready = all.filter(function (m) {
        if (m.downloaded) return true;
        if (loaded.indexOf(m.hf_repo) !== -1) return true;
        if (loaded.indexOf(m.id) !== -1) return true;
        return false;
      });

      var cv = select.value;
      if (ready.length === 0) {
        select.innerHTML = '<option value="">-- No models downloaded --</option>';
        return;
      }
      select.innerHTML = '<option value="">-- SELECT MODEL --</option>' +
        ready.map(function (m) {
          var repo = m.hf_repo || m.id;
          var label = m.name || repo;
          var sizeNote = m.size_gb ? ' (' + Math.round(m.size_gb) + ' GB)' : '';
          return '<option value="' + self.esc(repo) + '">' + self.esc(label) + sizeNote + '</option>';
        }).join('');
      if (cv) select.value = cv;
    });
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

    // Drag-and-drop image support on the center-stage area
    var stage = document.getElementById('center-stage');
    if (stage) {
      stage.addEventListener('dragover', function (e) {
        if (e.dataTransfer && Array.from(e.dataTransfer.items || []).some(function (i) { return i.kind === 'file'; })) {
          e.preventDefault();
          stage.classList.add('drag-over');
        }
      });
      stage.addEventListener('dragleave', function (e) {
        if (e.target === stage) stage.classList.remove('drag-over');
      });
      stage.addEventListener('drop', function (e) {
        e.preventDefault();
        stage.classList.remove('drag-over');
        var files = Array.from(e.dataTransfer.files || []).filter(function (f) {
          return f.type.startsWith('image/');
        });
        if (files.length > 0) self.attachImages(files);
        else self.toast('Only image files are supported', 'warning');
      });
    }

    // Also support paste of images
    if (input) {
      input.addEventListener('paste', function (e) {
        var items = Array.from((e.clipboardData || {}).items || []);
        var imgs = items.filter(function (i) { return i.type.startsWith('image/'); }).map(function (i) { return i.getAsFile(); }).filter(Boolean);
        if (imgs.length > 0) {
          e.preventDefault();
          self.attachImages(imgs);
        }
      });
    }
  },

  attachImages(files) {
    var self = this;
    if (!this.state.pendingAttachments) this.state.pendingAttachments = [];
    files.forEach(function (file) {
      if (file.size > 10 * 1024 * 1024) {
        self.toast(file.name + ' is too large (max 10 MB)', 'error');
        return;
      }
      var reader = new FileReader();
      reader.onload = function (e) {
        self.state.pendingAttachments.push({
          name: file.name,
          type: file.type,
          size: file.size,
          dataUrl: e.target.result,
        });
        self.renderAttachmentPreview();
      };
      reader.readAsDataURL(file);
    });
    self.toast(files.length + ' image' + (files.length > 1 ? 's' : '') + ' attached', 'success');
  },

  renderAttachmentPreview() {
    var wrapper = document.querySelector('.chat-input-wrapper');
    if (!wrapper) return;
    var existing = document.getElementById('chat-attachments');
    var attachments = this.state.pendingAttachments || [];
    if (attachments.length === 0) {
      if (existing) existing.remove();
      return;
    }
    var self = this;
    var html = attachments.map(function (att, i) {
      return '<div class="chat-attachment">' +
        '<img src="' + att.dataUrl + '" alt="' + self.esc(att.name) + '">' +
        '<button class="chat-attachment-remove" data-idx="' + i + '" title="Remove">×</button>' +
        '</div>';
    }).join('');
    if (existing) {
      existing.innerHTML = html;
    } else {
      var div = document.createElement('div');
      div.id = 'chat-attachments';
      div.className = 'chat-attachments';
      div.innerHTML = html;
      wrapper.insertBefore(div, wrapper.firstChild);
    }
    document.querySelectorAll('.chat-attachment-remove').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.state.pendingAttachments.splice(parseInt(btn.dataset.idx), 1);
        self.renderAttachmentPreview();
      });
    });
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

    // Auto-navigate to chat view when sending
    this.navigate('chat');
    this.renderChatMessages();
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
              this.updateStreamingMessage(assistantMsg);
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
    var mount = document.getElementById('view-chat-mount');
    if (!mount) return;
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
      mount.appendChild(overlay);
    }
    // Hide the empty state when we have messages
    var empty = document.getElementById('chat-view-empty');
    if (empty) empty.style.display = this.state.messages.length > 0 ? 'none' : '';
    overlay.style.display = this.state.messages.length > 0 ? '' : 'none';
  },

  hideChatOverlay() {
    var overlay = document.getElementById('chat-overlay');
    if (overlay) overlay.style.display = 'none';
    var empty = document.getElementById('chat-view-empty');
    if (empty) empty.style.display = '';
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

  // Targeted update for streaming — only rewrites the current assistant message
  updateStreamingMessage(assistantMsg) {
    var container = document.getElementById('chat-messages');
    if (!container) { this.renderChatMessages(); return; }
    var last = container.lastElementChild;
    if (!last || !last.classList.contains('assistant')) {
      // Not yet rendered — full render once
      this.renderChatMessages();
      return;
    }
    var contentEl = last.querySelector('.chat-msg-content');
    if (!contentEl) { this.renderChatMessages(); return; }
    contentEl.innerHTML = this.formatMarkdown(assistantMsg.content);
    // Auto-scroll to bottom only if user hasn't scrolled up
    var atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 120;
    if (atBottom) container.scrollTop = container.scrollHeight;
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

    // Code block copy buttons
    container.querySelectorAll('.code-copy-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var codeEl = document.getElementById(btn.dataset.codeId);
        if (!codeEl) return;
        var code = codeEl.textContent;
        navigator.clipboard.writeText(code).then(function () {
          var label = btn.querySelector('span');
          if (label) {
            var orig = label.textContent;
            label.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(function () {
              label.textContent = orig;
              btn.classList.remove('copied');
            }, 1500);
          }
          self.toast('Code copied', 'success');
        }).catch(function () { self.toast('Failed to copy', 'error'); });
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
        '<div id="downloads-queue" class="downloads-queue"></div>' +
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
          var capBadges = self.renderCapabilityBadges(m);
          var statusBadge = isLoaded ?
            '<span class="model-badge loaded">Loaded</span>' :
            '<span class="model-badge available">Available</span>';
          var ageStr = m.created_at ? self.relativeTime(m.created_at) : '';
          var downloadsStr = m.downloads ? self.formatNumber(m.downloads) + ' ⬇' : '';
          var likesStr = m.likes ? '❤ ' + self.formatNumber(m.likes) : '';
          var metaParts = [paramsStr, sizeStr, ageStr, downloadsStr, likesStr].filter(Boolean);
          var detailsBtn = '<button class="btn-sm download-details-btn" data-info-repo="' + self.esc(m.hf_repo) + '">Details</button>';
          var downloadBtn = !isLoaded ?
            '<button class="btn-sm downloads-download-btn" data-model-id="' + self.esc(m.hf_repo) + '">Download</button>' : '';
          return '<div class="download-card" data-model-id="' + self.esc(m.hf_repo) + '">' +
            '<div class="download-card-main">' +
            '<div class="download-card-info">' +
            '<div class="download-card-header">' +
            '<div class="download-card-name">' + self.esc(m.name || m.hf_repo) + '</div>' +
            '<div class="download-card-badges">' + recBadge + quantBadge + capBadges + fitBadge + statusBadge + '</div>' +
            '</div>' +
            '<div class="download-card-repo">' + self.esc(m.hf_repo) + '</div>' +
            '<div class="download-card-desc">' + metaParts.join(' · ') + '</div>' +
            '</div>' +
            '<div class="download-card-actions">' + detailsBtn + downloadBtn + '</div>' +
            '</div>' +
            '</div>';
        }).join('');
        resultsContainer.innerHTML = '<div class="downloads-grid">' + rows + '</div>';
        self.bindRepoDownloadButtons(resultsContainer);
      })
      .catch(function (err) {
        if (resultsContainer) resultsContainer.innerHTML = '<div class="downloads-empty">Failed to fetch: ' + self.esc(err.message || 'network error') + '</div>';
      });
  },

  bindRepoDownloadButtons(container) {
    var self = this;
    container.querySelectorAll('.downloads-download-btn').forEach(function (btn) {
      if (btn.dataset.bound) return;
      btn.dataset.bound = '1';
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var hfRepo = btn.dataset.modelId;
        if (!hfRepo || hfRepo.indexOf('/') === -1) {
          self.toast('Invalid model repo', 'error');
          return;
        }
        self.startRepoDownload(hfRepo);
      });
    });

    // Details button + capability badges open the detail modal
    container.querySelectorAll('[data-info-repo]').forEach(function (el) {
      if (el.dataset.infoBound) return;
      el.dataset.infoBound = '1';
      el.style.cursor = 'pointer';
      el.addEventListener('click', function (e) {
        e.stopPropagation();
        var repo = el.dataset.infoRepo;
        if (repo) self.showModelDetail(repo);
      });
    });

    // Delete buttons
    container.querySelectorAll('.downloads-delete-btn').forEach(function (btn) {
      if (btn.dataset.bound) return;
      btn.dataset.bound = '1';
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var repo = btn.dataset.modelId;
        self.confirmDeleteModel(repo);
      });
    });
  },

  confirmDeleteModel(hfRepo) {
    var self = this;
    var existing = document.getElementById('confirm-delete-modal');
    if (existing) existing.remove();

    var modal = document.createElement('div');
    modal.id = 'confirm-delete-modal';
    modal.className = 'model-detail-modal-overlay';
    modal.innerHTML =
      '<div class="model-detail-modal" style="max-width:480px">' +
        '<div class="md-header">' +
          '<div class="md-header-left">' +
            '<div class="md-icon" style="color:var(--red);border-color:rgba(255,51,51,0.4)">⚠</div>' +
            '<div class="md-title">Delete Model</div>' +
          '</div>' +
          '<button class="md-close">×</button>' +
        '</div>' +
        '<div class="md-description">' +
          'Permanently remove <strong>' + self.esc(hfRepo) + '</strong> from disk?' +
          '<br><span style="color:var(--text-muted);font-size:13px">This frees the disk space immediately. You can re-download anytime.</span>' +
        '</div>' +
        '<div class="md-footer">' +
          '<button class="btn-sm" id="cd-cancel" style="background:transparent;color:var(--text-secondary);border:1px solid var(--border-hover)">Cancel</button>' +
          '<button class="btn-sm" id="cd-confirm" style="background:var(--red);color:#fff;border:1px solid var(--red);font-weight:700;letter-spacing:0.5px;padding:10px 22px">DELETE</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(modal);

    var close = function () { modal.remove(); };
    modal.querySelector('.md-close').addEventListener('click', close);
    modal.querySelector('#cd-cancel').addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });

    modal.querySelector('#cd-confirm').addEventListener('click', function () {
      var btn = modal.querySelector('#cd-confirm');
      btn.disabled = true;
      btn.textContent = 'Deleting...';
      fetch('/api/models/delete-repo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hf_repo: hfRepo }),
      }).then(function (r) { return r.json(); }).then(function (data) {
        if (data.error) {
          self.toast('Delete failed: ' + data.error, 'error');
          btn.disabled = false;
          btn.textContent = 'DELETE';
          return;
        }
        self.toast('Deleted ' + hfRepo + ' (freed ' + (data.freed_gb || '?') + ' GB)', 'success');
        close();
        self.state.catalog = null;  // force catalog refresh
        self.refresh();
        self._downloadsViewInitialized = false;
        self.renderDownloads();
      }).catch(function (err) {
        self.toast('Delete failed: ' + err.message, 'error');
        btn.disabled = false;
        btn.textContent = 'DELETE';
      });
    });
  },

  // Persist active downloads to localStorage so page refresh doesn't lose them
  saveActiveDownloads() {
    try {
      var toSave = {};
      Object.keys(this.state.activeDownloads || {}).forEach(function (repo) {
        var dl = this.state.activeDownloads[repo];
        toSave[repo] = {
          jobId: dl.jobId,
          hfRepo: dl.hfRepo,
          startedAt: dl.startedAt,
          status: dl.status,
          totalBytes: dl.totalBytes || 0,
          downloadedBytes: dl.downloadedBytes || 0,
          progress: dl.progress,
        };
      }, this);
      localStorage.setItem('ainode_active_downloads', JSON.stringify(toSave));
    } catch (e) { /* quota or disabled */ }
  },

  loadActiveDownloads() {
    try {
      var raw = localStorage.getItem('ainode_active_downloads');
      if (!raw) return;
      var saved = JSON.parse(raw) || {};
      this.state.activeDownloads = this.state.activeDownloads || {};
      var self = this;
      Object.keys(saved).forEach(function (repo) {
        var dl = saved[repo];
        if (!dl || !dl.jobId) return;
        // Discard entries older than 12 hours — stale
        if (dl.startedAt && Date.now() - dl.startedAt > 12 * 3600 * 1000) return;
        if (dl.status === 'completed' || dl.status === 'failed') return;
        self.state.activeDownloads[repo] = Object.assign({ elapsed: 0 }, dl);
        self.resumeDownloadPolling(repo);
      });
    } catch (e) { /* ignore */ }
  },

  resumeDownloadPolling(hfRepo) {
    var self = this;
    var dl = self.state.activeDownloads[hfRepo];
    if (!dl || !dl.jobId) return;
    if (dl._pollId) return;  // already polling

    dl._pollId = setInterval(function () {
      self.pollDownloadOnce(hfRepo);
    }, 2000);

    dl._tickId = setInterval(function () {
      var d = self.state.activeDownloads[hfRepo];
      if (!d || d.status !== 'downloading') return;
      d.elapsed = Math.floor((Date.now() - d.startedAt) / 1000);
      self.renderQueueItemInPlace(hfRepo);
    }, 1000);
  },

  pollDownloadOnce(hfRepo) {
    var self = this;
    var dl = self.state.activeDownloads && self.state.activeDownloads[hfRepo];
    if (!dl) return;

    fetch('/api/models/download/status?job_id=' + encodeURIComponent(dl.jobId))
      .then(function (r) { return r.json(); })
      .then(function (st) {
        if (st.error || st.status === 'unknown') {
          // Job vanished — treat as probably completed if the model is downloaded
          self.stopDownloadPolling(hfRepo);
          self.checkIfDownloaded(hfRepo);
          return;
        }
        dl.status = st.status;
        dl.elapsed = Math.floor((Date.now() - dl.startedAt) / 1000);
        dl.totalBytes = st.total_bytes || dl.totalBytes || 0;
        // Monotonic — only go up, never down
        var newBytes = st.downloaded_bytes || 0;
        if (newBytes >= (dl.downloadedBytes || 0)) dl.downloadedBytes = newBytes;
        if (st.progress != null && (dl.progress == null || st.progress >= dl.progress)) {
          dl.progress = st.progress;
        }

        self.saveActiveDownloads();
        self.renderQueueItemInPlace(hfRepo);
        self.updateNavDownloadBadge();

        if (st.status === 'completed') {
          self.stopDownloadPolling(hfRepo);
          self.toast('Downloaded: ' + hfRepo, 'success');
          self.state.catalog = null;  // refresh catalog next time
          // Keep in queue 8s so user sees completion, then remove
          dl.status = 'completed';
          dl.progress = 100;
          dl.downloadedBytes = dl.totalBytes || dl.downloadedBytes;
          self.renderQueueItemInPlace(hfRepo);
          setTimeout(function () {
            delete self.state.activeDownloads[hfRepo];
            self.saveActiveDownloads();
            self.renderDownloadsQueue();
            self.updateNavDownloadBadge();
          }, 6000);
        } else if (st.status === 'failed') {
          self.stopDownloadPolling(hfRepo);
          self.toast('Download failed: ' + (st.error || 'unknown'), 'error');
          dl.error = st.error;
          self.renderQueueItemInPlace(hfRepo);
          setTimeout(function () {
            delete self.state.activeDownloads[hfRepo];
            self.saveActiveDownloads();
            self.renderDownloadsQueue();
            self.updateNavDownloadBadge();
          }, 10000);
        }
      })
      .catch(function () { /* keep polling on transient errors */ });
  },

  stopDownloadPolling(hfRepo) {
    var dl = this.state.activeDownloads && this.state.activeDownloads[hfRepo];
    if (!dl) return;
    if (dl._pollId) { clearInterval(dl._pollId); dl._pollId = null; }
    if (dl._tickId) { clearInterval(dl._tickId); dl._tickId = null; }
  },

  checkIfDownloaded(hfRepo) {
    var self = this;
    var slug = hfRepo.replace(/\//g, '--').toLowerCase();
    fetch('/api/models/' + encodeURIComponent(slug)).then(function (r) {
      return r.ok ? r.json() : null;
    }).then(function (info) {
      var dl = self.state.activeDownloads && self.state.activeDownloads[hfRepo];
      if (!dl) return;
      if (info && info.downloaded) {
        dl.status = 'completed';
        dl.progress = 100;
        self.toast('Downloaded: ' + hfRepo, 'success');
      } else {
        dl.status = 'failed';
        dl.error = 'Job expired';
      }
      self.renderQueueItemInPlace(hfRepo);
      self.saveActiveDownloads();
      setTimeout(function () {
        delete self.state.activeDownloads[hfRepo];
        self.saveActiveDownloads();
        self.renderDownloadsQueue();
        self.updateNavDownloadBadge();
      }, 6000);
    }).catch(function () { /* ignore */ });
  },

  // Update just one queue item's DOM in place — no re-rendering
  renderQueueItemInPlace(hfRepo) {
    var dl = this.state.activeDownloads && this.state.activeDownloads[hfRepo];
    if (!dl) return;
    var queue = document.getElementById('downloads-queue');
    if (!queue) return;
    if (queue.style.display === 'none' || !queue.children.length) {
      this.renderDownloadsQueue();
      return;
    }
    var existing = queue.querySelector('[data-queue-repo="' + CSS.escape(hfRepo) + '"]');
    if (!existing) {
      this.renderDownloadsQueue();
      return;
    }
    existing.outerHTML = this.renderQueueItem(dl);
  },

  // Reconcile on startup — ask server for active jobs that might be ours
  reconcileActiveDownloads() {
    var self = this;
    fetch('/api/models/downloads/active')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var jobs = (data.jobs || []).filter(function (j) {
          return j.status === 'downloading';
        });
        jobs.forEach(function (job) {
          var repo = job.model_id;
          if (!repo || !repo.includes('/')) return;
          if (self.state.activeDownloads && self.state.activeDownloads[repo]) return;
          self.state.activeDownloads = self.state.activeDownloads || {};
          self.state.activeDownloads[repo] = {
            jobId: job.job_id,
            hfRepo: repo,
            status: 'downloading',
            startedAt: Date.now() - 1000,
            elapsed: 0,
            totalBytes: job.total_bytes || 0,
            downloadedBytes: job.downloaded_bytes || 0,
            progress: job.progress,
          };
          self.resumeDownloadPolling(repo);
          self.toast('Resumed tracking: ' + repo, 'info');
        });
        self.saveActiveDownloads();
        self.renderDownloadsQueue();
        self.updateNavDownloadBadge();
      }).catch(function () { /* ignore */ });
  },

  startRepoDownload(hfRepo) {
    var self = this;
    if (!self.state.activeDownloads) self.state.activeDownloads = {};

    if (self.state.activeDownloads[hfRepo] &&
        self.state.activeDownloads[hfRepo].status === 'downloading') {
      self.toast('Already downloading ' + hfRepo, 'info');
      self.navigate('downloads');
      return;
    }

    fetch('/api/models/download-repo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hf_repo: hfRepo }),
    }).then(function (resp) {
      if (!resp.ok) return resp.text().then(function (t) { throw new Error('HTTP ' + resp.status + ': ' + t.slice(0, 80)); });
      return resp.json();
    }).then(function (data) {
      if (data.error) { self.toast(data.error, 'error'); return; }

      self.state.activeDownloads[hfRepo] = {
        jobId: data.job_id,
        status: 'downloading',
        startedAt: Date.now(),
        elapsed: 0,
        hfRepo: hfRepo,
        totalBytes: 0,
        downloadedBytes: 0,
        progress: null,
      };
      self.saveActiveDownloads();

      self.navigate('downloads');
      self.renderDownloadsQueue();
      self.toast('Downloading ' + hfRepo, 'info');
      self.resumeDownloadPolling(hfRepo);
      self.updateNavDownloadBadge();
    }).catch(function (err) {
      self.toast('Download failed: ' + err.message, 'error');
    });
  },

  renderDownloadsQueue() {
    var container = document.getElementById('downloads-queue');
    if (!container) return;
    var self = this;
    var active = Object.values(this.state.activeDownloads || {});
    if (active.length === 0) {
      container.innerHTML = '';
      container.style.display = 'none';
      return;
    }
    container.style.display = '';
    container.innerHTML =
      '<div class="queue-header">' +
        '<span class="queue-title">⬇ Downloads Queue</span>' +
        '<span class="queue-count">' + active.length + ' active</span>' +
      '</div>' +
      '<div class="queue-list">' +
        active.map(function (dl) {
          return self.renderQueueItem(dl);
        }).join('') +
      '</div>';
  },

  renderQueueItem(dl) {
    var mins = Math.floor(dl.elapsed / 60);
    var secs = dl.elapsed % 60;
    var elapsedStr = mins > 0 ? mins + 'm ' + secs + 's' : secs + 's';

    var statusClass = dl.status === 'completed' ? 'done' :
                      dl.status === 'failed' ? 'failed' :
                      'active';

    var pct = dl.progress;
    var hasPct = pct != null && !isNaN(pct);
    var pctStr = hasPct ? pct.toFixed(1) + '%' : '';

    var total = dl.totalBytes || 0;
    var got = dl.downloadedBytes || 0;
    var sizeStr = '';
    if (total > 0) {
      sizeStr = this.formatBytes(got) + ' / ' + this.formatBytes(total);
    } else if (got > 0) {
      sizeStr = this.formatBytes(got) + ' downloaded';
    }

    // Throughput + ETA
    var rate = dl.elapsed > 0 ? got / dl.elapsed : 0;
    var rateStr = rate > 0 ? this.formatBytes(rate) + '/s' : '';
    var etaStr = '';
    if (rate > 0 && total > 0 && got < total) {
      var remaining = (total - got) / rate;
      etaStr = 'ETA ' + this.formatDuration(remaining);
    }

    var statusLabel;
    var barHtml = '';
    if (dl.status === 'completed') {
      statusLabel = '✓ Complete · ' + (total > 0 ? this.formatBytes(total) : elapsedStr);
    } else if (dl.status === 'failed') {
      statusLabel = '⚠ Failed' + (dl.error ? ' — ' + this.esc(dl.error) : '');
    } else if (hasPct) {
      statusLabel = pctStr + ' · ' + sizeStr + (rateStr ? ' · ' + rateStr : '') + (etaStr ? ' · ' + etaStr : '');
      barHtml =
        '<div class="queue-item-bar">' +
          '<div class="queue-item-bar-progress" style="width:' + pct.toFixed(2) + '%"></div>' +
        '</div>';
    } else {
      // No total yet — show indeterminate shimmer
      statusLabel = 'Starting... ' + elapsedStr + (sizeStr ? ' · ' + sizeStr : '');
      barHtml =
        '<div class="queue-item-bar">' +
          '<div class="queue-item-bar-fill"></div>' +
        '</div>';
    }

    return '<div class="queue-item ' + statusClass + '" data-queue-repo="' + this.esc(dl.hfRepo) + '">' +
      '<div class="queue-item-info">' +
        '<div class="queue-item-repo">' + this.esc(dl.hfRepo) + '</div>' +
        '<div class="queue-item-status">' + statusLabel + '</div>' +
      '</div>' +
      barHtml +
    '</div>';
  },

  formatBytes(bytes) {
    if (!bytes || bytes < 0) return '0 B';
    if (bytes < 1024) return bytes.toFixed(0) + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    if (bytes < 1024 * 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    return (bytes / (1024 * 1024 * 1024 * 1024)).toFixed(2) + ' TB';
  },

  formatDuration(seconds) {
    if (!seconds || seconds < 0 || !isFinite(seconds)) return '—';
    seconds = Math.round(seconds);
    if (seconds < 60) return seconds + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + (seconds % 60) + 's';
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    return h + 'h ' + m + 'm';
  },

  updateNavDownloadBadge() {
    var navPill = document.querySelector('.nav-pill[data-view="downloads"]');
    if (!navPill) return;
    var active = Object.values(this.state.activeDownloads || {}).filter(function (dl) {
      return dl.status === 'downloading';
    });
    var existingBadge = navPill.querySelector('.nav-pill-badge');
    if (active.length === 0) {
      if (existingBadge) existingBadge.remove();
      return;
    }
    if (existingBadge) {
      existingBadge.textContent = active.length;
    } else {
      var badge = document.createElement('span');
      badge.className = 'nav-pill-badge';
      badge.textContent = active.length;
      navPill.appendChild(badge);
    }
  },

  updateDownloadProgress(hfRepo) {
    var dl = (this.state.activeDownloads || {})[hfRepo];
    if (!dl) return;
    // Find all download cards for this model (regardless of which tab user is on)
    document.querySelectorAll('[data-model-id="' + CSS.escape(hfRepo) + '"]').forEach(function (card) {
      var actions = card.querySelector('.download-card-actions');
      if (!actions) return;
      var mins = Math.floor(dl.elapsed / 60);
      var secs = dl.elapsed % 60;
      var timeStr = mins > 0 ? mins + 'm ' + secs + 's' : secs + 's';
      var label = dl.status === 'failed' ? '⚠ Failed' :
                  dl.status === 'completed' ? '✓ Downloaded' :
                  'Downloading ' + timeStr;
      actions.innerHTML =
        '<div class="download-progress-block">' +
          '<div class="download-progress-label">' + label + '</div>' +
          (dl.status === 'downloading' ?
            '<div class="download-progress-bar"><div class="download-progress-fill"></div></div>' : '') +
        '</div>';
    });
  },

  renderCapabilityBadges(model) {
    var caps = model.capabilities || [];
    var defs = {
      vision:       { label: 'Vision',       icon: '👁',  cls: 'cap-vision' },
      tool_use:     { label: 'Tool Use',     icon: '🔧', cls: 'cap-tool' },
      reasoning:    { label: 'Reasoning',    icon: '🧠', cls: 'cap-reasoning' },
      code:         { label: 'Code',         icon: '❮❯', cls: 'cap-code' },
      multilingual: { label: 'Multilingual', icon: '🌐', cls: 'cap-multilingual' },
    };
    var self = this;
    var repo = model.hf_repo || model.id;
    return caps.map(function (c) {
      var d = defs[c];
      if (!d) return '';
      return '<span class="cap-badge ' + d.cls + '" data-info-repo="' + self.esc(repo) + '" title="Click for details">' +
        '<span class="cap-badge-icon">' + d.icon + '</span>' + self.esc(d.label) +
      '</span>';
    }).join('');
  },

  showModelDetail(repoOrId) {
    var self = this;
    // Look up the model from our active source (catalog first, then live-catalog buffer)
    var pool = (this.state.catalog || []).slice();
    if (this.state.liveCatalogBuffer) pool = pool.concat(this.state.liveCatalogBuffer);
    var model = pool.find(function (m) {
      return m.hf_repo === repoOrId || m.id === repoOrId;
    });
    if (!model) {
      // Fetch details from HF search as a fallback
      fetch('/api/models/search?q=' + encodeURIComponent(repoOrId.split('/').pop() || repoOrId) + '&limit=5')
        .then(function (r) { return r.json(); })
        .then(function (data) {
          var hit = (data.models || []).find(function (m) { return m.hf_repo === repoOrId; });
          if (hit) self.renderModelDetailModal(hit);
          else self.toast('Model details unavailable', 'error');
        }).catch(function () { self.toast('Failed to fetch details', 'error'); });
      return;
    }
    this.renderModelDetailModal(model);
  },

  renderModelDetailModal(m) {
    var self = this;
    var existing = document.getElementById('model-detail-modal');
    if (existing) existing.remove();

    var repo = m.hf_repo || m.id;
    var name = m.name || repo;
    var isLoaded = ((this.state.status && this.state.status.models_loaded) || []).includes(repo);
    var capDefs = {
      vision:       { label: 'Vision',       icon: '👁',  cls: 'cap-vision' },
      tool_use:     { label: 'Tool Use',     icon: '🔧', cls: 'cap-tool' },
      reasoning:    { label: 'Reasoning',    icon: '🧠', cls: 'cap-reasoning' },
      code:         { label: 'Code',         icon: '❮❯', cls: 'cap-code' },
      multilingual: { label: 'Multilingual', icon: '🌐', cls: 'cap-multilingual' },
    };
    var caps = (m.capabilities || []).map(function (c) {
      var d = capDefs[c]; if (!d) return '';
      return '<span class="cap-badge large ' + d.cls + '"><span class="cap-badge-icon">' + d.icon + '</span>' + d.label + '</span>';
    }).join('');

    var sizeStr = m.size_gb ? '~' + Math.round(m.size_gb) + ' GB' : (m.size || 'size unknown');
    var downloadsStr = m.downloads ? self.formatNumber(m.downloads) : '—';
    var likesStr = m.likes ? self.formatNumber(m.likes) : '—';
    var ageStr = m.created_at ? self.relativeTime(m.created_at).replace('released ', '') : '—';
    var license = m.license || '—';
    var arch = (m.architecture || m.family || '—');
    var fmt = m.format || (m.quantization ? m.quantization.toUpperCase() : 'SafeTensors');
    var paramsStr = m.params_b ? m.params_b + 'B' : (m.params || '—');
    var ctxStr = m.context_length ? self.formatNumber(m.context_length) + ' tokens' : '—';

    var metaRow =
      '<div class="md-meta-row">' +
        '<div class="md-meta-pair"><span class="md-meta-label">Params</span><span class="md-meta-value">' + self.esc(paramsStr) + '</span></div>' +
        '<div class="md-meta-pair"><span class="md-meta-label">Arch</span><span class="md-meta-value">' + self.esc(arch) + '</span></div>' +
        '<div class="md-meta-pair"><span class="md-meta-label">Format</span><span class="md-meta-value md-format">' + self.esc(fmt) + '</span></div>' +
        '<div class="md-meta-pair"><span class="md-meta-label">Context</span><span class="md-meta-value">' + self.esc(ctxStr) + '</span></div>' +
        '<div class="md-meta-pair"><span class="md-meta-label">License</span><span class="md-meta-value">' + self.esc(license) + '</span></div>' +
      '</div>';

    var primaryCta = isLoaded
      ? '<button class="btn-nvidia md-cta" id="md-use-chat">▶ Use in New Chat</button>'
      : '<button class="btn-nvidia md-cta" id="md-download" data-hf-repo="' + self.esc(repo) + '">▼ Download (' + sizeStr + ')</button>';

    var modal = document.createElement('div');
    modal.id = 'model-detail-modal';
    modal.className = 'model-detail-modal-overlay';
    modal.innerHTML =
      '<div class="model-detail-modal">' +
        '<div class="md-header">' +
          '<div class="md-header-left">' +
            '<div class="md-icon">⬡</div>' +
            '<div class="md-title">' + self.esc(repo) + '</div>' +
            '<button class="md-copy" data-copy="' + self.esc(repo) + '" title="Copy repo">⧉</button>' +
          '</div>' +
          '<button class="md-close" title="Close">×</button>' +
        '</div>' +
        '<div class="md-stats-row">' +
          '<div class="md-stat"><span class="md-stat-icon">⬇</span>' + downloadsStr + '</div>' +
          '<div class="md-stat"><span class="md-stat-icon">★</span>' + likesStr + '</div>' +
          '<div class="md-stat md-stat-age">Last updated: ' + self.esc(ageStr) + '</div>' +
          (m.recommended ? '<div class="md-staff-pick">✨ Recommended</div>' : '') +
        '</div>' +
        (m.description || m.desc ? '<div class="md-description">' + self.esc(m.description || m.desc) + '</div>' : '') +
        metaRow +
        (caps ? '<div class="md-capabilities"><span class="md-section-label">Capabilities</span><div class="md-cap-list">' + caps + '</div></div>' : '') +
        '<div class="md-footer">' +
          '<div class="md-footer-status">' + (isLoaded ? '● Model loaded and ready' : '○ Not yet downloaded') + '</div>' +
          primaryCta +
        '</div>' +
      '</div>';

    document.body.appendChild(modal);

    // Close handlers
    var close = function () { modal.remove(); };
    modal.querySelector('.md-close').addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    document.addEventListener('keydown', function escHandler(e) {
      if (e.key === 'Escape') { close(); document.removeEventListener('keydown', escHandler); }
    });

    // Copy repo
    modal.querySelector('.md-copy').addEventListener('click', function (e) {
      navigator.clipboard.writeText(e.currentTarget.dataset.copy);
      self.toast('Repo copied', 'success');
    });

    // Download
    var dlBtn = modal.querySelector('#md-download');
    if (dlBtn) {
      dlBtn.addEventListener('click', function () {
        self.startRepoDownload(dlBtn.dataset.hfRepo);
        close();
      });
    }

    // Use in chat
    var chatBtn = modal.querySelector('#md-use-chat');
    if (chatBtn) {
      chatBtn.addEventListener('click', function () {
        self.state.messages = [];
        self.state.currentConversation = null;
        self.navigate('chat');
        var sel = document.getElementById('chat-model');
        if (sel) {
          for (var i = 0; i < sel.options.length; i++) {
            if (sel.options[i].value === repo) { sel.value = repo; break; }
          }
        }
        close();
      });
    }
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
        '<div id="downloads-queue" class="downloads-queue"></div>' +
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
          var detailsBtn = '<button class="btn-sm download-details-btn" data-info-repo="' + self.esc(m.hf_repo) + '">Details</button>';
          var downloadBtn = !isLoaded ?
            '<button class="btn-sm downloads-download-btn" data-model-id="' + self.esc(m.hf_repo) + '">Download</button>' : '';
          return '<div class="download-card" data-model-id="' + self.esc(m.hf_repo) + '">' +
            '<div class="download-card-main">' +
            '<div class="download-card-info">' +
            '<div class="download-card-header">' +
            '<div class="download-card-name">' + self.esc(m.name) + '</div>' +
            '<div class="download-card-badges">' + catalogBadge + fitBadge + statusBadge + '</div>' +
            '</div>' +
            '<div class="download-card-repo">' + self.esc(m.hf_repo) + '</div>' +
            '<div class="download-card-desc">' + sizeStr + (statsLine ? ' &middot; ' + statsLine : '') + '</div>' +
            '</div>' +
            '<div class="download-card-actions">' + detailsBtn + downloadBtn + '</div>' +
            '</div>' +
            '</div>';
        }).join('');
        if (resultsContainer) {
          resultsContainer.innerHTML = '<div class="downloads-grid">' + rows + '</div>';
          self.bindRepoDownloadButtons(resultsContainer);
        }
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
            capabilities: m.capabilities || [],
            architecture: m.architecture || '',
            format: m.format || '',
            hf_repo: m.hf_repo || '',
            context_length: m.context_length || 0,
            license: m.license || '',
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
        '<div id="downloads-queue" class="downloads-queue"></div>' +
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
      var capabilityBadges = self.renderCapabilityBadges(model);
      var isDownloaded = model.downloaded === true;
      var actionBtn = isDownloaded
        ? '<button class="btn-sm downloads-delete-btn" data-model-id="' + self.esc(model.hf_repo || model.id) + '">Delete</button>'
        : '<button class="btn-sm downloads-download-btn" data-model-id="' + self.esc(model.hf_repo || model.id) + '">Download</button>';
      var shardBtn = '';
      if (!fits && clusterNodeCount > 1 && totalClusterMem >= model.sizeGb && !isDownloaded) {
        shardBtn = '<button class="btn-sm downloads-shard-btn" data-model-id="' + self.esc(model.id) + '">Shard Across Cluster</button>';
      }
      var detailsBtn = '<button class="btn-sm download-details-btn" data-info-repo="' + self.esc(model.hf_repo || model.id) + '">Details</button>';
      return '<div class="download-card" data-model-id="' + self.esc(model.hf_repo || model.id) + '">' +
        '<div class="download-card-main">' +
        '<div class="download-card-info">' +
        '<div class="download-card-header">' +
        '<div class="download-card-name">' + self.esc(model.name || model.id) + '</div>' +
        '<div class="download-card-badges">' + recBadge + quantBadge + capabilityBadges + fitBadge + statusBadge + '</div>' +
        '</div>' +
        '<div class="download-card-repo">' + self.esc(model.id) + '</div>' +
        '<div class="download-card-desc">' + descParts.join(' &middot; ') + (model.desc ? '<br><span class="download-card-tagline">' + self.esc(model.desc) + '</span>' : '') + '</div>' +
        '</div>' +
        '<div class="download-card-actions">' + detailsBtn + actionBtn + shardBtn + '</div>' +
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

    // Bind download buttons — unified to repo-based download with progress view
    self.bindRepoDownloadButtons(container);

    // Render active downloads queue
    self.renderDownloadsQueue();

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
  //  TRAINING VIEW  (overview / datasets / runs / templates / benchmarks)
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

  // ---- Sidebar ------------------------------------------------------------
  async renderTrainingSidebar() {
    var mount = document.getElementById('left-panel-training');
    if (!mount) return;
    var self = this;

    var jobs = this.state.trainingJobs || [];
    // Pull fresh jobs if we don't have them yet
    if (!jobs.length) {
      var data = await this.fetchJSON('/api/training/jobs');
      this.state.trainingJobs = (data && data.jobs) || [];
      jobs = this.state.trainingJobs;
    }

    var tab = this.state.trainingTab || 'overview';
    var items = [
      { id: 'overview',   label: 'Overview',   icon: '&#9711;' },
      { id: 'datasets',   label: 'Datasets',   icon: '&#8864;' },
      { id: 'runs',       label: 'Runs',       icon: '&#9873;' },
      { id: 'templates',  label: 'Templates',  icon: '&#9641;' },
      { id: 'benchmarks', label: 'Benchmarks', icon: '&#9775;' },
    ];
    var guides = [
      { id: 'beginners',    label: "Beginner's Guide" },
      { id: 'distributed',  label: 'Distributed Training' },
      { id: 'pipeline',     label: 'Training Pipeline' },
      { id: 'standalone',   label: 'Standalone Training' },
    ];

    var recent = jobs.slice().sort(function (a, b) {
      return (b.start_time || 0) - (a.start_time || 0);
    }).slice(0, 5);

    var subnavHtml = items.map(function (it) {
      var active = it.id === tab ? ' active' : '';
      return '<button class="training-subnav-item' + active + '" data-tab="' + it.id + '">' +
        '<span class="sub-icon">' + it.icon + '</span>' + self.esc(it.label) + '</button>';
    }).join('');

    var recentHtml = recent.length
      ? recent.map(function (j) {
          var s = self._statusInfo(j.status);
          var model = (j.config && j.config.base_model ? j.config.base_model.split('/').pop() : '—');
          var name = (j.config && j.config.run_name) || model;
          return '<div class="training-recent-item" data-job-id="' + self.esc(j.job_id) + '">' +
            '<div class="training-recent-top">' +
              '<span class="training-recent-name">' + self.esc(name) + '</span>' +
              '<span class="job-status-badge ' + s.cls + '">' + s.label + '</span>' +
            '</div>' +
            '<div class="training-recent-meta">' + self.esc(j.job_id) + '</div>' +
            '</div>';
        }).join('')
      : '<div class="conv-empty" style="padding:10px 14px;font-size:12px;color:var(--text-muted)">No runs yet</div>';

    var guidesHtml = guides.map(function (g) {
      return '<a class="training-guide-link" data-guide="' + g.id + '">&#9733; ' + self.esc(g.label) + '</a>';
    }).join('');

    mount.innerHTML =
      '<button class="btn-nvidia" id="training-new-run-btn">+ NEW RUN</button>' +
      '<div class="training-sidebar-title">Training</div>' +
      '<div class="training-sidebar-section training-subnav">' + subnavHtml + '</div>' +
      '<div class="training-sidebar-section">' +
        '<div class="training-sidebar-title" style="padding:0 8px 8px">Recent Runs</div>' +
        recentHtml +
      '</div>' +
      '<div class="training-sidebar-section">' +
        '<div class="training-sidebar-title" style="padding:0 8px 8px">Guides</div>' +
        guidesHtml +
      '</div>';

    // Bindings
    var newBtn = document.getElementById('training-new-run-btn');
    if (newBtn) newBtn.addEventListener('click', function () { self.showNewRunWizard(); });

    mount.querySelectorAll('.training-subnav-item').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.state.trainingTab = btn.dataset.tab;
        self.state.trainingView = 'list';
        self.state.trainingDetailId = null;
        self.renderTrainingSidebar();
        self.renderTraining();
      });
    });
    mount.querySelectorAll('.training-recent-item').forEach(function (el) {
      el.addEventListener('click', function () {
        self.state.trainingTab = 'runs';
        self.state.trainingView = 'detail';
        self.state.trainingDetailId = el.dataset.jobId;
        self.state.trainingLossData = [];
        self.renderTrainingSidebar();
        self.renderTraining();
      });
    });
    mount.querySelectorAll('.training-guide-link').forEach(function (el) {
      el.addEventListener('click', function () {
        self.showGuideModal(el.dataset.guide);
      });
    });
  },

  _statusInfo(status) {
    var m = {
      pending:   { cls: 'status-pending',   label: 'Pending' },
      running:   { cls: 'status-running',   label: 'Running' },
      completed: { cls: 'status-completed', label: 'Completed' },
      failed:    { cls: 'status-failed',    label: 'Failed' },
      cancelled: { cls: 'status-cancelled', label: 'Cancelled' },
    };
    return m[status] || { cls: '', label: status || '—' };
  },

  async renderTraining() {
    var container = document.getElementById('training-content');
    if (!container) return;

    // If we're drilling into a specific job detail, handle that first
    if (this.state.trainingView === 'detail' && this.state.trainingDetailId) {
      var data = await this.fetchJSON('/api/training/jobs');
      this.state.trainingJobs = (data && data.jobs) || [];
      await this.renderTrainingDetail(container);
      return;
    }

    var tab = this.state.trainingTab || 'overview';
    // Fetch jobs + datasets + stats in parallel
    var results = await Promise.all([
      this.fetchJSON('/api/training/jobs'),
      this.fetchJSON('/api/training/stats'),
      this.fetchJSON('/api/datasets'),
      this.fetchJSON('/api/training/templates'),
    ]);
    this.state.trainingJobs = (results[0] && results[0].jobs) || [];
    this.state.trainingStats = results[1] || null;
    this.state.datasets = (results[2] && results[2].datasets) || [];
    this.state.trainingTemplates = (results[3] && results[3].templates) || [];

    // Refresh sidebar so Recent Runs shows latest
    this.renderTrainingSidebar();

    switch (tab) {
      case 'overview':   this.renderTrainingOverview(container); break;
      case 'datasets':   this.renderTrainingDatasets(container); break;
      case 'runs':       this.renderTrainingRuns(container); break;
      case 'templates':  this.renderTrainingTemplates(container); break;
      case 'benchmarks': this.renderTrainingBenchmarks(container); break;
      default:           this.renderTrainingOverview(container);
    }
  },

  // ---- Overview ----------------------------------------------------------
  renderTrainingOverview(container) {
    var self = this;
    var stats = this.state.trainingStats || {};
    var jobs = this.state.trainingJobs || [];
    var recent = jobs.slice().sort(function (a, b) {
      return (b.start_time || 0) - (a.start_time || 0);
    }).slice(0, 4);

    var tilesHtml =
      '<div class="training-stat-tile accent"><div class="stat-label">Active Runs</div><div class="stat-value">' + (stats.running || 0) + '</div></div>' +
      '<div class="training-stat-tile"><div class="stat-label">Completed Today</div><div class="stat-value">' + (stats.completed_today || 0) + '</div></div>' +
      '<div class="training-stat-tile"><div class="stat-label">Total Runs</div><div class="stat-value">' + (stats.total || jobs.length) + '</div></div>' +
      '<div class="training-stat-tile"><div class="stat-label">GPU Hours</div><div class="stat-value">' + (stats.total_gpu_hours != null ? stats.total_gpu_hours.toFixed(1) : '0.0') + '</div></div>';

    var quickstart = [
      { id: 'lora', icon: 'L', title: 'Fine-tune with LoRA', desc: 'Small adapter weights. Recommended for most users — trains fast, stays memory-efficient.' },
      { id: 'distributed', icon: 'D', title: 'Distributed Training', desc: 'Scale across multiple DGX Spark nodes with DDP + NCCL. Great for large datasets.' },
      { id: 'full', icon: 'F', title: 'Full Fine-tune', desc: 'Updates all weights. Highest quality, needs the most memory — use for single large-memory nodes.' },
    ].map(function (q) {
      return '<div class="quickstart-tile" data-qs="' + q.id + '">' +
        '<div class="quickstart-icon">' + q.icon + '</div>' +
        '<h4>' + self.esc(q.title) + '</h4>' +
        '<p>' + self.esc(q.desc) + '</p>' +
        '<button class="btn-nvidia" data-qs-btn="' + q.id + '">Start &rsaquo;</button>' +
        '</div>';
    }).join('');

    var recentHtml = recent.length
      ? recent.map(function (j) { return self.renderJobCard(j); }).join('')
      : '<div class="training-empty"><h3>No runs yet</h3><p>Kick off your first fine-tune from the quick-start tiles above.</p></div>';

    container.innerHTML =
      '<div class="training-overview">' +
        '<div class="training-stat-tiles">' + tilesHtml + '</div>' +
        '<div>' +
          '<div class="training-section-head"><h3>Quick Start</h3><span class="muted">Jump straight in</span></div>' +
          '<div class="quickstart-grid">' + quickstart + '</div>' +
        '</div>' +
        '<div>' +
          '<div class="training-section-head"><h3>Recent Activity</h3><span class="muted">' + recent.length + ' run' + (recent.length === 1 ? '' : 's') + '</span></div>' +
          '<div class="recent-activity-list">' + recentHtml + '</div>' +
        '</div>' +
      '</div>';

    container.querySelectorAll('[data-qs-btn]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.showNewRunWizard({ method: btn.dataset.qsBtn === 'full' ? 'full' : 'lora', distributed: btn.dataset.qsBtn === 'distributed' });
      });
    });
    container.querySelectorAll('.training-job-card').forEach(function (card) {
      card.addEventListener('click', function () {
        self.state.trainingTab = 'runs';
        self.state.trainingView = 'detail';
        self.state.trainingDetailId = card.dataset.jobId;
        self.state.trainingLossData = [];
        self.renderTrainingSidebar();
        self.renderTraining();
      });
    });
  },

  // ---- Datasets ----------------------------------------------------------
  renderTrainingDatasets(container) {
    var self = this;
    var datasets = this.state.datasets || [];
    var list = datasets.length
      ? '<div class="dataset-grid">' + datasets.map(function (d) { return self.renderDatasetCard(d); }).join('') + '</div>'
      : '<div class="training-empty"><h3>No datasets yet</h3><p>Upload a file, reference a HuggingFace dataset, or point at a local path to get started.</p></div>';

    container.innerHTML =
      '<div class="training-overview">' +
        '<div class="dataset-toolbar">' +
          '<div><h2 style="font-size:18px;font-weight:700">Datasets</h2>' +
          '<p style="font-size:12.5px;color:var(--text-muted);margin-top:4px">Training data you can attach to any run.</p></div>' +
          '<button class="btn-nvidia" id="dataset-add-btn">+ Add Dataset</button>' +
        '</div>' +
        list +
      '</div>';

    var addBtn = document.getElementById('dataset-add-btn');
    if (addBtn) addBtn.addEventListener('click', function () { self.showDatasetAddModal(); });

    container.querySelectorAll('[data-ds-delete]').forEach(function (b) {
      b.addEventListener('click', async function (e) {
        e.stopPropagation();
        if (!confirm('Delete this dataset?')) return;
        var resp = await fetch('/api/datasets/' + encodeURIComponent(b.dataset.dsDelete), { method: 'DELETE' });
        if (resp.ok) { self.toast('Dataset deleted', 'success'); self.renderTraining(); }
        else { self.toast('Failed to delete dataset', 'error'); }
      });
    });
    container.querySelectorAll('[data-ds-use]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        self.showNewRunWizard({ dataset_id: b.dataset.dsUse });
      });
    });
    container.querySelectorAll('[data-ds-preview]').forEach(function (b) {
      b.addEventListener('click', async function (e) {
        e.stopPropagation();
        var id = b.dataset.dsPreview;
        var target = document.getElementById('ds-preview-' + id);
        if (!target) return;
        if (target.style.display !== 'none' && target.dataset.loaded === '1') {
          target.style.display = 'none';
          target.dataset.loaded = '0';
          return;
        }
        target.style.display = '';
        target.textContent = 'Loading preview…';
        var data = await self.fetchJSON('/api/datasets/' + encodeURIComponent(id) + '/preview');
        if (!data) { target.textContent = 'Preview unavailable.'; return; }
        target.textContent = JSON.stringify(data.samples, null, 2);
        target.dataset.loaded = '1';
      });
    });
  },

  renderDatasetCard(d) {
    var name = this.esc(d.name || d.id);
    var badge = '<span class="dataset-badge src-' + this.esc(d.source) + '">' + this.esc(d.source) + '</span>';
    var fmt = d.format ? '<span class="dataset-badge">' + this.esc(d.format) + '</span>' : '';
    var sz = this.formatBytes(d.size_bytes || 0);
    var samples = (d.samples || 0).toLocaleString();
    var created = d.created_at ? new Date(d.created_at * 1000).toLocaleDateString() : '—';

    return '<div class="dataset-card">' +
      '<div class="dataset-card-top">' +
        '<div><div class="dataset-card-name">' + name + '</div>' +
        '<div class="dataset-card-id">' + this.esc(d.id) + '</div></div>' +
        '<div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end">' + badge + fmt + '</div>' +
      '</div>' +
      '<div class="dataset-card-stats">' +
        '<span>' + samples + ' samples</span>' +
        '<span>' + sz + '</span>' +
        '<span>' + this.esc(created) + '</span>' +
      '</div>' +
      '<div class="dataset-card-path" title="' + this.esc(d.path || '') + '">' + this.esc(d.path || '') + '</div>' +
      '<div class="dataset-card-actions">' +
        '<button class="btn-nvidia btn-sm" data-ds-use="' + this.esc(d.id) + '">Use in Run</button>' +
        '<button class="btn-ghost btn-sm" data-ds-preview="' + this.esc(d.id) + '">Preview</button>' +
        '<button class="btn-danger btn-sm" data-ds-delete="' + this.esc(d.id) + '">Delete</button>' +
      '</div>' +
      '<div class="dataset-preview-samples" id="ds-preview-' + this.esc(d.id) + '" style="display:none" data-loaded="0"></div>' +
    '</div>';
  },

  // ---- Runs --------------------------------------------------------------
  renderTrainingRuns(container) {
    var self = this;
    var jobs = this.state.trainingJobs || [];
    var filter = this.state.runsFilter || 'all';
    if (filter !== 'all') jobs = jobs.filter(function (j) { return j.status === filter; });
    jobs = jobs.slice().sort(function (a, b) { return (b.start_time || 0) - (a.start_time || 0); });

    var filters = ['all', 'running', 'completed', 'failed'];
    var filterBar = '<div class="runs-filter-bar">' + filters.map(function (f) {
      var a = f === self.state.runsFilter ? ' active' : '';
      return '<button class="pill' + a + '" data-filter="' + f + '">' + f.charAt(0).toUpperCase() + f.slice(1) + '</button>';
    }).join('') + '</div>';

    var body;
    if (jobs.length === 0) {
      body = '<div class="training-empty"><h3>No runs found</h3><p>Try a different filter or start a new run from the sidebar.</p></div>';
    } else {
      var rows = jobs.map(function (j) {
        var s = self._statusInfo(j.status);
        var cfg = j.config || {};
        var modelShort = (cfg.base_model || '').split('/').pop() || '—';
        var runName = cfg.run_name || modelShort;
        var started = j.start_time ? new Date(j.start_time * 1000).toLocaleString() : '—';
        var elapsed = j.elapsed_seconds ? self.formatUptime(Math.round(j.elapsed_seconds)) : '—';
        var method = (cfg.method || 'lora').toUpperCase();
        return '<tr class="run-row" data-job-id="' + self.esc(j.job_id) + '">' +
          '<td><strong>' + self.esc(runName) + '</strong><div class="mono" style="color:var(--text-muted);font-size:10.5px">' + self.esc(j.job_id) + '</div></td>' +
          '<td class="mono">' + self.esc(modelShort) + '</td>' +
          '<td>' + method + '</td>' +
          '<td><span class="job-status-badge ' + s.cls + '">' + s.label + '</span></td>' +
          '<td>' + self.esc(started) + '</td>' +
          '<td class="mono">' + self.esc(elapsed) + '</td>' +
          '<td class="mono">' + (j.current_loss != null ? j.current_loss.toFixed(4) : '—') + '</td>' +
          '</tr>';
      }).join('');

      body = '<table class="runs-table">' +
        '<thead><tr><th>Run</th><th>Model</th><th>Method</th><th>Status</th><th>Started</th><th>Duration</th><th>Loss</th></tr></thead>' +
        '<tbody>' + rows + '</tbody>' +
        '</table>';
    }

    container.innerHTML = '<div class="training-overview">' +
      '<div class="training-section-head"><h3>Runs</h3><span class="muted">' + (this.state.trainingJobs || []).length + ' total</span></div>' +
      filterBar + body +
    '</div>';

    container.querySelectorAll('[data-filter]').forEach(function (p) {
      p.addEventListener('click', function () {
        self.state.runsFilter = p.dataset.filter;
        self.renderTrainingRuns(container);
      });
    });
    container.querySelectorAll('.run-row').forEach(function (row) {
      row.addEventListener('click', function () {
        self.state.trainingView = 'detail';
        self.state.trainingDetailId = row.dataset.jobId;
        self.state.trainingLossData = [];
        self.renderTraining();
      });
    });
  },

  // ---- Templates ---------------------------------------------------------
  renderTrainingTemplates(container) {
    var self = this;
    var templates = this.state.trainingTemplates || [];
    var cards = templates.length ? templates.map(function (t) {
      var shape = typeof t.sample_shape === 'object'
        ? JSON.stringify(t.sample_shape, null, 2)
        : String(t.sample_shape || '');
      return '<div class="template-card" data-template-id="' + self.esc(t.id) + '">' +
        '<h4>' + self.esc(t.name) + '</h4>' +
        '<p>' + self.esc(t.description || '') + '</p>' +
        '<div class="template-shape">' + self.esc(shape) + '</div>' +
        '<div class="template-meta-row">' +
          '<span>&#9830; ' + self.esc((t.method || 'lora').toUpperCase()) + '</span>' +
          '<span>&#9200; ' + self.esc(t.estimated_time || '—') + '</span>' +
          (t.distributed ? '<span>&#9733; Distributed</span>' : '') +
        '</div>' +
        '<button class="btn-nvidia btn-sm" data-use-template="' + self.esc(t.id) + '">Use this template</button>' +
      '</div>';
    }).join('') : '<div class="training-empty"><h3>No templates available</h3></div>';

    container.innerHTML = '<div class="training-overview">' +
      '<div class="training-section-head"><h3>Templates</h3><span class="muted">Starter recipes</span></div>' +
      '<div class="templates-gallery">' + cards + '</div>' +
    '</div>';

    container.querySelectorAll('[data-use-template]').forEach(function (b) {
      b.addEventListener('click', function () {
        var tpl = (self.state.trainingTemplates || []).find(function (t) { return t.id === b.dataset.useTemplate; });
        if (!tpl) return;
        self.showNewRunWizard({
          template_id: tpl.id,
          method: tpl.method || 'lora',
          distributed: !!tpl.distributed,
          num_epochs: tpl.recommended_epochs,
          batch_size: tpl.recommended_batch_size,
          learning_rate: tpl.recommended_lr,
        });
      });
    });
  },

  // ---- Benchmarks --------------------------------------------------------
  renderTrainingBenchmarks(container) {
    var self = this;
    // This is a placeholder view — wires to /api/benchmarks in a future pass.
    container.innerHTML = '<div class="training-overview">' +
      '<div class="training-section-head"><h3>Benchmarks</h3><span class="muted">NCCL / RDMA / GPU-direct</span></div>' +
      '<div class="benchmark-grid">' +
        '<div class="benchmark-card"><h4>NCCL all-reduce</h4><div class="benchmark-value">—</div><div class="benchmark-meta">not yet run</div><button class="btn-nvidia btn-sm" data-bench="nccl">Run</button></div>' +
        '<div class="benchmark-card"><h4>GPU-direct RDMA</h4><div class="benchmark-value">—</div><div class="benchmark-meta">not yet run</div><button class="btn-nvidia btn-sm" data-bench="rdma">Run</button></div>' +
        '<div class="benchmark-card"><h4>Storage Throughput</h4><div class="benchmark-value">—</div><div class="benchmark-meta">not yet run</div><button class="btn-nvidia btn-sm" data-bench="storage">Run</button></div>' +
      '</div>' +
    '</div>';

    container.querySelectorAll('[data-bench]').forEach(function (b) {
      b.addEventListener('click', function () {
        self.toast('Benchmark runners coming in the next release', 'info');
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

  // =======================================================================
  //  NEW RUN WIZARD (modal, multi-step)
  // =======================================================================

  async showNewRunWizard(prefill) {
    var self = this;
    // Gather fresh data
    var dsets = this.state.datasets.length
      ? this.state.datasets
      : (((await this.fetchJSON('/api/datasets')) || {}).datasets || []);
    this.state.datasets = dsets;

    // Try to get downloaded models from /api/models; fall back to built-in list
    var modelsResp = await this.fetchJSON('/api/models');
    var modelList = [];
    if (modelsResp && Array.isArray(modelsResp.models)) {
      modelList = modelsResp.models
        .filter(function (m) { return m.downloaded || m.is_downloaded || m.status === 'downloaded'; })
        .map(function (m) { return { id: m.id || m.name || m.repo_id, name: m.name || m.id, size: m.size || '' }; });
    }
    if (!modelList.length) modelList = this.trainingModels.slice();

    var initial = Object.assign({
      base_model: modelList[0] ? modelList[0].id : '',
      dataset_id: '',
      method: 'lora',
      num_epochs: 3,
      batch_size: 4,
      learning_rate: 2e-4,
      max_seq_length: 2048,
      lora_rank: 16,
      lora_alpha: 32,
      gradient_accumulation_steps: 1,
      warmup_steps: 0,
      weight_decay: 0.0,
      use_gradient_checkpointing: false,
      distributed: false,
      num_nodes: 1,
      run_name: '',
      template_id: null,
      step: 1,
      models: modelList,
      datasets: dsets,
    }, prefill || {});
    this.state.wizardState = initial;

    // Build overlay
    var existing = document.querySelector('.model-detail-modal-overlay.wizard-overlay');
    if (existing) existing.remove();

    var overlay = document.createElement('div');
    overlay.className = 'model-detail-modal-overlay wizard-overlay';
    overlay.innerHTML =
      '<div class="model-detail-modal wizard-modal">' +
        '<div class="md-header">' +
          '<div class="md-header-left"><div class="md-icon">&#9881;</div><div><div style="font-weight:700;font-size:15px">New Training Run</div><div style="font-size:11.5px;color:var(--text-muted)">Configure and launch a fine-tune</div></div></div>' +
          '<button class="btn-ghost btn-sm" id="wizard-close">Close</button>' +
        '</div>' +
        '<div class="wizard-body">' +
          '<div class="wizard-steps" id="wizard-steps"></div>' +
          '<div class="wizard-panel" id="wizard-panel"></div>' +
        '</div>' +
        '<div class="wizard-footer">' +
          '<button class="btn-ghost" id="wizard-prev">&larr; Back</button>' +
          '<span class="estimate-pill" id="wizard-estimate"></span>' +
          '<div class="spacer"></div>' +
          '<button class="btn-nvidia" id="wizard-next">Next &rarr;</button>' +
        '</div>' +
      '</div>';

    document.body.appendChild(overlay);

    overlay.addEventListener('click', function (e) { if (e.target === overlay) self.closeWizard(); });
    document.getElementById('wizard-close').addEventListener('click', function () { self.closeWizard(); });
    document.getElementById('wizard-prev').addEventListener('click', function () { self.wizardGo(-1); });
    document.getElementById('wizard-next').addEventListener('click', function () { self.wizardGo(+1); });

    this.renderWizardStep();
  },

  closeWizard() {
    var o = document.querySelector('.wizard-overlay');
    if (o) o.remove();
    this.state.wizardState = null;
  },

  wizardGo(delta) {
    var s = this.state.wizardState;
    if (!s) return;
    var next = s.step + delta;
    if (next < 1) return;
    if (next > 5) { this.submitWizard(); return; }
    // Validate current step before moving forward
    if (delta > 0 && !this.validateWizardStep()) return;
    s.step = next;
    this.renderWizardStep();
  },

  validateWizardStep() {
    var s = this.state.wizardState;
    if (s.step === 1 && !s.base_model) { this.toast('Pick a base model to continue', 'error'); return false; }
    if (s.step === 2 && !s.dataset_id && !s.dataset_path_inline) { this.toast('Select or add a dataset', 'error'); return false; }
    return true;
  },

  renderWizardStep() {
    var s = this.state.wizardState;
    if (!s) return;
    var self = this;
    var stepDefs = [
      { n: 1, label: 'Base Model' },
      { n: 2, label: 'Dataset' },
      { n: 3, label: 'Method' },
      { n: 4, label: 'Distribution' },
      { n: 5, label: 'Review' },
    ];

    document.getElementById('wizard-steps').innerHTML = stepDefs.map(function (d) {
      var cls = d.n === s.step ? ' active' : (d.n < s.step ? ' done' : '');
      return '<div class="wizard-step-nav' + cls + '"><span class="step-num">' + d.n + '</span>' + d.label + '</div>';
    }).join('');

    var panel = document.getElementById('wizard-panel');
    var nextBtn = document.getElementById('wizard-next');
    if (s.step === 5) nextBtn.textContent = 'Launch Training Run';
    else nextBtn.textContent = 'Next →';
    document.getElementById('wizard-prev').style.visibility = s.step === 1 ? 'hidden' : '';

    switch (s.step) {
      case 1: panel.innerHTML = this._renderWizardStep1(); break;
      case 2: panel.innerHTML = this._renderWizardStep2(); break;
      case 3: panel.innerHTML = this._renderWizardStep3(); break;
      case 4: panel.innerHTML = this._renderWizardStep4(); break;
      case 5: panel.innerHTML = this._renderWizardStep5(); break;
    }
    this._bindWizardStep();
    this._updateEstimate();
  },

  _renderWizardStep1() {
    var s = this.state.wizardState;
    var opts = (s.models || []).map(function (m) {
      var active = m.id === s.base_model ? ' active' : '';
      return '<div class="wizard-option' + active + '" data-model="' + AINode.esc(m.id) + '">' +
        '<div class="wizard-option-title">' + AINode.esc(m.name || m.id) + '</div>' +
        '<div class="wizard-option-sub">' + AINode.esc(m.id) + (m.size ? ' · ' + AINode.esc(m.size) : '') + '</div>' +
        '</div>';
    }).join('');
    return '<h3>Choose a Base Model</h3>' +
      '<p class="panel-sub">Downloaded models are shown. Your fine-tune will build on top of this one.</p>' +
      '<div class="wizard-option-grid">' + opts + '</div>' +
      '<div class="form-group" style="margin-top:16px"><label class="form-label">Or enter a HuggingFace model ID</label>' +
      '<input type="text" id="wz-custom-model" class="form-input" placeholder="org/model" value="' + AINode.esc(s.custom_model || '') + '"></div>';
  },

  _renderWizardStep2() {
    var s = this.state.wizardState;
    var datasets = s.datasets || [];
    var options = datasets.map(function (d) {
      var active = d.id === s.dataset_id ? ' active' : '';
      return '<div class="wizard-option' + active + '" data-dataset="' + AINode.esc(d.id) + '">' +
        '<div class="wizard-option-title">' + AINode.esc(d.name) + '</div>' +
        '<div class="wizard-option-sub">' + (d.samples || 0).toLocaleString() + ' samples · ' + AINode.esc(d.source) + ' · ' + AINode.esc(d.format || '') + '</div>' +
        '</div>';
    }).join('');
    var body = datasets.length
      ? '<div class="wizard-option-grid">' + options + '</div>'
      : '<div class="training-empty" style="padding:24px"><h3>No datasets yet</h3><p>Add one below or via the Datasets tab.</p></div>';

    return '<h3>Select Training Dataset</h3>' +
      '<p class="panel-sub">Reference a saved dataset, or paste a path / HuggingFace ID inline.</p>' +
      body +
      '<div class="form-group" style="margin-top:16px">' +
        '<label class="form-label">Or inline dataset path</label>' +
        '<input type="text" id="wz-inline-ds" class="form-input" placeholder="~/.ainode/datasets/file.jsonl or alpaca.jsonl" value="' + AINode.esc(s.dataset_path_inline || '') + '">' +
        '<div class="form-hint">Fields accepted: <code>text</code>, or <code>instruction</code>+<code>output</code>, or <code>prompt</code>+<code>completion</code>.</div>' +
      '</div>';
  },

  _renderWizardStep3() {
    var s = this.state.wizardState;
    var methods = [
      { id: 'lora',  title: 'LoRA',      sub: 'Train small adapter weights. Fast, low memory.' },
      { id: 'qlora', title: 'QLoRA',     sub: '4-bit quantized LoRA. Fits huge models in limited memory.' },
      { id: 'full',  title: 'Full',      sub: 'Update all weights. Highest quality, highest memory.' },
    ];
    var methodCards = methods.map(function (m) {
      var active = m.id === s.method ? ' active' : '';
      return '<div class="wizard-option' + active + '" data-method="' + m.id + '">' +
        '<div class="wizard-option-title">' + m.title + '</div>' +
        '<div class="wizard-option-sub">' + m.sub + '</div>' +
        '</div>';
    }).join('');

    var loraBlock = (s.method === 'lora' || s.method === 'qlora') ?
      '<div class="form-grid">' +
        '<div class="form-group"><label class="form-label">LoRA Rank</label><input type="number" id="wz-lora-rank" class="form-input" value="' + s.lora_rank + '" min="1" max="256"></div>' +
        '<div class="form-group"><label class="form-label">LoRA Alpha</label><input type="number" id="wz-lora-alpha" class="form-input" value="' + s.lora_alpha + '" min="1" max="512"></div>' +
      '</div>' : '';

    return '<h3>Training Method</h3>' +
      '<p class="panel-sub">LoRA is recommended. Full fine-tune needs significant GPU memory.</p>' +
      '<div class="wizard-option-grid">' + methodCards + '</div>' +
      '<div style="margin-top:20px">' +
        '<div class="form-grid">' +
          '<div class="form-group"><label class="form-label">Epochs</label><input type="number" id="wz-epochs" class="form-input" value="' + s.num_epochs + '" min="1" max="100"></div>' +
          '<div class="form-group"><label class="form-label">Batch Size</label><input type="number" id="wz-batch" class="form-input" value="' + s.batch_size + '" min="1" max="128"></div>' +
          '<div class="form-group"><label class="form-label">Learning Rate</label><input type="text" id="wz-lr" class="form-input" value="' + s.learning_rate + '"></div>' +
          '<div class="form-group"><label class="form-label">Max Seq Length</label><input type="number" id="wz-seq" class="form-input" value="' + s.max_seq_length + '" min="128" step="128"></div>' +
          '<div class="form-group"><label class="form-label">Grad Accumulation</label><input type="number" id="wz-ga" class="form-input" value="' + s.gradient_accumulation_steps + '" min="1"></div>' +
          '<div class="form-group"><label class="form-label">Warmup Steps</label><input type="number" id="wz-warmup" class="form-input" value="' + s.warmup_steps + '" min="0"></div>' +
        '</div>' +
        loraBlock +
        '<div style="margin-top:12px"><button type="button" class="btn-ghost btn-sm" id="wz-smart">&#9733; Smart Defaults</button></div>' +
      '</div>';
  },

  _renderWizardStep4() {
    var s = this.state.wizardState;
    var nodeCount = (this.state.nodes || []).length || 1;
    var dropdownMax = Math.max(1, nodeCount);
    var nodesInput = '';
    for (var i = 1; i <= Math.max(3, dropdownMax); i++) {
      nodesInput += '<button type="button" class="node-dot' + (i === s.num_nodes ? ' active' : '') + '" data-nodes="' + i + '">' + i + '</button>';
    }
    return '<h3>Distribution</h3>' +
      '<p class="panel-sub">Single node is the default. Enable distributed DDP to shard across multiple cluster nodes.</p>' +
      '<div class="wizard-option-grid">' +
        '<div class="wizard-option' + (!s.distributed ? ' active' : '') + '" data-dist="0">' +
          '<div class="wizard-option-title">Single Node</div>' +
          '<div class="wizard-option-sub">Run on this node\'s GPU(s).</div>' +
        '</div>' +
        '<div class="wizard-option' + (s.distributed ? ' active' : '') + '" data-dist="1">' +
          '<div class="wizard-option-title">Multi-node DDP</div>' +
          '<div class="wizard-option-sub">Data-parallel across ' + nodeCount + ' online node' + (nodeCount === 1 ? '' : 's') + '.</div>' +
        '</div>' +
      '</div>' +
      (s.distributed ? '<div style="margin-top:16px"><label class="form-label">Number of nodes</label><div class="node-selector">' + nodesInput + '</div></div>' : '') +
      '<div class="form-group" style="margin-top:16px"><label class="form-label">Run name (optional)</label><input type="text" id="wz-run-name" class="form-input" placeholder="alpaca-3b-epoch3" value="' + AINode.esc(s.run_name || '') + '"></div>';
  },

  _renderWizardStep5() {
    var s = this.state.wizardState;
    var self = this;
    var summaryRow = function (k, v) { return '<div class="k">' + k + '</div><div class="v">' + AINode.esc(String(v == null ? '—' : v)) + '</div>'; };
    var dsname = '—';
    if (s.dataset_id) {
      var match = (s.datasets || []).find(function (d) { return d.id === s.dataset_id; });
      if (match) dsname = match.name + ' (' + match.id + ')';
    } else if (s.dataset_path_inline) {
      dsname = s.dataset_path_inline;
    }
    return '<h3>Review & Launch</h3>' +
      '<p class="panel-sub">Double-check the config, then launch. You can cancel any time from the run detail view.</p>' +
      '<div class="wizard-summary">' +
        summaryRow('Run name', s.run_name || 'auto') +
        summaryRow('Base model', s.base_model) +
        summaryRow('Dataset', dsname) +
        summaryRow('Method', (s.method || 'lora').toUpperCase()) +
        summaryRow('Epochs', s.num_epochs) +
        summaryRow('Batch size', s.batch_size) +
        summaryRow('Learning rate', s.learning_rate) +
        summaryRow('Max seq len', s.max_seq_length) +
        summaryRow('Grad accum', s.gradient_accumulation_steps) +
        summaryRow('Distributed', s.distributed ? (s.num_nodes + ' nodes') : 'single node') +
      '</div>' +
      '<div class="form-group" style="margin-top:16px"><label style="display:flex;align-items:center;gap:8px;color:var(--text-secondary);font-size:12.5px"><input type="checkbox" id="wz-save-template"> Save as a custom template</label></div>';
  },

  _bindWizardStep() {
    var self = this;
    var s = this.state.wizardState;
    // Step 1
    document.querySelectorAll('.wizard-option[data-model]').forEach(function (el) {
      el.addEventListener('click', function () {
        s.base_model = el.dataset.model;
        s.custom_model = '';
        self.renderWizardStep();
      });
    });
    var custom = document.getElementById('wz-custom-model');
    if (custom) custom.addEventListener('change', function () {
      s.custom_model = custom.value.trim();
      if (s.custom_model) s.base_model = s.custom_model;
    });

    // Step 2
    document.querySelectorAll('.wizard-option[data-dataset]').forEach(function (el) {
      el.addEventListener('click', function () {
        s.dataset_id = el.dataset.dataset;
        s.dataset_path_inline = '';
        self.renderWizardStep();
      });
    });
    var inlineDs = document.getElementById('wz-inline-ds');
    if (inlineDs) inlineDs.addEventListener('change', function () {
      s.dataset_path_inline = inlineDs.value.trim();
      if (s.dataset_path_inline) s.dataset_id = '';
    });

    // Step 3
    document.querySelectorAll('.wizard-option[data-method]').forEach(function (el) {
      el.addEventListener('click', function () {
        s.method = el.dataset.method;
        self.renderWizardStep();
      });
    });
    var hook = function (id, key, parser) {
      var el = document.getElementById(id);
      if (el) el.addEventListener('change', function () {
        var v = parser ? parser(el.value) : el.value;
        if (!isNaN(v) || typeof v === 'string') s[key] = v;
        self._updateEstimate();
      });
    };
    hook('wz-epochs', 'num_epochs', parseInt);
    hook('wz-batch', 'batch_size', parseInt);
    hook('wz-lr', 'learning_rate', parseFloat);
    hook('wz-seq', 'max_seq_length', parseInt);
    hook('wz-ga', 'gradient_accumulation_steps', parseInt);
    hook('wz-warmup', 'warmup_steps', parseInt);
    hook('wz-lora-rank', 'lora_rank', parseInt);
    hook('wz-lora-alpha', 'lora_alpha', parseInt);

    var smart = document.getElementById('wz-smart');
    if (smart) smart.addEventListener('click', function () {
      var model = (s.base_model || '').toLowerCase();
      var big = /70b|72b|405b/.test(model);
      s.batch_size = big ? 1 : 4;
      s.gradient_accumulation_steps = big ? 8 : 2;
      s.learning_rate = s.method === 'full' ? 5e-5 : 2e-4;
      s.warmup_steps = 50;
      s.max_seq_length = 2048;
      self.toast('Smart defaults applied', 'success');
      self.renderWizardStep();
    });

    // Step 4
    document.querySelectorAll('.wizard-option[data-dist]').forEach(function (el) {
      el.addEventListener('click', function () {
        s.distributed = el.dataset.dist === '1';
        if (!s.distributed) s.num_nodes = 1;
        self.renderWizardStep();
      });
    });
    document.querySelectorAll('.node-dot[data-nodes]').forEach(function (el) {
      el.addEventListener('click', function () {
        s.num_nodes = parseInt(el.dataset.nodes, 10);
        self.renderWizardStep();
      });
    });
    var runName = document.getElementById('wz-run-name');
    if (runName) runName.addEventListener('input', function () { s.run_name = runName.value; });
  },

  async _updateEstimate() {
    var s = this.state.wizardState;
    if (!s) return;
    var pill = document.getElementById('wizard-estimate');
    if (!pill) return;
    try {
      var sampleCount = 0;
      if (s.dataset_id) {
        var match = (s.datasets || []).find(function (d) { return d.id === s.dataset_id; });
        if (match) sampleCount = match.samples || 0;
      }
      var resp = await fetch('/api/training/estimate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_model: s.base_model,
          dataset_path: s.dataset_path_inline || 'placeholder',
          dataset_id: s.dataset_id,
          method: s.method,
          num_epochs: s.num_epochs,
          batch_size: s.batch_size,
          max_seq_length: s.max_seq_length,
          gradient_accumulation_steps: s.gradient_accumulation_steps,
          distributed: s.distributed,
          num_nodes: s.num_nodes,
          sample_count: sampleCount,
        }),
      });
      if (!resp.ok) { pill.textContent = ''; return; }
      var data = await resp.json();
      var parts = [];
      if (data.memory_gb_per_node) parts.push('&#9881; ~' + data.memory_gb_per_node.toFixed(0) + ' GB/node');
      if (data.estimated_seconds) parts.push('&#9201; ~' + this.formatUptime(Math.round(data.estimated_seconds)));
      if (data.samples_per_sec) parts.push('&rarr; ' + data.samples_per_sec.toFixed(1) + ' samp/s');
      pill.innerHTML = parts.join(' · ');
    } catch (e) {
      pill.textContent = '';
    }
  },

  async submitWizard() {
    var s = this.state.wizardState;
    if (!s) return;
    var payload = {
      base_model: s.base_model,
      method: s.method,
      num_epochs: s.num_epochs,
      batch_size: s.batch_size,
      learning_rate: parseFloat(s.learning_rate),
      max_seq_length: s.max_seq_length,
      gradient_accumulation_steps: s.gradient_accumulation_steps,
      warmup_steps: s.warmup_steps,
      weight_decay: s.weight_decay,
      distributed: s.distributed,
      num_nodes: s.num_nodes,
    };
    if (s.run_name) payload.run_name = s.run_name;
    if (s.template_id) payload.template_id = s.template_id;
    if (s.method === 'lora' || s.method === 'qlora') {
      payload.lora_rank = s.lora_rank;
      payload.lora_alpha = s.lora_alpha;
    }
    if (s.dataset_id) {
      payload.dataset_id = s.dataset_id;
      payload.dataset_path = 'pending';  // backend will overwrite via dataset_id resolution
    } else {
      payload.dataset_path = s.dataset_path_inline;
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
        return;
      }
      this.toast('Training run launched', 'success');
      this.closeWizard();
      this.state.trainingTab = 'runs';
      this.state.trainingView = 'detail';
      this.state.trainingDetailId = result.job_id;
      this.state.trainingLossData = [];
      this.renderTrainingSidebar();
      this.renderTraining();
    } catch (err) {
      this.toast('Network error: ' + err.message, 'error');
    }
  },

  // =======================================================================
  //  DATASET ADD MODAL (Upload / HF / Local / URL)
  // =======================================================================

  showDatasetAddModal() {
    var self = this;
    var existing = document.querySelector('.model-detail-modal-overlay.dataset-overlay');
    if (existing) existing.remove();
    var overlay = document.createElement('div');
    overlay.className = 'model-detail-modal-overlay dataset-overlay';
    overlay.innerHTML =
      '<div class="model-detail-modal" style="width:min(640px,94vw)">' +
        '<div class="md-header">' +
          '<div class="md-header-left"><div class="md-icon">&#8864;</div><div><div style="font-weight:700;font-size:15px">Add Dataset</div></div></div>' +
          '<button class="btn-ghost btn-sm" id="dsm-close">Close</button>' +
        '</div>' +
        '<div style="padding:20px 24px">' +
          '<div class="pill-group">' +
            '<button class="pill active" data-src="upload">Upload</button>' +
            '<button class="pill" data-src="huggingface">HuggingFace</button>' +
            '<button class="pill" data-src="local">Local Path</button>' +
            '<button class="pill" data-src="url">URL</button>' +
          '</div>' +
          '<div id="dsm-panel"></div>' +
        '</div>' +
        '<div class="wizard-footer"><div class="spacer"></div><button class="btn-nvidia" id="dsm-submit">Add Dataset</button></div>' +
      '</div>';
    document.body.appendChild(overlay);
    overlay.addEventListener('click', function (e) { if (e.target === overlay) overlay.remove(); });
    document.getElementById('dsm-close').addEventListener('click', function () { overlay.remove(); });

    var currentSrc = 'upload';
    var renderPanel = function () {
      var p = document.getElementById('dsm-panel');
      var nameField = '<div class="form-group"><label class="form-label">Name (optional)</label><input type="text" id="dsm-name" class="form-input" placeholder="my-dataset"></div>';
      if (currentSrc === 'upload') {
        p.innerHTML = nameField +
          '<div class="form-group"><label class="form-label">File</label>' +
          '<div class="upload-drop-zone" id="dsm-drop"><strong>Drop file here</strong> or click to select<br><small style="color:var(--text-muted)">JSONL · JSON · CSV · TSV · Parquet · TXT — up to 2 GB</small></div>' +
          '<input type="file" id="dsm-file" accept=".json,.jsonl,.csv,.tsv,.parquet,.txt" style="display:none"></div>' +
          '<div id="dsm-file-name" style="font-family:var(--font-mono);font-size:11.5px;color:var(--nvidia-green);margin-top:-8px"></div>';
        var drop = document.getElementById('dsm-drop');
        var fileIn = document.getElementById('dsm-file');
        drop.addEventListener('click', function () { fileIn.click(); });
        fileIn.addEventListener('change', function () {
          if (fileIn.files[0]) document.getElementById('dsm-file-name').textContent = fileIn.files[0].name + ' · ' + self.formatBytes(fileIn.files[0].size);
        });
        drop.addEventListener('dragover', function (e) { e.preventDefault(); drop.classList.add('drag-over'); });
        drop.addEventListener('dragleave', function () { drop.classList.remove('drag-over'); });
        drop.addEventListener('drop', function (e) {
          e.preventDefault();
          drop.classList.remove('drag-over');
          if (e.dataTransfer.files[0]) {
            fileIn.files = e.dataTransfer.files;
            document.getElementById('dsm-file-name').textContent = e.dataTransfer.files[0].name + ' · ' + self.formatBytes(e.dataTransfer.files[0].size);
          }
        });
      } else if (currentSrc === 'huggingface') {
        p.innerHTML = nameField +
          '<div class="form-group"><label class="form-label">HuggingFace Repo ID</label><input type="text" id="dsm-path" class="form-input" placeholder="tatsu-lab/alpaca"></div>' +
          '<div class="form-grid">' +
            '<div class="form-group"><label class="form-label">Config (optional)</label><input type="text" id="dsm-config" class="form-input" placeholder="default"></div>' +
            '<div class="form-group"><label class="form-label">Split (optional)</label><input type="text" id="dsm-split" class="form-input" placeholder="train"></div>' +
          '</div>';
      } else if (currentSrc === 'local') {
        p.innerHTML = nameField +
          '<div class="form-group"><label class="form-label">Absolute path</label><input type="text" id="dsm-path" class="form-input" placeholder="/home/user/data/train.jsonl"></div>' +
          '<div class="form-hint">The file must already exist and be readable. We will not copy it — just reference it.</div>';
      } else if (currentSrc === 'url') {
        p.innerHTML = nameField +
          '<div class="form-group"><label class="form-label">URL</label><input type="text" id="dsm-path" class="form-input" placeholder="https://example.com/data.jsonl"></div>' +
          '<div class="form-hint">The file will be downloaded to <code>~/.ainode/datasets/</code>.</div>';
      }
    };
    renderPanel();

    overlay.querySelectorAll('.pill[data-src]').forEach(function (p) {
      p.addEventListener('click', function () {
        overlay.querySelectorAll('.pill[data-src]').forEach(function (x) { x.classList.remove('active'); });
        p.classList.add('active');
        currentSrc = p.dataset.src;
        renderPanel();
      });
    });

    document.getElementById('dsm-submit').addEventListener('click', async function () {
      var btn = this;
      btn.disabled = true;
      btn.textContent = 'Adding…';
      var name = (document.getElementById('dsm-name') || {}).value || '';
      try {
        var resp;
        if (currentSrc === 'upload') {
          var fileIn = document.getElementById('dsm-file');
          if (!fileIn.files[0]) { self.toast('Please choose a file', 'error'); btn.disabled = false; btn.textContent = 'Add Dataset'; return; }
          var fd = new FormData();
          fd.append('file', fileIn.files[0]);
          if (name) fd.append('name', name);
          resp = await fetch('/api/datasets/upload', { method: 'POST', body: fd });
        } else {
          var body = { source: currentSrc, name: name };
          var pathEl = document.getElementById('dsm-path');
          var pathVal = pathEl ? pathEl.value.trim() : '';
          if (currentSrc === 'huggingface') {
            body.repo_id = pathVal;
            body.config = (document.getElementById('dsm-config') || {}).value || null;
            body.split = (document.getElementById('dsm-split') || {}).value || null;
          } else if (currentSrc === 'url') {
            body.url = pathVal;
          } else {
            body.path = pathVal;
          }
          resp = await fetch('/api/datasets', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        }
        var data = await resp.json();
        if (!resp.ok) { self.toast('Error: ' + (data.error || 'Failed'), 'error'); btn.disabled = false; btn.textContent = 'Add Dataset'; return; }
        self.toast('Dataset added', 'success');
        overlay.remove();
        self.renderTraining();
      } catch (err) {
        self.toast('Network error: ' + err.message, 'error');
        btn.disabled = false; btn.textContent = 'Add Dataset';
      }
    });
  },

  // =======================================================================
  //  GUIDE MODAL (markdown explainers)
  // =======================================================================

  showGuideModal(id) {
    var guides = this.guideContent || {};
    var g = guides[id] || { title: 'Guide', html: '<p>Guide coming soon.</p>' };
    var existing = document.querySelector('.model-detail-modal-overlay.guide-overlay');
    if (existing) existing.remove();
    var overlay = document.createElement('div');
    overlay.className = 'model-detail-modal-overlay guide-overlay';
    overlay.innerHTML =
      '<div class="model-detail-modal guide-modal">' +
        '<div class="md-header"><div class="md-header-left"><div class="md-icon">&#9733;</div><div style="font-weight:700">' + this.esc(g.title) + '</div></div>' +
          '<button class="btn-ghost btn-sm" id="guide-close">Close</button></div>' +
        '<div style="padding:22px 28px">' + g.html + '</div>' +
      '</div>';
    document.body.appendChild(overlay);
    overlay.addEventListener('click', function (e) { if (e.target === overlay) overlay.remove(); });
    document.getElementById('guide-close').addEventListener('click', function () { overlay.remove(); });
  },

  guideContent: {
    beginners: {
      title: "Beginner's Guide to Training",
      html: '<h2>What is training?</h2><p>Training — more precisely <strong>fine-tuning</strong> — takes a base model and teaches it your specific domain. You show it hundreds or thousands of examples and it learns the patterns.</p>' +
        '<h3>What you need</h3><ul><li>A base model (start small: 3B params)</li><li>Training data (JSONL, JSON, CSV — a few hundred examples minimum)</li><li>An NVIDIA GPU (you have one!)</li></ul>' +
        '<h3>Recommended first run</h3><ol><li>Pick <code>Llama 3.2 3B Instruct</code> as base model</li><li>Upload an Alpaca-style JSONL (<code>instruction</code> / <code>output</code>)</li><li>Method: <strong>LoRA</strong></li><li>Epochs: 3, batch size: 4</li><li>Click Launch — monitor from the Runs tab</li></ol>' +
        '<h3>Why LoRA?</h3><p>Full fine-tuning updates every weight in the model — huge memory cost. LoRA trains small adapter matrices on top of the frozen base, giving 95%+ of the quality at a fraction of the memory.</p>',
    },
    distributed: {
      title: 'Distributed Training',
      html: '<h2>Multi-node DDP</h2><p>AINode uses <strong>PyTorch Distributed Data Parallel</strong> (DDP) over NCCL for multi-node training. Each node holds a copy of the model; gradients are all-reduced between nodes each step.</p>' +
        '<h3>Requirements</h3><ul><li>2+ online AINode cluster members</li><li>Fast interconnect (10G+ recommended, 100G+ ideal)</li><li>Same model cached on every node</li><li>Shared dataset path or replicated data</li></ul>' +
        '<h3>When to use it</h3><p>Your training time scales roughly linearly with nodes for data-parallel workloads. Use 2+ nodes when:</p>' +
        '<ul><li>Your dataset is large (10k+ samples)</li><li>You want to cut wall-clock time</li><li>The model fits in a single node (for DDP — otherwise consider sharding)</li></ul>',
    },
    pipeline: {
      title: 'Training Pipeline',
      html: '<h2>End-to-end pipeline</h2>' +
        '<ol><li><strong>Data prep</strong> — convert your source data to JSONL with the expected fields</li>' +
        '<li><strong>Train</strong> — pick a base model, dataset, method; launch the run</li>' +
        '<li><strong>Merge</strong> — if LoRA, merge adapter into base model for deployment</li>' +
        '<li><strong>Evaluate</strong> — test on a held-out set before deploy</li>' +
        '<li><strong>Deploy</strong> — load the merged model as a vLLM instance</li></ol>' +
        '<h3>Checkpointing</h3><p>Long runs save checkpoints every N steps to <code>~/.ainode/training/jobs/&lt;id&gt;/output/</code>. You can resume from a checkpoint if training is interrupted.</p>',
    },
    standalone: {
      title: 'Standalone Training',
      html: '<h2>Single-node fine-tuning</h2>' +
        '<p>For most users, a single DGX Spark with 128 GB unified memory is plenty. You can train up to ~8B models with full fine-tuning, or up to 70B+ with LoRA / QLoRA.</p>' +
        '<h3>Tips</h3><ul>' +
        '<li>Use <code>QLoRA</code> for huge models — 4-bit quantized weights leave room for gradients</li>' +
        '<li>Enable gradient checkpointing if you run out of memory (trades compute for memory)</li>' +
        '<li>Keep <code>max_seq_length</code> as small as your data allows — memory scales quadratically with seq length</li>' +
        '<li>Watch the loss chart — if it plateaus early, try a higher learning rate; if it diverges, lower it</li>' +
        '</ul>',
    },
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
    var self = this;
    var placeholders = [];
    var codeBlockIndex = 0;

    // 1. Extract fenced code blocks first (so inner content isn't re-parsed)
    var html = text.replace(/```(\w*)\n?([\s\S]*?)```/g, function (match, lang, code) {
      var language = (lang || 'plaintext').toLowerCase();
      var escapedCode = self.esc(code.replace(/\n$/, ''));
      var id = 'code-' + Date.now() + '-' + (codeBlockIndex++);
      var block =
        '<div class="code-block-wrapper">' +
          '<div class="code-block-header">' +
            '<span class="code-block-lang">' + self.esc(language) + '</span>' +
            '<button class="code-copy-btn" data-code-id="' + id + '" title="Copy code">' +
              '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12">' +
                '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>' +
                '<path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>' +
              '</svg>' +
              '<span>Copy</span>' +
            '</button>' +
          '</div>' +
          '<pre class="code-block"><code id="' + id + '" class="language-' + self.esc(language) + '">' + escapedCode + '</code></pre>' +
        '</div>';
      placeholders.push(block);
      return '\u0001CB' + (placeholders.length - 1) + '\u0001';
    });

    // 2. Extract inline code
    html = html.replace(/`([^`\n]+)`/g, function (m, code) {
      placeholders.push('<code class="inline-code">' + self.esc(code) + '</code>');
      return '\u0001IC' + (placeholders.length - 1) + '\u0001';
    });

    // 3. Escape remaining HTML
    html = self.esc(html);

    // 4. Markdown inline formatting
    html = html
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/~~(.+?)~~/g, '<del>$1</del>');

    // 5. Auto-link URLs
    html = html.replace(
      /(https?:\/\/[^\s<]+[^\s<.,;:?!\)])/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer" class="chat-link">$1</a>'
    );

    // 6. Markdown-style links [text](url)
    html = html.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer" class="chat-link">$1</a>'
    );

    // 7. Headers (#, ##, ###)
    html = html
      .replace(/^### (.+)$/gm, '<h4 class="chat-h">$1</h4>')
      .replace(/^## (.+)$/gm, '<h3 class="chat-h">$1</h3>')
      .replace(/^# (.+)$/gm, '<h2 class="chat-h">$1</h2>');

    // 8. Lists
    html = html.replace(/^(\s*)[-*] (.+)$/gm, '$1• $2');

    // 9. Newlines to <br> (but not inside code blocks which are already placeholders)
    html = html.replace(/\n/g, '<br>');

    // 10. Restore code blocks / inline code
    html = html.replace(/\u0001CB(\d+)\u0001/g, function (_, idx) {
      return placeholders[parseInt(idx)];
    });
    html = html.replace(/\u0001IC(\d+)\u0001/g, function (_, idx) {
      return placeholders[parseInt(idx)];
    });

    return html;
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

  // ========================================================================
  //  CONFIG VIEW
  // ========================================================================

  renderConfig() {
    var nav = document.getElementById('config-nav');
    var self = this;
    if (nav && !nav.dataset.bound) {
      nav.dataset.bound = '1';
      nav.querySelectorAll('.config-nav-item').forEach(function (btn) {
        btn.addEventListener('click', function () {
          self.state.configSection = btn.dataset.section;
          nav.querySelectorAll('.config-nav-item').forEach(function (b) {
            b.classList.toggle('active', b.dataset.section === self.state.configSection);
          });
          self.renderConfigSection();
        });
      });
    }
    // Keep sidebar active in sync
    if (nav) {
      nav.querySelectorAll('.config-nav-item').forEach(function (b) {
        b.classList.toggle('active', b.dataset.section === self.state.configSection);
      });
    }
    this.renderConfigSection();
  },

  renderConfigSection() {
    switch (this.state.configSection) {
      case 'credentials': return this.renderConfigCredentials();
      case 'cluster':     return this.renderConfigCluster();
      case 'node':        return this.renderConfigNode();
      case 'storage':     return this.renderConfigStorage();
      case 'training':    return this.renderConfigTrainingDefaults();
      case 'network':     return this.renderConfigNetwork();
      case 'about':       return this.renderConfigAbout();
    }
  },

  _configMount() {
    return document.getElementById('config-content');
  },

  async _fetchConfigBundle() {
    var self = this;
    var results = await Promise.all([
      this.fetchJSON('/api/secrets'),
      this.fetchJSON('/api/cluster/info'),
      this.fetchJSON('/api/config'),
    ]);
    self.state.configData.secrets = results[0];
    self.state.configData.cluster = results[1];
    self.state.configData.config = results[2];
  },

  // ----- Credentials --------------------------------------------------------
  async renderConfigCredentials() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading credentials…</div>';
    var data = await this.fetchJSON('/api/secrets');
    this.state.configData.secrets = data;
    if (!data) {
      mount.innerHTML = '<div class="config-empty">Unable to load secrets.</div>';
      return;
    }
    var self = this;
    var known = data.known || {};
    var custom = data.custom || {};
    var html = '';
    html += '<h2 class="config-section-title">Credentials</h2>';
    html += '<p class="config-section-desc">Store API tokens used by AINode to download gated models, log training metrics, and access hosted services. Values are stored locally at <code>~/.ainode/secrets.json</code> with mode 0600 and never transmitted to AINode servers.</p>';

    html += '<div class="config-card">';
    html += '<h3 class="config-card-title">Known credentials</h3>';
    html += '<p class="config-card-desc">Recognized services with built-in integrations.</p>';
    Object.keys(known).forEach(function (k) {
      html += self._renderSecretRow(known[k]);
    });
    html += '</div>';

    html += '<div class="config-card">';
    html += '<h3 class="config-card-title">Custom secrets</h3>';
    html += '<p class="config-card-desc">Arbitrary named values, e.g. for training scripts or third-party integrations.</p>';
    if (Object.keys(custom).length === 0) {
      html += '<div class="config-empty">No custom secrets yet.</div>';
    } else {
      Object.keys(custom).forEach(function (name) {
        var s = custom[name];
        html += '<div class="config-secret-row">';
        html += '  <div class="config-secret-main">';
        html += '    <div class="config-secret-label-row">';
        html += '      <span class="config-secret-name">' + self.esc(name) + '</span>';
        html += s.is_set ? '      <span class="config-secret-mask">' + self.esc(s.masked) + '</span>' : '      <span class="config-secret-unset">not set</span>';
        html += '    </div>';
        html += '  </div>';
        html += '  <span></span>';
        html += '  <button class="config-icon-btn danger" data-custom-delete="' + self.esc(name) + '">Delete</button>';
        html += '</div>';
      });
    }
    html += '<div class="config-add-custom">';
    html += '  <input class="form-input" id="cfg-custom-name" placeholder="name (alphanumeric)">';
    html += '  <input class="form-input" id="cfg-custom-value" type="password" placeholder="value">';
    html += '  <button class="config-btn" id="cfg-custom-add">+ Add</button>';
    html += '</div>';
    html += '</div>';

    mount.innerHTML = html;

    // Wire save buttons for each known secret
    Object.keys(known).forEach(function (k) {
      self._wireSecretRow(k);
    });

    mount.querySelectorAll('[data-custom-delete]').forEach(function (btn) {
      btn.addEventListener('click', async function () {
        var name = btn.dataset.customDelete;
        if (!confirm('Delete custom secret "' + name + '"?')) return;
        await fetch('/api/secrets/custom/' + encodeURIComponent(name), { method: 'DELETE' });
        self.toast('Deleted ' + name, 'info');
        self.renderConfigCredentials();
      });
    });

    var addBtn = document.getElementById('cfg-custom-add');
    if (addBtn) {
      addBtn.addEventListener('click', async function () {
        var nameEl = document.getElementById('cfg-custom-name');
        var valEl = document.getElementById('cfg-custom-value');
        var name = (nameEl.value || '').trim();
        var value = valEl.value || '';
        if (!name || !value) { self.toast('Name and value required', 'error'); return; }
        var resp = await fetch('/api/secrets/custom/' + encodeURIComponent(name), {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value: value }),
        });
        var body = await resp.json().catch(function () { return {}; });
        if (!resp.ok) {
          self.toast((body.error && body.error.message) || 'Failed to add secret', 'error');
          return;
        }
        self.toast('Added ' + name, 'success');
        self.renderConfigCredentials();
      });
    }
  },

  _renderSecretRow(entry) {
    var self = this;
    var k = entry.key;
    var html = '<div class="config-secret-row" data-secret-row="' + self.esc(k) + '">';
    html += '  <div class="config-secret-main">';
    html += '    <div class="config-secret-label-row">';
    html += '      <span class="config-secret-name">' + self.esc(entry.label) + '</span>';
    html += entry.is_set
      ? '      <span class="config-secret-mask">' + self.esc(entry.masked) + '</span>'
      : '      <span class="config-secret-unset">not set</span>';
    html += '    </div>';
    html += '    <div class="config-secret-desc">' + self.esc(entry.description || '') +
            (entry.docs_url ? ' <a href="' + self.esc(entry.docs_url) + '" target="_blank" rel="noopener" style="color:var(--nvidia-green)">Docs ↗</a>' : '') +
            '</div>';
    html += '    <div class="config-secret-input-row" id="cfg-sec-input-' + self.esc(k) + '" style="' + (entry.is_set ? 'display:none' : '') + '">';
    html += '      <input class="form-input" type="password" id="cfg-sec-val-' + self.esc(k) + '" placeholder="' + self.esc(entry.prefix_hint ? entry.prefix_hint + '…' : 'paste token here') + '">';
    html += '      <button class="config-eye-btn" data-eye="' + self.esc(k) + '" type="button">Show</button>';
    html += '      <button class="config-btn" data-save-secret="' + self.esc(k) + '">Save</button>';
    html += '    </div>';
    html += '    <div class="config-test-result" id="cfg-sec-result-' + self.esc(k) + '" style="display:none"></div>';
    html += '  </div>';
    if (entry.is_set) {
      html += '  <button class="config-btn secondary" data-replace-secret="' + self.esc(k) + '">Replace</button>';
      html += '  <button class="config-btn danger" data-delete-secret="' + self.esc(k) + '">Delete</button>';
    } else {
      html += '  <span></span><span></span>';
    }
    if (entry.testable) {
      html += '<div style="grid-column: 1 / -1; margin-top: 6px; text-align: right;"><button class="config-btn secondary" data-test-secret="' + self.esc(k) + '"' + (entry.is_set ? '' : ' disabled') + '>Test connection</button></div>';
    }
    html += '</div>';
    return html;
  },

  _wireSecretRow(key) {
    var self = this;
    var mount = this._configMount();
    if (!mount) return;
    var row = mount.querySelector('[data-secret-row="' + key + '"]');
    if (!row) return;

    var eye = row.querySelector('[data-eye="' + key + '"]');
    if (eye) eye.addEventListener('click', function () {
      var inp = document.getElementById('cfg-sec-val-' + key);
      if (!inp) return;
      var shown = inp.type === 'text';
      inp.type = shown ? 'password' : 'text';
      eye.textContent = shown ? 'Show' : 'Hide';
    });

    var save = row.querySelector('[data-save-secret="' + key + '"]');
    if (save) save.addEventListener('click', async function () {
      var inp = document.getElementById('cfg-sec-val-' + key);
      var val = inp ? inp.value : '';
      if (!val) { self.toast('Value is required', 'error'); return; }
      var resp = await fetch('/api/secrets/' + encodeURIComponent(key), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value: val }),
      });
      var body = await resp.json().catch(function () { return {}; });
      if (!resp.ok) {
        self.toast((body.error && body.error.message) || 'Save failed', 'error');
        return;
      }
      if (inp) inp.value = '';
      self.toast('Saved', 'success');
      self.renderConfigCredentials();
    });

    var replace = row.querySelector('[data-replace-secret="' + key + '"]');
    if (replace) replace.addEventListener('click', function () {
      var wrap = document.getElementById('cfg-sec-input-' + key);
      if (wrap) wrap.style.display = '';
      replace.style.display = 'none';
    });

    var del = row.querySelector('[data-delete-secret="' + key + '"]');
    if (del) del.addEventListener('click', async function () {
      if (!confirm('Delete ' + key + '?')) return;
      await fetch('/api/secrets/' + encodeURIComponent(key), { method: 'DELETE' });
      self.toast('Deleted', 'info');
      self.renderConfigCredentials();
    });

    var test = row.querySelector('[data-test-secret="' + key + '"]');
    if (test) test.addEventListener('click', async function () {
      var result = document.getElementById('cfg-sec-result-' + key);
      if (result) { result.style.display = ''; result.className = 'config-test-result'; result.textContent = 'Testing…'; }
      var resp = await fetch('/api/secrets/' + encodeURIComponent(key) + '/test');
      var body = await resp.json().catch(function () { return {}; });
      if (!result) return;
      if (body.ok) {
        result.className = 'config-test-result ok';
        result.textContent = 'OK — authenticated as ' + (body.identity || 'user');
      } else {
        result.className = 'config-test-result err';
        result.textContent = 'Failed: ' + (body.message || 'unknown error');
      }
    });
  },

  // ----- Cluster ------------------------------------------------------------
  async renderConfigCluster() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading cluster info…</div>';
    var data = await this.fetchJSON('/api/cluster/info');
    this.state.configData.cluster = data;
    if (!data) {
      mount.innerHTML = '<div class="config-empty">Unable to load cluster info.</div>';
      return;
    }
    var self = this;
    var role = data.my_role || 'worker';
    var configured = data.configured_role || 'auto';
    var masterId = data.master_node_id;
    var iAmMaster = role === 'master';
    var members = data.members || [];
    var peers = members.filter(function (m) { return m.node_id !== data.my_node_id; });

    var badgeTitle, badgeSub, badgeIcon, badgeCls;
    if (iAmMaster) {
      badgeIcon = '⚡';
      badgeTitle = 'MASTER — You are the cluster head';
      badgeSub = peers.length > 0 ? 'Serving ' + peers.length + ' worker(s)'
                                   : (configured === 'auto' ? 'Waiting for peers (auto-elected)' : 'Standalone');
      badgeCls = 'master';
    } else {
      badgeIcon = '🔗';
      var masterNode = members.find(function (m) { return m.node_id === masterId; });
      badgeTitle = 'WORKER';
      badgeSub = masterNode
        ? 'Connected to master: ' + masterNode.node_name + ' (' + (data.master_address || masterNode.node_id) + ')'
        : 'No master detected yet';
      badgeCls = 'worker';
    }

    var html = '';
    html += '<h2 class="config-section-title">Cluster</h2>';
    html += '<p class="config-section-desc">Nodes on the same network with a matching <code>cluster_id</code> form a cluster. Master is elected automatically, or you can pin a role explicitly.</p>';

    html += '<div class="config-role-badge ' + badgeCls + '">';
    html += '  <div class="config-role-badge-icon">' + badgeIcon + '</div>';
    html += '  <div class="config-role-badge-text">';
    html += '    <div class="config-role-badge-title">' + self.esc(badgeTitle) + '</div>';
    html += '    <div class="config-role-badge-subtitle">' + self.esc(badgeSub) + '</div>';
    html += '  </div>';
    html += '</div>';

    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">Role</h3>';
    html += '  <p class="config-card-desc">Auto = elected dynamically · Master = always cluster head · Worker = never becomes master.</p>';
    html += '  <div class="config-role-pills pill-group">';
    ['auto', 'master', 'worker'].forEach(function (r) {
      html += '<button class="pill' + (configured === r ? ' active' : '') + '" data-set-role="' + r + '">' + r.toUpperCase() + '</button>';
    });
    html += '  </div>';
    html += '</div>';

    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">Cluster ID</h3>';
    html += '  <p class="config-card-desc">Only nodes sharing this identifier will see each other.</p>';
    html += '  <div class="config-secret-input-row">';
    html += '    <input class="form-input" id="cfg-cluster-id" value="' + self.esc(data.cluster_id || 'default') + '">';
    html += '    <button class="config-btn" id="cfg-cluster-id-save">Save</button>';
    html += '  </div>';
    html += '</div>';

    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">Members</h3>';
    html += '  <p class="config-card-desc">Nodes currently visible in this cluster.</p>';
    if (members.length === 0) {
      html += '<div class="config-empty">No members yet.</div>';
    } else {
      html += '<div class="config-member-table">';
      html += '<div class="config-member-row header"><span>Node</span><span>Address</span><span>Role</span><span>Status</span><span>Last seen</span></div>';
      members.forEach(function (m) {
        var when = m.last_seen ? new Date(m.last_seen * 1000).toLocaleTimeString() : '—';
        html += '<div class="config-member-row">';
        html += '  <span><strong>' + self.esc(m.node_name || m.node_id) + '</strong><div style="color:var(--text-muted);font-size:11px" class="mono">' + self.esc(m.node_id) + '</div></span>';
        html += '  <span class="mono">' + self.esc(m.node_name) + ':' + self.esc(String(m.web_port)) + '</span>';
        html += '  <span><span class="config-member-role-tag ' + m.effective_role + '">' + m.effective_role + '</span></span>';
        html += '  <span>' + self.esc(m.status) + '</span>';
        html += '  <span class="mono" style="color:var(--text-muted)">' + when + '</span>';
        html += '</div>';
      });
      html += '</div>';
    }
    html += '</div>';

    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">How election works</h3>';
    html += '  <p class="config-card-desc">If any node is explicitly configured as <code>master</code>, the lowest-ID such node wins. Otherwise the lowest node_id among <code>auto</code> nodes becomes master. <code>worker</code> nodes are never elected. Election is re-run every 5s as heartbeats arrive; changing your role takes effect on the next broadcast (≤5s) without a restart.</p>';
    html += '</div>';

    mount.innerHTML = html;

    mount.querySelectorAll('[data-set-role]').forEach(function (btn) {
      btn.addEventListener('click', async function () {
        var role = btn.dataset.setRole;
        var resp = await fetch('/api/cluster/role', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ role: role }),
        });
        if (!resp.ok) { self.toast('Role change failed', 'error'); return; }
        self.toast('Role → ' + role.toUpperCase(), 'success');
        setTimeout(function () { self.renderConfigCluster(); }, 500);
      });
    });

    var idBtn = document.getElementById('cfg-cluster-id-save');
    if (idBtn) idBtn.addEventListener('click', async function () {
      var val = document.getElementById('cfg-cluster-id').value.trim() || 'default';
      var resp = await fetch('/api/cluster/id', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cluster_id: val }),
      });
      var body = await resp.json().catch(function () { return {}; });
      if (!resp.ok) { self.toast((body.error && body.error.message) || 'Save failed', 'error'); return; }
      self.toast('Cluster ID saved', 'success');
      setTimeout(function () { self.renderConfigCluster(); }, 500);
    });
  },

  // ----- Node Identity ------------------------------------------------------
  async renderConfigNode() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var cfg = await this.fetchJSON('/api/config');
    this.state.configData.config = cfg;
    if (!cfg) { mount.innerHTML = '<div class="config-empty">Unable to load config.</div>'; return; }
    var self = this;
    var html = '';
    html += '<h2 class="config-section-title">Node Identity</h2>';
    html += '<p class="config-section-desc">Human-readable name and owner contact for this node.</p>';
    html += '<div class="config-card"><div class="config-form-grid">';
    html += this._field('Node name', 'node_name', cfg.node_name || '');
    html += this._field('Node ID (read-only)', 'node_id', cfg.node_id || '', { readonly: true, mono: true });
    html += this._field('Email', 'email', cfg.email || '', { type: 'email' });
    html += '</div>';
    html += '<div class="config-actions"><button class="config-btn" id="cfg-node-save">Save</button></div>';
    html += '</div>';
    mount.innerHTML = html;
    document.getElementById('cfg-node-save').addEventListener('click', function () {
      self._patchConfig({
        node_name: document.getElementById('cfg-f-node_name').value,
        email: document.getElementById('cfg-f-email').value,
      });
    });
  },

  // ----- Storage ------------------------------------------------------------
  async renderConfigStorage() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var cfg = await this.fetchJSON('/api/config');
    this.state.configData.config = cfg;
    if (!cfg) { mount.innerHTML = '<div class="config-empty">Unable to load config.</div>'; return; }
    var self = this;
    var html = '';
    html += '<h2 class="config-section-title">Storage</h2>';
    html += '<p class="config-section-desc">Where AINode keeps downloaded models, datasets, and training artifacts. Leave blank to use defaults under <code>~/.ainode/</code>.</p>';
    html += '<div class="config-card"><div class="config-form-grid single">';
    html += this._field('Models directory', 'models_dir', cfg.models_dir || '', { hint: 'vLLM/HF model weights' });
    html += this._field('Datasets directory', 'datasets_dir', cfg.datasets_dir || '', { hint: 'Training / eval datasets' });
    html += this._field('Training output directory', 'training_dir', cfg.training_dir || '', { hint: 'Checkpoints and logs' });
    html += this._field('HuggingFace cache', 'hf_cache_dir', cfg.hf_cache_dir || '', { hint: 'Defaults to $HF_HOME / ~/.cache/huggingface' });
    html += '</div>';
    html += '<div class="config-actions"><button class="config-btn" id="cfg-storage-save">Save</button></div>';
    html += '</div>';
    mount.innerHTML = html;
    document.getElementById('cfg-storage-save').addEventListener('click', function () {
      self._patchConfig({
        models_dir: document.getElementById('cfg-f-models_dir').value,
        datasets_dir: document.getElementById('cfg-f-datasets_dir').value,
        training_dir: document.getElementById('cfg-f-training_dir').value,
        hf_cache_dir: document.getElementById('cfg-f-hf_cache_dir').value,
      });
    });
  },

  // ----- Training Defaults --------------------------------------------------
  async renderConfigTrainingDefaults() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var cfg = await this.fetchJSON('/api/config');
    this.state.configData.config = cfg;
    if (!cfg) { mount.innerHTML = '<div class="config-empty">Unable to load config.</div>'; return; }
    var self = this;
    var method = cfg.training_default_method || 'lora';
    var html = '';
    html += '<h2 class="config-section-title">Training Defaults</h2>';
    html += '<p class="config-section-desc">Default values prefilled when starting a new fine-tuning run.</p>';
    html += '<div class="config-card"><div class="config-form-grid">';
    html += '<div><label class="config-field-label">Default method</label>';
    html += '<select class="form-select" id="cfg-f-training_default_method">';
    ['lora', 'qlora', 'full'].forEach(function (m) {
      html += '<option value="' + m + '"' + (method === m ? ' selected' : '') + '>' + m.toUpperCase() + '</option>';
    });
    html += '</select></div>';
    html += this._field('Default epochs', 'training_default_epochs', cfg.training_default_epochs, { type: 'number' });
    html += this._field('Default batch size', 'training_default_batch_size', cfg.training_default_batch_size, { type: 'number' });
    html += this._field('Default learning rate', 'training_default_learning_rate', cfg.training_default_learning_rate, { type: 'number', step: '0.00001' });
    html += '</div>';
    html += '<div class="config-actions"><button class="config-btn" id="cfg-training-save">Save</button></div>';
    html += '</div>';
    mount.innerHTML = html;
    document.getElementById('cfg-training-save').addEventListener('click', function () {
      self._patchConfig({
        training_default_method: document.getElementById('cfg-f-training_default_method').value,
        training_default_epochs: parseInt(document.getElementById('cfg-f-training_default_epochs').value, 10),
        training_default_batch_size: parseInt(document.getElementById('cfg-f-training_default_batch_size').value, 10),
        training_default_learning_rate: parseFloat(document.getElementById('cfg-f-training_default_learning_rate').value),
      });
    });
  },

  // ----- Network ------------------------------------------------------------
  async renderConfigNetwork() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var cfg = await this.fetchJSON('/api/config');
    this.state.configData.config = cfg;
    if (!cfg) { mount.innerHTML = '<div class="config-empty">Unable to load config.</div>'; return; }
    var self = this;
    var host = cfg.host || '0.0.0.0';
    var html = '';
    html += '<h2 class="config-section-title">Network</h2>';
    html += '<p class="config-section-desc">Port and binding configuration. <strong>Changes to ports require an AINode restart to take effect.</strong></p>';
    html += '<div class="config-card"><div class="config-form-grid">';
    html += this._field('API port (vLLM)', 'api_port', cfg.api_port, { type: 'number' });
    html += this._field('Web port (UI + proxy)', 'web_port', cfg.web_port, { type: 'number' });
    html += this._field('Discovery port (UDP)', 'discovery_port', cfg.discovery_port, { type: 'number' });
    html += '<div><label class="config-field-label">Bind host</label>';
    html += '<select class="form-select" id="cfg-f-host">';
    [['0.0.0.0', 'All interfaces (0.0.0.0)'], ['127.0.0.1', 'Localhost only (127.0.0.1)']].forEach(function (pair) {
      html += '<option value="' + pair[0] + '"' + (host === pair[0] ? ' selected' : '') + '>' + pair[1] + '</option>';
    });
    html += '</select></div>';
    html += this._field('CORS origins', 'cors_origins', cfg.cors_origins || '', { hint: 'Comma-separated list of allowed origins' });
    html += '</div>';
    html += '<div class="config-actions"><button class="config-btn" id="cfg-net-save">Save</button></div>';
    html += '</div>';
    mount.innerHTML = html;
    document.getElementById('cfg-net-save').addEventListener('click', function () {
      self._patchConfig({
        api_port: parseInt(document.getElementById('cfg-f-api_port').value, 10),
        web_port: parseInt(document.getElementById('cfg-f-web_port').value, 10),
        discovery_port: parseInt(document.getElementById('cfg-f-discovery_port').value, 10),
        host: document.getElementById('cfg-f-host').value,
        cors_origins: document.getElementById('cfg-f-cors_origins').value,
      }, { restartHint: true });
    });
  },

  // ----- About --------------------------------------------------------------
  async renderConfigAbout() {
    var mount = this._configMount();
    if (!mount) return;
    var status = await this.fetchJSON('/api/status');
    var gpu = status && status.gpu;
    var html = '';
    html += '<h2 class="config-section-title">About</h2>';
    html += '<p class="config-section-desc">AINode — local AI platform powered by argentos.ai.</p>';
    html += '<div class="config-card"><div class="config-form-grid">';
    html += '<div><div class="config-field-label">AINode version</div><div>' + this.esc((status && status.version) || 'n/a') + '</div></div>';
    html += '<div><div class="config-field-label">Node ID</div><div class="mono" style="font-family:var(--font-mono);font-size:12px">' + this.esc((status && status.node_id) || 'n/a') + '</div></div>';
    html += '<div><div class="config-field-label">Current model</div><div>' + this.esc((status && status.model) || 'none') + '</div></div>';
    html += '<div><div class="config-field-label">GPU</div><div>' + this.esc(gpu ? (gpu.name + (gpu.memory_total_mb ? ' · ' + Math.round(gpu.memory_total_mb / 1024) + ' GB' : '')) : 'CPU only') + '</div></div>';
    html += '<div><div class="config-field-label">Cluster role</div><div>' + this.esc((status && status.cluster_role) || 'n/a') + '</div></div>';
    html += '<div><div class="config-field-label">Master node</div><div class="mono">' + this.esc((status && status.master_node_id) || '—') + '</div></div>';
    html += '</div></div>';
    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">Links</h3>';
    html += '  <p><a href="https://ainode.dev" target="_blank" rel="noopener" style="color:var(--nvidia-green)">ainode.dev</a> · ';
    html += '  <a href="https://github.com/getainode/ainode" target="_blank" rel="noopener" style="color:var(--nvidia-green)">GitHub</a> · ';
    html += '  <a href="https://docs.argentos.ai" target="_blank" rel="noopener" style="color:var(--nvidia-green)">Docs</a></p>';
    html += '  <p class="config-card-desc" style="margin-top:10px">Licensed under Apache 2.0. Powered by argentos.ai.</p>';
    html += '</div>';
    mount.innerHTML = html;
  },

  // ----- Helpers ------------------------------------------------------------
  _field(label, key, value, opts) {
    opts = opts || {};
    var type = opts.type || 'text';
    var readonly = opts.readonly ? ' readonly' : '';
    var step = opts.step ? ' step="' + opts.step + '"' : '';
    var mono = opts.mono ? ' style="font-family:var(--font-mono);font-size:12px"' : '';
    var val = value === null || value === undefined ? '' : String(value);
    var html = '<div>';
    html += '  <label class="config-field-label" for="cfg-f-' + key + '">' + this.esc(label) + '</label>';
    html += '  <input class="form-input" id="cfg-f-' + key + '" type="' + type + '"' + step + readonly + mono + ' value="' + this.esc(val) + '">';
    if (opts.hint) html += '<div class="config-field-hint">' + this.esc(opts.hint) + '</div>';
    html += '</div>';
    return html;
  },

  async _patchConfig(patch, opts) {
    opts = opts || {};
    // Strip NaN / empty numeric
    Object.keys(patch).forEach(function (k) {
      var v = patch[k];
      if (typeof v === 'number' && Number.isNaN(v)) delete patch[k];
      if (v === '') patch[k] = null;
    });
    var resp = await fetch('/api/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    var body = await resp.json().catch(function () { return {}; });
    if (!resp.ok) {
      this.toast((body.error && body.error.message) || 'Save failed', 'error');
      return;
    }
    if (opts.restartHint) {
      this.toast('Saved — restart AINode for network changes to apply', 'info');
    } else {
      this.toast('Saved', 'success');
    }
  },

  // ========================================================================
  //  SERVER VIEW (LM Studio-style console)
  // ========================================================================

  _serverState: {
    logsSince: 0,
    logsPoll: null,
    autoScroll: true,
    selectedModelId: null,
    lastStatus: null,
    endpoints: null,
    endpointTab: 'openai',
    logs: [],
  },

  async renderServer() {
    var mount = document.getElementById('server-content');
    if (!mount) return;
    var self = this;

    // Fetch status + endpoints in parallel (endpoints cached)
    var status = await this.fetchJSON('/api/server/status');
    this._serverState.lastStatus = status;
    if (!this._serverState.endpoints) {
      this._serverState.endpoints = await this.fetchJSON('/api/server/endpoints');
    }

    var s = status || { status: 'stopped', reachable_at: [], loaded_models: [] };
    var primaryUrl = (s.reachable_at && s.reachable_at.length > 1)
      ? s.reachable_at[1]
      : (s.reachable_at && s.reachable_at[0]) || '—';

    var html = '';

    // --- Top status bar ---
    html += '<div class="server-status-bar">';
    html += '  <div class="server-status-left">';
    html += '    <span class="server-status-indicator"><span class="server-status-dot running"></span> Running</span>';
    html += '    <button class="btn-ghost server-btn-sm" id="server-toggle">Stop</button>';
    html += '    <button class="btn-ghost server-btn-sm" id="server-settings-btn">Server Settings</button>';
    html += '    <button class="btn-ghost server-btn-sm" id="server-mcp-btn">mcp.json</button>';
    html += '  </div>';
    html += '  <div class="server-status-center">';
    html += '    <span class="server-reachable-label">Reachable at:</span>';
    html += '    <span class="server-reachable-url mono" id="server-reachable-url">' + this.esc(primaryUrl) + '</span>';
    html += '    <button class="server-copy-btn" data-copy="' + this.esc(primaryUrl) + '" title="Copy">⧉</button>';
    html += '    <span class="server-cluster-summary mono" id="server-cluster-summary" style="margin-left:16px;color:#76B900"></span>';
    html += '  </div>';
    html += '  <div class="server-status-right">';
    html += '    <button class="btn-nvidia server-btn-sm" id="server-load-model">+ Load Model</button>';
    html += '  </div>';
    html += '</div>';

    // --- Loaded Models ---
    html += '<section class="server-section">';
    html += '  <div class="server-section-header">';
    html += '    <h3 class="server-section-title">Loaded Models</h3>';
    html += '    <span class="server-section-meta">' + ((s.loaded_models || []).length) + ' loaded · ' + this.esc(this.formatUptime(Math.floor(s.uptime_seconds || 0))) + ' uptime</span>';
    html += '  </div>';
    if (!s.loaded_models || s.loaded_models.length === 0) {
      html += '  <div class="server-empty">No models loaded — click <strong>+ Load Model</strong> to start one</div>';
    } else {
      html += '  <div class="server-loaded-list" id="server-loaded-list">';
      s.loaded_models.forEach(function (m, idx) {
        html += self._renderLoadedCard(m, idx);
      });
      html += '  </div>';
    }
    html += '</section>';

    // --- Supported Endpoints ---
    html += '<section class="server-section">';
    html += '  <div class="server-section-header">';
    html += '    <h3 class="server-section-title">Supported endpoints</h3>';
    html += '    <span class="badge-new">NEW REST API v1</span>';
    html += '  </div>';
    html += '  <div class="server-tab-pills" id="server-endpoint-tabs">';
    html += '    <button class="server-tab-pill' + (this._serverState.endpointTab === 'openai' ? ' active' : '') + '" data-tab="openai">OpenAI-compatible</button>';
    html += '    <button class="server-tab-pill' + (this._serverState.endpointTab === 'lmstudio' ? ' active' : '') + '" data-tab="lmstudio">LM Studio API</button>';
    html += '    <button class="server-tab-pill' + (this._serverState.endpointTab === 'anthropic' ? ' active' : '') + '" data-tab="anthropic">Anthropic-compatible</button>';
    html += '  </div>';
    html += '  <div class="server-endpoints-list" id="server-endpoints-list">';
    html += this._renderEndpointRows(primaryUrl);
    html += '  </div>';
    html += '</section>';

    // --- Developer Logs ---
    html += '<section class="server-section">';
    html += '  <div class="server-section-header">';
    html += '    <h3 class="server-section-title">Developer Logs</h3>';
    html += '    <div class="server-log-controls">';
    html += '      <label class="server-log-toggle"><input type="checkbox" id="server-log-autoscroll"' + (this._serverState.autoScroll ? ' checked' : '') + '> auto-scroll</label>';
    html += '      <button class="btn-ghost server-btn-sm" id="server-log-clear">Clear</button>';
    html += '    </div>';
    html += '  </div>';
    html += '  <div class="server-log-panel" id="server-log-panel">';
    html += this._renderLogEntries(this._serverState.logs);
    html += '  </div>';
    html += '</section>';

    mount.innerHTML = html;

    this._bindServerEvents();
    this._renderServerRightPanel(this._serverState.selectedModelId);
  },

  _renderLoadedCard(m, idx) {
    var id = m.id || 'unknown';
    var nodeHost = m.node_hostname || m.node_id || 'local';
    var type = m.type || 'llm';
    var isEmbed = type === 'embed';
    var sizeStr = m.size_bytes > 0 ? this.formatBytes(m.size_bytes) : '—';
    var parallel = m.parallel || 1;
    var selected = (this._serverState.selectedModelId === id) ? ' selected' : '';
    var typeTagStyle = isEmbed
      ? ' style="color:var(--cyan);border-color:var(--cyan)"'
      : '';
    var dimsMeta = isEmbed && m.dimensions
      ? '  <span class="server-meta">· ' + m.dimensions + 'd</span>'
      : '';
    var primaryIconBtn = isEmbed
      ? '  <button class="server-icon-btn" data-action="show-info" data-model="' + this.esc(id) + '" title="Embedding info">ℹ</button>'
      : '  <button class="server-icon-btn" data-action="open-chat" data-model="' + this.esc(id) + '" title="Open in Chat">🔍</button>';
    return '<div class="server-loaded-card' + selected + '" data-model-id="' + this.esc(id) + '" data-idx="' + idx + '">' +
      '<div class="server-loaded-left">' +
      '  <span class="server-badge ready">READY</span>' +
      '  <span class="server-node-pill">' + this.esc(nodeHost) + '</span>' +
      '  <span class="server-type-tag"' + typeTagStyle + '>' + this.esc(type) + '</span>' +
      '  <span class="server-model-id mono" data-copy="' + this.esc(id) + '" title="Click to copy">' + this.esc(id) + '</span>' +
      '</div>' +
      '<div class="server-loaded-right">' +
      '  <span class="server-meta">' + this.esc(sizeStr) + '</span>' +
      '  <span class="server-meta">· ' + parallel + 'x</span>' +
      dimsMeta +
      '  <button class="server-icon-btn" data-action="preview" title="Preview">👁</button>' +
      primaryIconBtn +
      '  <button class="server-icon-btn" data-action="copy-curl" data-model="' + this.esc(id) + '" data-type="' + this.esc(type) + '" title="Copy curl">⎘</button>' +
      '  <button class="server-eject-btn" data-action="eject" data-model="' + this.esc(id) + '">Eject</button>' +
      '</div>' +
      '</div>';
  },

  _renderEndpointRows(baseUrl) {
    var self = this;
    var tab = this._serverState.endpointTab;
    var catalog = this._serverState.endpoints || {};
    var rows = catalog[tab] || [];
    if (!rows.length) return '<div class="server-empty">No endpoints</div>';
    return rows.map(function (ep) {
      var planned = ep.status === 'planned';
      var methodClass = 'server-method-' + ep.method.toLowerCase();
      var dim = planned ? ' planned' : '';
      return '<div class="server-endpoint-row' + dim + '">' +
        '<span class="server-method-badge ' + methodClass + '">' + self.esc(ep.method) + '</span>' +
        '<span class="server-endpoint-path mono">' + self.esc(ep.path) + '</span>' +
        '<span class="server-endpoint-desc">' + self.esc(ep.description || '') + '</span>' +
        (planned ? '<span class="server-endpoint-planned">planned</span>' : '') +
        '<button class="server-copy-btn" data-copy="curl -X ' + self.esc(ep.method) + ' ' + self.esc(baseUrl) + self.esc(ep.path) + '" title="Copy curl">⧉</button>' +
        '</div>';
    }).join('');
  },

  _renderLogEntries(entries) {
    var self = this;
    if (!entries || entries.length === 0) {
      return '<div class="server-log-empty">Waiting for API requests…</div>';
    }
    return entries.map(function (e) {
      var ts = new Date((e.timestamp || 0) * 1000);
      var tsStr = ts.toISOString().replace('T', ' ').replace('Z', '').slice(0, 19);
      var levelCls = 'log-' + (e.level || 'INFO').toLowerCase();
      var modelStr = e.model ? ' [' + self.esc(e.model) + ']' : '';
      var sizeStr = e.content_length ? ' ' + self.formatBytes(e.content_length) : '';
      return '<div class="server-log-entry ' + levelCls + '">' +
        '<span class="log-ts mono">' + tsStr + '</span> ' +
        '<span class="log-level">[' + self.esc(e.level || 'INFO') + ']</span>' +
        modelStr +
        ' <span class="log-method mono">' + self.esc(e.method) + '</span> ' +
        '<span class="log-path mono">' + self.esc(e.path) + '</span> ' +
        '<span class="log-status mono">' + (e.status || 0) + '</span> ' +
        '<span class="log-dur mono">' + (e.duration_ms || 0) + 'ms</span>' +
        '<span class="log-size mono">' + sizeStr + '</span>' +
        '</div>';
    }).join('');
  },

  _bindServerEvents() {
    var self = this;
    var root = document.getElementById('server-content');
    if (!root) return;

    // Copy buttons
    root.querySelectorAll('[data-copy]').forEach(function (el) {
      el.addEventListener('click', function (e) {
        e.stopPropagation();
        var text = el.dataset.copy || '';
        if (!text) return;
        navigator.clipboard.writeText(text).then(function () {
          self.toast('Copied to clipboard', 'success');
        }).catch(function () { self.toast('Copy failed', 'error'); });
      });
    });

    // Endpoint tab pills
    root.querySelectorAll('.server-tab-pill').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self._serverState.endpointTab = btn.dataset.tab;
        var s = self._serverState.lastStatus || {};
        var primary = (s.reachable_at && s.reachable_at.length > 1) ? s.reachable_at[1] : (s.reachable_at && s.reachable_at[0]) || '';
        var list = document.getElementById('server-endpoints-list');
        if (list) list.innerHTML = self._renderEndpointRows(primary);
        root.querySelectorAll('.server-tab-pill').forEach(function (b) {
          b.classList.toggle('active', b.dataset.tab === self._serverState.endpointTab);
        });
        // Rebind copy buttons in newly rendered rows
        self._bindServerEvents();
      });
    });

    // Loaded card click → select + update right panel
    root.querySelectorAll('.server-loaded-card').forEach(function (card) {
      card.addEventListener('click', function () {
        var id = card.dataset.modelId;
        self._serverState.selectedModelId = id;
        root.querySelectorAll('.server-loaded-card').forEach(function (c) { c.classList.remove('selected'); });
        card.classList.add('selected');
        self._renderServerRightPanel(id);
      });
    });

    // Loaded card actions
    root.querySelectorAll('.server-loaded-card [data-action]').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var action = btn.dataset.action;
        var model = btn.dataset.model;
        if (action === 'eject') self._serverEjectModel(model);
        else if (action === 'open-chat') {
          self.state.chatModel = model;
          self.navigate('chat');
        }
        else if (action === 'show-info') self._serverShowEmbedInfo(model);
        else if (action === 'copy-curl') self._serverShowCurl(model, btn.dataset.type || 'llm');
        else if (action === 'preview') self.toast('Preview coming soon', 'info');
      });
    });

    // Top-bar buttons
    var toggleBtn = document.getElementById('server-toggle');
    if (toggleBtn) toggleBtn.addEventListener('click', function () { self.toast('Server toggle not yet implemented', 'info'); });
    var settingsBtn = document.getElementById('server-settings-btn');
    if (settingsBtn) settingsBtn.addEventListener('click', function () { self.navigate('config'); });
    var mcpBtn = document.getElementById('server-mcp-btn');
    if (mcpBtn) mcpBtn.addEventListener('click', function () { self.toast('mcp.json export coming soon', 'info'); });
    var loadBtn = document.getElementById('server-load-model');
    if (loadBtn) loadBtn.addEventListener('click', function () { self._openLoadModelModal(); });

    // Logs
    var clearBtn = document.getElementById('server-log-clear');
    if (clearBtn) clearBtn.addEventListener('click', function () { self._serverClearLogs(); });
    var autoScroll = document.getElementById('server-log-autoscroll');
    if (autoScroll) autoScroll.addEventListener('change', function () { self._serverState.autoScroll = autoScroll.checked; });
  },

  async _serverEjectModel(modelId) {
    if (!modelId) return;
    if (!confirm('Eject model ' + modelId + '?')) return;
    try {
      var resp = await fetch('/api/server/models/' + encodeURIComponent(modelId) + '/eject', { method: 'POST' });
      var body = await resp.json().catch(function () { return {}; });
      if (resp.ok) this.toast(body.message || 'Model ejected', 'success');
      else this.toast(body.message || 'Eject not available', 'info');
      this.renderServer();
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }
  },

  _serverShowCurl(modelId, modelType) {
    var s = this._serverState.lastStatus || {};
    var primary = (s.reachable_at && s.reachable_at.length > 1) ? s.reachable_at[1] : (s.reachable_at && s.reachable_at[0]) || 'http://localhost:3000';
    var curl;
    if (modelType === 'embed') {
      curl = 'curl ' + primary + '/v1/embeddings \\\n' +
        '  -H "Content-Type: application/json" \\\n' +
        '  -d \'{"model":"' + modelId + '","input":"The quick brown fox"}\'';
    } else {
      curl = 'curl ' + primary + '/v1/chat/completions \\\n' +
        '  -H "Content-Type: application/json" \\\n' +
        '  -d \'{"model":"' + modelId + '","messages":[{"role":"user","content":"Hello!"}]}\'';
    }
    navigator.clipboard.writeText(curl).then(function () {
      AINode.toast('curl example copied', 'success');
    }).catch(function () { AINode.toast('Copy failed', 'error'); });
  },

  _serverShowEmbedInfo(modelId) {
    var s = this._serverState.lastStatus || {};
    var models = s.loaded_models || [];
    var m = models.find(function (x) { return x.id === modelId; }) || { id: modelId };
    var primary = (s.reachable_at && s.reachable_at.length > 1) ? s.reachable_at[1] : (s.reachable_at && s.reachable_at[0]) || 'http://localhost:3000';
    var self = this;

    var existing = document.getElementById('embed-info-modal');
    if (existing) existing.remove();

    var modal = document.createElement('div');
    modal.id = 'embed-info-modal';
    modal.className = 'model-detail-modal-overlay';
    modal.innerHTML =
      '<div class="model-detail-modal" style="max-width:560px">' +
        '<div class="md-header">' +
          '<div class="md-header-left">' +
            '<div class="md-icon" style="color:var(--cyan);border-color:var(--cyan)">ℹ</div>' +
            '<div class="md-title">Embedding Model</div>' +
          '</div>' +
          '<button class="md-close">×</button>' +
        '</div>' +
        '<div class="md-description">' +
          '<div style="margin-bottom:8px" class="mono">' + self.esc(m.id) + '</div>' +
          '<div style="color:var(--text-muted);font-size:13px;margin-bottom:12px">' +
          'Embedding models turn text into vectors. They are an API surface for external apps ' +
          '(RAG, semantic search, clustering) — not used directly by the chat UI.' +
          '</div>' +
          '<div style="display:grid;grid-template-columns:auto 1fr;gap:6px 16px;font-size:13px">' +
            '<div style="color:var(--text-muted)">Dimensions</div><div>' + (m.dimensions || '—') + '</div>' +
            '<div style="color:var(--text-muted)">Max seq length</div><div>' + (m.max_seq_length || '—') + '</div>' +
            '<div style="color:var(--text-muted)">Endpoint</div><div class="mono">' + self.esc(primary) + '/v1/embeddings</div>' +
          '</div>' +
        '</div>' +
        '<div class="md-footer">' +
          '<button class="btn-sm" id="embed-info-close" style="background:transparent;color:var(--text-secondary);border:1px solid var(--border-hover)">Close</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(modal);
    var close = function () { modal.remove(); };
    modal.querySelector('.md-close').addEventListener('click', close);
    modal.querySelector('#embed-info-close').addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
  },

  _openLoadModelModal() {
    var self = this;
    var existing = document.getElementById('load-model-modal');
    if (existing) existing.remove();

    var modal = document.createElement('div');
    modal.id = 'load-model-modal';
    modal.className = 'model-detail-modal-overlay';
    modal.innerHTML =
      '<div class="model-detail-modal" style="max-width:720px">' +
        '<div class="md-header">' +
          '<div class="md-header-left">' +
            '<div class="md-icon">+</div>' +
            '<div class="md-title">Load Model</div>' +
          '</div>' +
          '<button class="md-close">×</button>' +
        '</div>' +
        '<div class="md-description" style="padding-bottom:0">' +
          '<div class="server-tab-pills" id="load-model-tabs">' +
            '<button class="server-tab-pill active" data-tab="llms">LLMs</button>' +
            '<button class="server-tab-pill" data-tab="embeddings">Embeddings</button>' +
          '</div>' +
          '<div id="load-model-body" style="margin-top:12px;max-height:420px;overflow:auto"></div>' +
        '</div>' +
        '<div class="md-footer">' +
          '<button class="btn-sm" id="load-model-close" style="background:transparent;color:var(--text-secondary);border:1px solid var(--border-hover)">Close</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(modal);
    var close = function () { modal.remove(); };
    modal.querySelector('.md-close').addEventListener('click', close);
    modal.querySelector('#load-model-close').addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });

    var currentTab = 'llms';
    var render = function () { self._renderLoadModelTab(modal, currentTab); };
    modal.querySelectorAll('#load-model-tabs .server-tab-pill').forEach(function (btn) {
      btn.addEventListener('click', function () {
        currentTab = btn.dataset.tab;
        modal.querySelectorAll('#load-model-tabs .server-tab-pill').forEach(function (b) {
          b.classList.toggle('active', b.dataset.tab === currentTab);
        });
        render();
      });
    });
    render();
  },

  async _renderLoadModelTab(modal, tab) {
    var self = this;
    var body = modal.querySelector('#load-model-body');
    if (!body) return;
    body.innerHTML = '<div class="server-empty">Loading…</div>';

    if (tab === 'llms') {
      try {
        var data = await this.fetchJSON('/api/models');
        var models = (data && data.models) || [];
        var downloaded = models.filter(function (m) { return m.downloaded; });
        if (!downloaded.length) {
          body.innerHTML = '<div class="server-empty">No LLMs downloaded yet. Use the Downloads view to get one.</div>';
          return;
        }
        var html = '<div class="server-loaded-list">';
        downloaded.forEach(function (m) {
          var id = m.hf_repo || m.id || m.name || 'unknown';
          html += '<div class="server-loaded-card">' +
            '<div class="server-loaded-left">' +
            '  <span class="server-type-tag">llm</span>' +
            '  <span class="server-model-id mono">' + self.esc(id) + '</span>' +
            '</div>' +
            '<div class="server-loaded-right">' +
            '  <button class="btn-nvidia server-btn-sm" data-action="llm-load" data-model="' + self.esc(id) + '">Load</button>' +
            '</div>' +
            '</div>';
        });
        html += '</div>';
        body.innerHTML = html;
        body.querySelectorAll('[data-action="llm-load"]').forEach(function (btn) {
          btn.addEventListener('click', function () {
            self.toast('LLM load requires a restart with --model ' + btn.dataset.model + ' (coming soon via engine swap)', 'info');
          });
        });
      } catch (err) {
        body.innerHTML = '<div class="server-empty">Failed to load models: ' + self.esc(err.message) + '</div>';
      }
      return;
    }

    // Embeddings tab
    try {
      var edata = await this.fetchJSON('/api/embeddings/models');
      var emodels = (edata && edata.models) || [];
      if (!emodels.length) {
        body.innerHTML = '<div class="server-empty">No embedding models in the catalog.</div>';
        return;
      }
      var ehtml = '<div class="server-loaded-list">';
      emodels.forEach(function (m) {
        var loaded = !!m.loaded;
        ehtml += '<div class="server-loaded-card">' +
          '<div class="server-loaded-left" style="flex-direction:column;align-items:flex-start;gap:4px">' +
          '  <div>' +
          '    <span class="server-type-tag" style="color:var(--cyan);border-color:var(--cyan)">embed</span> ' +
          '    <span class="server-model-id mono">' + self.esc(m.id) + '</span>' +
          '  </div>' +
          '  <div style="font-size:12px;color:var(--text-muted)">' +
          (m.dimensions || '?') + 'd · ' + (m.max_seq_length || '?') + ' ctx · ' + (m.size_mb || '?') + ' MB' +
          '  </div>' +
          '  <div style="font-size:12px;color:var(--text-muted);max-width:480px">' + self.esc(m.description || '') + '</div>' +
          '</div>' +
          '<div class="server-loaded-right">' +
          (loaded
            ? '  <span class="server-badge ready">LOADED</span>'
            : '  <button class="btn-nvidia server-btn-sm" data-action="embed-load" data-model="' + self.esc(m.id) + '">Load</button>'
          ) +
          '</div>' +
          '</div>';
      });
      ehtml += '</div>';
      body.innerHTML = ehtml;
      body.querySelectorAll('[data-action="embed-load"]').forEach(function (btn) {
        btn.addEventListener('click', async function () {
          var id = btn.dataset.model;
          btn.disabled = true;
          btn.textContent = 'Loading…';
          try {
            var resp = await fetch('/api/embeddings/models/' + encodeURIComponent(id) + '/load', { method: 'POST' });
            var payload = await resp.json().catch(function () { return {}; });
            if (resp.ok) {
              self.toast('Loaded ' + id, 'success');
              self._renderLoadModelTab(modal, 'embeddings');
              self.renderServer();
            } else {
              self.toast((payload.error && payload.error.message) || 'Load failed', 'error');
              btn.disabled = false;
              btn.textContent = 'Load';
            }
          } catch (err) {
            self.toast('Load failed: ' + err.message, 'error');
            btn.disabled = false;
            btn.textContent = 'Load';
          }
        });
      });
    } catch (err) {
      body.innerHTML = '<div class="server-empty">Failed to load embeddings: ' + self.esc(err.message) + '</div>';
    }
  },

  async _serverClearLogs() {
    try {
      await fetch('/api/server/logs', { method: 'DELETE' });
      this._serverState.logs = [];
      this._serverState.logsSince = 0;
      var panel = document.getElementById('server-log-panel');
      if (panel) panel.innerHTML = this._renderLogEntries([]);
      this.toast('Logs cleared', 'success');
    } catch (err) {
      this.toast('Failed to clear logs', 'error');
    }
  },

  startServerLogPolling() {
    var self = this;
    if (this._serverState.logsPoll) return;
    this._serverState.logsPoll = setInterval(function () { self._pollServerLogs(); }, 2000);
    this._pollServerLogs();
  },

  stopServerLogPolling() {
    if (this._serverState.logsPoll) {
      clearInterval(this._serverState.logsPoll);
      this._serverState.logsPoll = null;
    }
  },

  async _pollServerLogs() {
    var since = this._serverState.logsSince || 0;
    var data = await this.fetchJSON('/api/server/logs?since=' + since);
    if (!data || !data.entries) return;
    if (data.entries.length > 0) {
      this._serverState.logs = this._serverState.logs.concat(data.entries).slice(-500);
      this._serverState.logsSince = data.entries[data.entries.length - 1].timestamp || data.now;
      var panel = document.getElementById('server-log-panel');
      if (panel) {
        panel.innerHTML = this._renderLogEntries(this._serverState.logs);
        if (this._serverState.autoScroll) panel.scrollTop = panel.scrollHeight;
      }
    } else if (!this._serverState.logsSince) {
      this._serverState.logsSince = data.now || Date.now() / 1000;
    }
  },

  _renderServerRightPanel(modelId) {
    var mount = document.getElementById('right-panel-server');
    if (!mount) return;
    var s = this._serverState.lastStatus || {};
    var models = s.loaded_models || [];
    var model = null;
    if (modelId) model = models.find(function (m) { return m.id === modelId; });
    if (!model && models.length > 0) {
      model = models[0];
      this._serverState.selectedModelId = model.id;
    }
    if (!model) {
      mount.innerHTML = '<div class="panel-section"><h3 class="panel-title">MODEL INFO</h3>' +
        '<div class="server-empty" style="margin:12px 16px">No model selected</div></div>';
      return;
    }
    var self = this;
    var primary = (s.reachable_at && s.reachable_at.length > 1) ? s.reachable_at[1] : (s.reachable_at && s.reachable_at[0]) || '—';
    var arch = (model.id || '').split('/')[0] || 'unknown';
    var fileName = (model.id || '').split('/').pop();
    var sizeStr = model.size_bytes > 0 ? this.formatBytes(model.size_bytes) : '—';

    var html = '';
    html += '<div class="panel-section">';
    html += '  <h3 class="panel-title">MODEL INFO</h3>';
    html += '  <div class="server-info-host">';
    html += '    <span class="label">Hosted on</span>';
    html += '    <span class="mono">' + this.esc(model.node_hostname || '') + '</span>';
    html += '    <button class="server-copy-btn" data-copy="' + this.esc(model.node_hostname || '') + '">⧉</button>';
    html += '  </div>';
    html += '  <div class="server-tab-pills server-info-tabs" id="server-info-tabs">';
    html += '    <button class="server-tab-pill active" data-info-tab="info">Info</button>';
    html += '    <button class="server-tab-pill" data-info-tab="load">Load</button>';
    html += '    <button class="server-tab-pill" data-info-tab="inference">Inference</button>';
    html += '  </div>';
    html += '  <div id="server-info-body">';
    html += this._renderServerInfoTab('info', model, arch, fileName, sizeStr);
    html += '  </div>';
    html += '</div>';

    html += '<div class="panel-section">';
    html += '  <h3 class="panel-title">API USAGE</h3>';
    html += '  <div class="server-info-section">';
    html += '    <div class="server-info-row"><span class="label">Model ID</span><span class="mono val">' + this.esc(model.id) + '</span><button class="server-copy-btn" data-copy="' + this.esc(model.id) + '">⧉</button></div>';
    html += '    <div class="server-info-row"><span class="label">Reachable at</span><span class="mono val">' + this.esc(primary) + '</span><button class="server-copy-btn" data-copy="' + this.esc(primary) + '">⧉</button></div>';
    html += '  </div>';
    html += '</div>';

    mount.innerHTML = html;

    mount.querySelectorAll('[data-copy]').forEach(function (el) {
      el.addEventListener('click', function () {
        var text = el.dataset.copy || '';
        if (!text) return;
        navigator.clipboard.writeText(text).then(function () { self.toast('Copied', 'success'); });
      });
    });
    mount.querySelectorAll('[data-info-tab]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        mount.querySelectorAll('[data-info-tab]').forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        var body = document.getElementById('server-info-body');
        if (body) body.innerHTML = self._renderServerInfoTab(btn.dataset.infoTab, model, arch, fileName, sizeStr);
      });
    });
  },

  _renderServerInfoTab(tab, model, arch, fileName, sizeStr) {
    if (tab === 'load') {
      return '<div class="server-info-section">' +
        '<div class="server-info-row"><span class="label">Context length</span><input class="form-input server-slider-stub" type="number" value="4096" disabled></div>' +
        '<div class="server-info-row"><span class="label">GPU layers</span><input class="form-input server-slider-stub" type="number" value="-1" disabled></div>' +
        '<div class="server-info-row"><span class="label">Parallel</span><input class="form-input server-slider-stub" type="number" value="' + (model.parallel || 1) + '" disabled></div>' +
        '<div class="server-hint">Load parameters are read-only for Docker-managed engines.</div>' +
        '</div>';
    }
    if (tab === 'inference') {
      return '<div class="server-info-section">' +
        '<div class="server-info-row"><span class="label">Temperature</span><input class="form-input server-slider-stub" type="number" step="0.01" value="0.7" disabled></div>' +
        '<div class="server-info-row"><span class="label">Top-p</span><input class="form-input server-slider-stub" type="number" step="0.01" value="0.95" disabled></div>' +
        '<div class="server-info-row"><span class="label">Top-k</span><input class="form-input server-slider-stub" type="number" value="40" disabled></div>' +
        '<div class="server-hint">Override these per-request via the API.</div>' +
        '</div>';
    }
    // Info tab
    var caps = (model.capabilities || []).map(function (c) { return '<span class="server-cap-badge">' + AINode.esc(c) + '</span>'; }).join('');
    return '<div class="server-info-section">' +
      '<div class="server-info-row"><span class="label">Model</span><span class="mono val">' + AINode.esc(model.id) + '</span></div>' +
      '<div class="server-info-row"><span class="label">File</span><span class="mono val">' + AINode.esc(fileName) + '</span></div>' +
      '<div class="server-info-row"><span class="label">Format</span><span class="val">' + AINode.esc(model.format || 'SafeTensors') + '</span></div>' +
      '<div class="server-info-row"><span class="label">Quantization</span><span class="val">' + AINode.esc(model.quantization || 'none') + '</span></div>' +
      '<div class="server-info-row"><span class="label">Arch</span><span class="val">' + AINode.esc(arch) + '</span></div>' +
      '<div class="server-info-row"><span class="label">Capabilities</span><span class="val">' + (caps || '—') + '</span></div>' +
      '<div class="server-info-row"><span class="label">Domain</span><span class="val">' + AINode.esc(model.type || 'llm') + '</span></div>' +
      '<div class="server-info-row"><span class="label">Size on disk</span><span class="val">' + AINode.esc(sizeStr) + '</span></div>' +
      '</div>';
  },
};

// ========================================================================
//  BOOT
// ========================================================================

document.addEventListener('DOMContentLoaded', function () { AINode.init(); });
