/* AINode Dashboard — Single Page Application */

const AINode = {
  state: {
    currentView: 'dashboard',
    status: null,
    nodes: [],
    messages: [],
    streaming: false,
    pollInterval: null,
  },

  // ─── Initialization ────────────────────────────────
  init() {
    this.bindNav();
    this.bindChat();
    this.navigate(window.location.hash.slice(1) || 'dashboard');
    this.startPolling();
  },

  // ─── Navigation ────────────────────────────────────
  bindNav() {
    document.querySelectorAll('.nav-item').forEach(el => {
      el.addEventListener('click', () => {
        const view = el.dataset.view;
        if (view) this.navigate(view);
      });
    });
  },

  navigate(view) {
    this.state.currentView = view;
    window.location.hash = view;

    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(el => {
      el.classList.toggle('active', el.dataset.view === view);
    });

    // Show/hide views
    document.querySelectorAll('.view').forEach(el => {
      el.style.display = el.id === `view-${view}` ? 'block' : 'none';
    });

    // Refresh data for the view
    this.refresh();
  },

  // ─── Data Fetching ─────────────────────────────────
  async fetchJSON(url) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) return null;
      return await resp.json();
    } catch {
      return null;
    }
  },

  async refresh() {
    const [status, nodes] = await Promise.all([
      this.fetchJSON('/api/status'),
      this.fetchJSON('/api/nodes'),
    ]);

    this.state.status = status;
    this.state.nodes = nodes?.nodes || [];

    switch (this.state.currentView) {
      case 'dashboard': this.renderDashboard(); break;
      case 'chat': this.renderChatHeader(); break;
      case 'models': this.renderModels(); break;
      case 'training': this.renderTraining(); break;
    }
  },

  startPolling() {
    this.state.pollInterval = setInterval(() => this.refresh(), 5000);
  },

  // ─── Dashboard View ────────────────────────────────
  renderDashboard() {
    const s = this.state.status;
    const nodes = this.state.nodes;

    if (!s) {
      document.getElementById('dashboard-stats').innerHTML = this.skeletonCards(4);
      return;
    }

    // Stats row
    const totalGPUs = nodes.reduce((sum, n) => sum + (n.gpu_count || 1), 0);
    const totalMem = nodes.reduce((sum, n) => sum + (n.gpu_memory_gb || 0), 0);
    const onlineNodes = nodes.filter(n => n.status === 'online').length;
    const modelsLoaded = s.models_loaded?.length || 0;

    document.getElementById('dashboard-stats').innerHTML = `
      <div class="card">
        <div class="stat-value">${onlineNodes || 1}</div>
        <div class="stat-label">Nodes Online</div>
      </div>
      <div class="card">
        <div class="stat-value">${totalGPUs || 1}</div>
        <div class="stat-label">GPUs</div>
      </div>
      <div class="card">
        <div class="stat-value">${totalMem || (s.gpu?.memory_gb || 0)} GB</div>
        <div class="stat-label">Total Memory</div>
      </div>
      <div class="card">
        <div class="stat-value">${modelsLoaded}</div>
        <div class="stat-label">Models Loaded</div>
      </div>
    `;

    // Node cards
    this.renderNodes(nodes, s);

    // Cluster health
    this.renderClusterHealth(s);
  },

  renderNodes(nodes, status) {
    const container = document.getElementById('dashboard-nodes');
    if (!container) return;

    // If no cluster nodes, show this node
    if (nodes.length === 0 && status) {
      nodes = [{
        node_id: status.node_id || 'local',
        node_name: status.node_name || 'This Node',
        gpu_name: status.gpu?.name || 'Unknown GPU',
        gpu_memory_gb: status.gpu?.memory_gb || 0,
        unified_memory: status.gpu?.unified_memory || false,
        model: status.model || 'none',
        status: status.engine_ready ? 'online' : 'starting',
        is_leader: true,
      }];
    }

    container.innerHTML = nodes.map(node => {
      const statusClass = node.status === 'online' ? 'status-online' :
                          node.status === 'starting' ? 'status-starting' : 'status-offline';
      const leaderClass = node.is_leader ? 'is-leader' : '';
      const memLabel = node.unified_memory ? 'unified' : 'VRAM';
      const memPct = node.gpu_memory_used_pct || 0;
      const barClass = memPct > 90 ? 'red' : memPct > 70 ? 'yellow' : 'green';

      return `
        <div class="node-card ${leaderClass}">
          <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:12px">
            <div>
              <div class="node-name">${this.esc(node.node_name || node.node_id)}</div>
              <div class="node-id">${this.esc(node.node_id)}</div>
            </div>
            <div class="status ${statusClass}">
              <span class="status-dot"></span>
              ${node.status || 'unknown'}
            </div>
          </div>
          <div class="node-gpu">${this.esc(node.gpu_name || 'Unknown')} &middot; ${node.gpu_memory_gb || '?'} GB ${memLabel}</div>
          <div class="node-model">${this.esc(node.model || 'no model')}</div>
          <div class="progress-bar">
            <div class="progress-fill ${barClass}" style="width:${memPct}%"></div>
          </div>
          <div style="font-size:11px; color:var(--text-muted); margin-top:4px">
            Memory: ${memPct}% used
          </div>
        </div>
      `;
    }).join('');
  },

  renderClusterHealth(status) {
    const container = document.getElementById('dashboard-health');
    if (!container) return;

    const uptime = status.uptime_seconds ? this.formatUptime(status.uptime_seconds) : 'N/A';

    container.innerHTML = `
      <div class="card">
        <div class="card-header">
          <div class="card-title">Engine Status</div>
        </div>
        <table class="table">
          <tr>
            <td>Engine</td>
            <td>${status.engine_ready ?
              '<span class="status status-online"><span class="status-dot"></span>Ready</span>' :
              '<span class="status status-starting"><span class="status-dot"></span>Starting</span>'}</td>
          </tr>
          <tr>
            <td>Model</td>
            <td style="font-family:var(--font-mono)">${this.esc(status.model || 'none')}</td>
          </tr>
          <tr>
            <td>API Port</td>
            <td style="font-family:var(--font-mono)">${status.api_port || 8000}</td>
          </tr>
          <tr>
            <td>Uptime</td>
            <td>${uptime}</td>
          </tr>
          <tr>
            <td>Version</td>
            <td>${this.esc(status.version || 'unknown')}</td>
          </tr>
        </table>
      </div>
    `;
  },

  // ─── Chat View ─────────────────────────────────────
  bindChat() {
    const input = document.getElementById('chat-input');
    const send = document.getElementById('chat-send');
    if (!input || !send) return;

    send.addEventListener('click', () => this.sendMessage());
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Auto-resize textarea
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 200) + 'px';
    });
  },

  renderChatHeader() {
    const s = this.state.status;
    const select = document.getElementById('chat-model');
    if (!select || !s) return;

    const models = s.models_loaded || [];
    if (models.length === 0) {
      select.innerHTML = '<option>No models loaded</option>';
    } else {
      select.innerHTML = models.map(m =>
        `<option value="${this.esc(m)}">${this.esc(m)}</option>`
      ).join('');
    }
  },

  async sendMessage() {
    const input = document.getElementById('chat-input');
    const content = input.value.trim();
    if (!content || this.state.streaming) return;

    input.value = '';
    input.style.height = 'auto';

    // Add user message
    this.state.messages.push({ role: 'user', content });
    this.renderMessages();

    // Get model
    const select = document.getElementById('chat-model');
    const model = select?.value || '';

    // Stream response
    this.state.streaming = true;
    document.getElementById('chat-send').disabled = true;

    const assistantMsg = { role: 'assistant', content: '' };
    this.state.messages.push(assistantMsg);

    try {
      const resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: this.state.messages.slice(0, -1).map(m => ({
            role: m.role, content: m.content
          })),
          stream: true,
        }),
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6);
          if (data === '[DONE]') break;

          try {
            const json = JSON.parse(data);
            const delta = json.choices?.[0]?.delta?.content;
            if (delta) {
              assistantMsg.content += delta;
              this.renderMessages();
            }
          } catch { /* skip malformed chunks */ }
        }
      }
    } catch (err) {
      assistantMsg.content = `Error: ${err.message}. Is the engine running?`;
    }

    this.state.streaming = false;
    document.getElementById('chat-send').disabled = false;
    this.renderMessages();
  },

  renderMessages() {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    container.innerHTML = this.state.messages.map(msg => `
      <div class="chat-message ${msg.role}">
        ${this.formatMarkdown(msg.content)}
      </div>
    `).join('');

    container.scrollTop = container.scrollHeight;
  },

  // ─── Models View ───────────────────────────────────
  renderModels() {
    const container = document.getElementById('models-list');
    if (!container) return;

    const s = this.state.status;
    const loaded = s?.models_loaded || [];

    const recommended = [
      { id: 'meta-llama/Llama-3.2-3B-Instruct', size: '~6 GB', desc: 'Quick start, fast inference' },
      { id: 'meta-llama/Llama-3.1-8B-Instruct', size: '~16 GB', desc: 'Recommended for most tasks' },
      { id: 'meta-llama/Llama-3.1-70B-Instruct-AWQ', size: '~35 GB', desc: 'High quality, needs 40+ GB' },
      { id: 'Qwen/Qwen2.5-72B-Instruct', size: '~40 GB', desc: 'Great for coding + multilingual' },
      { id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', size: '~14 GB', desc: 'Reasoning specialist' },
      { id: 'mistralai/Mistral-7B-Instruct-v0.3', size: '~14 GB', desc: 'Fast, general purpose' },
    ];

    container.innerHTML = recommended.map(model => {
      const isLoaded = loaded.includes(model.id);
      return `
        <div class="model-card">
          <div class="model-info">
            <div class="model-name">${this.esc(model.id)}</div>
            <div class="model-meta">${model.size} &middot; ${model.desc}</div>
          </div>
          <div class="model-status">
            ${isLoaded ?
              '<span class="model-badge loaded">Loaded</span>' :
              '<span class="model-badge available">Available</span>'}
          </div>
        </div>
      `;
    }).join('');
  },

  // ─── Training View ─────────────────────────────────
  renderTraining() {
    const container = document.getElementById('training-content');
    if (!container) return;

    container.innerHTML = `
      <div class="training-placeholder">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"/>
        </svg>
        <h3>Fine-Tuning Studio</h3>
        <p>Upload datasets, configure LoRA adapters, and train models — all from your browser.</p>
        <p style="margin-top:16px; font-size:13px">Coming in AINode v0.2</p>
      </div>
    `;
  },

  // ─── Utilities ─────────────────────────────────────
  esc(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  },

  formatUptime(seconds) {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  },

  formatMarkdown(text) {
    if (!text) return '';
    // Basic markdown: code blocks, inline code, bold, newlines
    return text
      .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
  },

  skeletonCards(n) {
    return Array(n).fill(
      '<div class="card"><div class="skeleton" style="height:48px;margin-bottom:8px"></div><div class="skeleton" style="height:14px;width:60%"></div></div>'
    ).join('');
  },
};

// Boot
document.addEventListener('DOMContentLoaded', () => AINode.init());
