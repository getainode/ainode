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
  },

  // --- Initialization ---
  init() {
    this.bindNav();
    this.bindChat();
    this.navigate(window.location.hash.slice(1) || 'dashboard');
    this.startPolling();
  },

  // --- Navigation ---
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
    document.querySelectorAll('.nav-item').forEach(el => {
      el.classList.toggle('active', el.dataset.view === view);
    });
    document.querySelectorAll('.view').forEach(el => {
      el.style.display = el.id === `view-${view}` ? 'block' : 'none';
    });
    this.refresh();
  },

  // --- Data Fetching ---
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

  // --- Dashboard View ---
  renderDashboard() {
    const s = this.state.status;
    const nodes = this.state.nodes;
    if (!s) {
      document.getElementById('dashboard-stats').innerHTML = this.skeletonCards(4);
      return;
    }
    const totalGPUs = nodes.reduce((sum, n) => sum + (n.gpu_count || 1), 0);
    const totalMem = nodes.reduce((sum, n) => sum + (n.gpu_memory_gb || 0), 0);
    const onlineNodes = nodes.filter(n => n.status === 'online').length;
    const modelsLoaded = s.models_loaded?.length || 0;
    document.getElementById('dashboard-stats').innerHTML = `
      <div class="card"><div class="stat-value">${onlineNodes || 1}</div><div class="stat-label">Nodes Online</div></div>
      <div class="card"><div class="stat-value">${totalGPUs || 1}</div><div class="stat-label">GPUs</div></div>
      <div class="card"><div class="stat-value">${totalMem || (s.gpu?.memory_gb || 0)} GB</div><div class="stat-label">Total Memory</div></div>
      <div class="card"><div class="stat-value">${modelsLoaded}</div><div class="stat-label">Models Loaded</div></div>
    `;
    this.renderNodes(nodes, s);
    this.renderClusterHealth(s);
  },

  renderNodes(nodes, status) {
    const container = document.getElementById('dashboard-nodes');
    if (!container) return;
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
      const statusClass = node.status === 'online' ? 'status-online' : node.status === 'starting' ? 'status-starting' : 'status-offline';
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
            <div class="status ${statusClass}"><span class="status-dot"></span>${node.status || 'unknown'}</div>
          </div>
          <div class="node-gpu">${this.esc(node.gpu_name || 'Unknown')} &middot; ${node.gpu_memory_gb || '?'} GB ${memLabel}</div>
          <div class="node-model">${this.esc(node.model || 'no model')}</div>
          <div class="progress-bar"><div class="progress-fill ${barClass}" style="width:${memPct}%"></div></div>
          <div style="font-size:11px; color:var(--text-muted); margin-top:4px">Memory: ${memPct}% used</div>
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
        <div class="card-header"><div class="card-title">Engine Status</div></div>
        <table class="table">
          <tr><td>Engine</td><td>${status.engine_ready ? '<span class="status status-online"><span class="status-dot"></span>Ready</span>' : '<span class="status status-starting"><span class="status-dot"></span>Starting</span>'}</td></tr>
          <tr><td>Model</td><td style="font-family:var(--font-mono)">${this.esc(status.model || 'none')}</td></tr>
          <tr><td>API Port</td><td style="font-family:var(--font-mono)">${status.api_port || 8000}</td></tr>
          <tr><td>Uptime</td><td>${uptime}</td></tr>
          <tr><td>Version</td><td>${this.esc(status.version || 'unknown')}</td></tr>
        </table>
      </div>
    `;
  },

  // --- Chat View ---
  bindChat() {
    const input = document.getElementById('chat-input');
    const send = document.getElementById('chat-send');
    if (!input || !send) return;
    send.addEventListener('click', () => this.sendMessage());
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendMessage(); }
    });
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
    if (models.length === 0) { select.innerHTML = '<option>No models loaded</option>'; }
    else { select.innerHTML = models.map(m => `<option value="${this.esc(m)}">${this.esc(m)}</option>`).join(''); }
  },

  async sendMessage() {
    const input = document.getElementById('chat-input');
    const content = input.value.trim();
    if (!content || this.state.streaming) return;
    input.value = ''; input.style.height = 'auto';
    this.state.messages.push({ role: 'user', content });
    this.renderMessages();
    const select = document.getElementById('chat-model');
    const model = select?.value || '';
    this.state.streaming = true;
    document.getElementById('chat-send').disabled = true;
    const assistantMsg = { role: 'assistant', content: '' };
    this.state.messages.push(assistantMsg);
    try {
      const resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages: this.state.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })), stream: true }),
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
          try { const json = JSON.parse(data); const delta = json.choices?.[0]?.delta?.content; if (delta) { assistantMsg.content += delta; this.renderMessages(); } } catch { }
        }
      }
    } catch (err) { assistantMsg.content = 'Error: ' + err.message + '. Is the engine running?'; }
    this.state.streaming = false;
    document.getElementById('chat-send').disabled = false;
    this.renderMessages();
  },

  renderMessages() {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    container.innerHTML = this.state.messages.map(msg => `<div class="chat-message ${msg.role}">${this.formatMarkdown(msg.content)}</div>`).join('');
    container.scrollTop = container.scrollHeight;
  },

  // --- Models View ---
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
            ${isLoaded ? '<span class="model-badge loaded">Loaded</span>' : '<span class="model-badge available">Available</span>'}
          </div>
        </div>
      `;
    }).join('');
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
    const container = document.getElementById('training-content');
    if (!container) return;
    const data = await this.fetchJSON('/api/training/jobs');
    this.state.trainingJobs = data?.jobs || [];
    switch (this.state.trainingView) {
      case 'list': this.renderTrainingList(container); break;
      case 'new': this.renderTrainingForm(container); break;
      case 'detail': await this.renderTrainingDetail(container); break;
    }
  },

  renderTrainingList(container) {
    const jobs = this.state.trainingJobs;
    const hasJobs = jobs.length > 0;
    container.innerHTML = '<div class="training-header">' +
      '<div class="training-stat-row">' +
        '<div class="card training-stat-card"><div class="stat-value">' + jobs.length + '</div><div class="stat-label">Total Jobs</div></div>' +
        '<div class="card training-stat-card"><div class="stat-value">' + jobs.filter(j => j.status === 'running').length + '</div><div class="stat-label">Running</div></div>' +
        '<div class="card training-stat-card"><div class="stat-value">' + jobs.filter(j => j.status === 'completed').length + '</div><div class="stat-label">Completed</div></div>' +
        '<div class="card training-stat-card"><div class="stat-value">' + jobs.filter(j => j.status === 'pending').length + '</div><div class="stat-label">Queued</div></div>' +
      '</div>' +
      '<button class="btn btn-primary" id="training-new-btn">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>' +
        'New Training Job</button>' +
    '</div>' +
    (hasJobs ?
      '<div class="training-jobs-list">' + jobs.sort((a, b) => (b.start_time || 0) - (a.start_time || 0)).map(job => this.renderJobCard(job)).join('') + '</div>'
      :
      '<div class="training-empty">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="56" height="56"><path d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"/></svg>' +
        '<h3>No training jobs yet</h3><p>Create your first fine-tuning job to get started.</p>' +
      '</div>'
    );
    document.getElementById('training-new-btn')?.addEventListener('click', () => { this.state.trainingView = 'new'; this.renderTraining(); });
    container.querySelectorAll('.training-job-card').forEach(card => {
      card.addEventListener('click', () => { this.state.trainingView = 'detail'; this.state.trainingDetailId = card.dataset.jobId; this.state.trainingLossData = []; this.renderTraining(); });
    });
  },

  renderJobCard(job) {
    const statusMap = { pending: { cls: 'status-pending', label: 'Pending' }, running: { cls: 'status-running', label: 'Running' }, completed: { cls: 'status-completed', label: 'Completed' }, failed: { cls: 'status-failed', label: 'Failed' }, cancelled: { cls: 'status-cancelled', label: 'Cancelled' } };
    const s = statusMap[job.status] || { cls: '', label: job.status };
    const modelShort = job.config?.base_model?.split('/').pop() || 'Unknown';
    const methodLabel = (job.config?.method || 'lora').toUpperCase();
    const started = job.start_time ? new Date(job.start_time * 1000).toLocaleString() : 'Not started';
    const elapsed = job.elapsed_seconds ? this.formatUptime(Math.round(job.elapsed_seconds)) : '--';
    let html = '<div class="training-job-card" data-job-id="' + this.esc(job.job_id) + '">' +
      '<div class="job-card-top"><div class="job-card-left">' +
        '<div class="job-card-model">' + this.esc(modelShort) + '</div>' +
        '<div class="job-card-meta"><span class="job-method-badge">' + methodLabel + '</span><span class="job-card-id">' + this.esc(job.job_id) + '</span></div>' +
      '</div><div class="job-card-right"><span class="job-status-badge ' + s.cls + '">' + s.label + '</span></div></div>';
    if (job.status === 'running') {
      html += '<div class="job-card-progress">' +
        '<div class="progress-bar training-progress-bar"><div class="progress-fill accent training-progress-animated" style="width:' + (job.progress || 0) + '%"></div></div>' +
        '<div class="job-progress-text">' + (job.progress || 0).toFixed(1) + '% &middot; Epoch ' + (job.current_epoch || 0) + '/' + (job.config?.num_epochs || '?') + (job.current_loss != null ? ' &middot; Loss: ' + job.current_loss.toFixed(4) : '') + '</div></div>';
    }
    html += '<div class="job-card-footer"><span>Started: ' + started + '</span><span>Duration: ' + elapsed + '</span></div></div>';
    return html;
  },

  renderTrainingForm(container) {
    const models = this.trainingModels;
    const optionsHtml = models.map(m => '<option value="' + this.esc(m.id) + '">' + this.esc(m.name) + ' (' + m.size + ')</option>').join('');
    container.innerHTML = '<div class="training-form-wrapper">' +
      '<div class="training-form-header">' +
        '<button class="btn btn-ghost" id="training-back-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polyline points="15 18 9 12 15 6"/></svg>Back to Jobs</button>' +
        '<h2 class="training-form-title">New Training Job</h2>' +
      '</div>' +
      '<form id="training-form" class="training-form">' +
        '<div class="form-section"><div class="form-section-title">Model</div>' +
          '<div class="form-group"><label class="form-label" for="train-model">Base Model</label><select id="train-model" class="form-select" required>' + optionsHtml + '</select><div class="form-hint">Or enter a custom HuggingFace model ID below</div></div>' +
          '<div class="form-group"><label class="form-label" for="train-model-custom">Custom Model ID (optional)</label><input type="text" id="train-model-custom" class="form-input" placeholder="e.g. org/my-model"></div>' +
        '</div>' +
        '<div class="form-section"><div class="form-section-title">Dataset</div>' +
          '<div class="form-group"><label class="form-label" for="train-dataset">Dataset Path</label><input type="text" id="train-dataset" class="form-input" placeholder="/path/to/dataset.jsonl" required><div class="form-hint">Path to a JSONL file on this machine. Each line should have &quot;instruction&quot; and &quot;output&quot; fields.</div></div>' +
        '</div>' +
        '<div class="form-section"><div class="form-section-title">Training Method</div>' +
          '<div class="form-group"><div class="method-toggle">' +
            '<button type="button" class="method-btn active" data-method="lora"><div class="method-btn-title">LoRA</div><div class="method-btn-desc">Recommended. Trains adapter weights only. Fast, memory-efficient.</div></button>' +
            '<button type="button" class="method-btn" data-method="full"><div class="method-btn-title">Full Fine-Tune</div><div class="method-btn-desc">Updates all model weights. Requires more VRAM and time.</div></button>' +
          '</div><input type="hidden" id="train-method" value="lora"></div>' +
        '</div>' +
        '<div class="form-section"><div class="form-section-title collapsible" id="advanced-toggle"><span>Advanced Settings</span><svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polyline points="6 9 12 15 18 9"/></svg></div>' +
          '<div class="advanced-settings collapsed" id="advanced-settings">' +
            '<div class="form-grid">' +
              '<div class="form-group"><label class="form-label" for="train-epochs">Epochs</label><input type="number" id="train-epochs" class="form-input" value="3" min="1" max="100"></div>' +
              '<div class="form-group"><label class="form-label" for="train-batch">Batch Size</label><input type="number" id="train-batch" class="form-input" value="4" min="1" max="128"></div>' +
              '<div class="form-group"><label class="form-label" for="train-lr">Learning Rate</label><input type="text" id="train-lr" class="form-input" value="2e-4"></div>' +
              '<div class="form-group"><label class="form-label" for="train-seq-len">Max Sequence Length</label><input type="number" id="train-seq-len" class="form-input" value="2048" min="128" max="32768" step="128"></div>' +
            '</div>' +
            '<div class="form-grid lora-settings" id="lora-settings">' +
              '<div class="form-group"><label class="form-label" for="train-lora-rank">LoRA Rank</label><input type="number" id="train-lora-rank" class="form-input" value="16" min="1" max="256"><div class="form-hint">Higher = more capacity, more memory</div></div>' +
              '<div class="form-group"><label class="form-label" for="train-lora-alpha">LoRA Alpha</label><input type="number" id="train-lora-alpha" class="form-input" value="32" min="1" max="512"><div class="form-hint">Typically 2x the rank</div></div>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="form-actions"><button type="button" class="btn btn-ghost" id="training-cancel-btn">Cancel</button><button type="submit" class="btn btn-primary btn-lg" id="training-submit-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><polygon points="5 3 19 12 5 21 5 3"/></svg>Start Training</button></div>' +
      '</form></div>';
    this.bindTrainingForm();
  },

  bindTrainingForm() {
    const goBack = () => { this.state.trainingView = 'list'; this.renderTraining(); };
    document.getElementById('training-back-btn')?.addEventListener('click', goBack);
    document.getElementById('training-cancel-btn')?.addEventListener('click', goBack);
    document.querySelectorAll('.method-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.method-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('train-method').value = btn.dataset.method;
        const ls = document.getElementById('lora-settings');
        if (ls) ls.style.display = btn.dataset.method === 'lora' ? '' : 'none';
      });
    });
    document.getElementById('advanced-toggle')?.addEventListener('click', () => {
      const s = document.getElementById('advanced-settings');
      const t = document.getElementById('advanced-toggle');
      if (s) { s.classList.toggle('collapsed'); t.classList.toggle('open'); }
    });
    document.getElementById('training-form')?.addEventListener('submit', async (e) => { e.preventDefault(); await this.submitTrainingJob(); });
  },

  async submitTrainingJob() {
    const submitBtn = document.getElementById('training-submit-btn');
    if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Submitting...'; }
    const customModel = document.getElementById('train-model-custom')?.value?.trim();
    const baseModel = customModel || document.getElementById('train-model')?.value;
    const method = document.getElementById('train-method')?.value || 'lora';
    const payload = {
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
      const resp = await fetch('/api/training/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const result = await resp.json();
      if (!resp.ok) { alert('Error: ' + (result.error || 'Failed to create job')); if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Start Training'; } return; }
      this.state.trainingView = 'detail'; this.state.trainingDetailId = result.job_id; this.state.trainingLossData = []; this.renderTraining();
    } catch (err) { alert('Network error: ' + err.message); if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Start Training'; } }
  },

  async renderTrainingDetail(container) {
    const jobId = this.state.trainingDetailId;
    if (!jobId) { this.state.trainingView = 'list'; this.renderTraining(); return; }
    const jobData = await this.fetchJSON('/api/training/jobs/' + jobId);
    if (!jobData) { container.innerHTML = '<div class="training-empty"><h3>Job not found</h3></div>'; return; }
    const logsData = await this.fetchJSON('/api/training/jobs/' + jobId + '/logs?tail=200');
    const logs = logsData?.logs || [];
    if (jobData.current_loss != null && jobData.progress > 0) {
      const lastPoint = this.state.trainingLossData[this.state.trainingLossData.length - 1];
      if (!lastPoint || lastPoint.progress !== jobData.progress) {
        this.state.trainingLossData.push({ progress: jobData.progress, loss: jobData.current_loss, epoch: jobData.current_epoch });
      }
    }
    if (this.state.trainingLossData.length === 0) {
      for (const line of logs) {
        const marker = 'AINODE_PROGRESS:'; const idx = line.indexOf(marker);
        if (idx !== -1) { try { const p = JSON.parse(line.slice(idx + marker.length)); if (p.loss != null && p.progress != null) { this.state.trainingLossData.push({ progress: p.progress, loss: p.loss, epoch: p.epoch || 0 }); } } catch { } }
      }
    }
    const statusMap = { pending: { cls: 'status-pending', label: 'Pending' }, running: { cls: 'status-running', label: 'Running' }, completed: { cls: 'status-completed', label: 'Completed' }, failed: { cls: 'status-failed', label: 'Failed' }, cancelled: { cls: 'status-cancelled', label: 'Cancelled' } };
    const s = statusMap[jobData.status] || { cls: '', label: jobData.status };
    const cfg = jobData.config || {};
    const modelShort = cfg.base_model?.split('/').pop() || 'Unknown';
    const started = jobData.start_time ? new Date(jobData.start_time * 1000).toLocaleString() : 'Not started';
    const ended = jobData.end_time ? new Date(jobData.end_time * 1000).toLocaleString() : '--';
    const elapsed = jobData.elapsed_seconds ? this.formatUptime(Math.round(jobData.elapsed_seconds)) : '--';
    const isActive = jobData.status === 'running' || jobData.status === 'pending';
    let html = '<div class="training-detail">' +
      '<div class="training-form-header">' +
        '<button class="btn btn-ghost" id="training-back-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polyline points="15 18 9 12 15 6"/></svg>Back to Jobs</button>' +
        '<div class="training-detail-title"><h2>' + this.esc(modelShort) + '</h2><span class="job-status-badge ' + s.cls + '">' + s.label + '</span></div>' +
      '</div>';
    if (jobData.status === 'running') {
      html += '<div class="training-detail-progress card">' +
        '<div class="progress-header"><span class="progress-pct">' + (jobData.progress || 0).toFixed(1) + '%</span><span class="progress-epoch">Epoch ' + (jobData.current_epoch || 0) + ' / ' + (cfg.num_epochs || '?') + '</span></div>' +
        '<div class="progress-bar training-progress-bar-lg"><div class="progress-fill accent training-progress-animated" style="width:' + (jobData.progress || 0) + '%"></div></div>' +
        '<div class="progress-stats">' + (jobData.current_loss != null ? '<span>Loss: <strong>' + jobData.current_loss.toFixed(4) + '</strong></span>' : '') + '<span>Elapsed: <strong>' + elapsed + '</strong></span></div></div>';
    }
    html += '<div class="training-detail-grid">' +
      '<div class="card training-config-card"><div class="card-header"><div class="card-title">Configuration</div></div><table class="table">' +
        '<tr><td>Model</td><td style="font-family:var(--font-mono)">' + this.esc(cfg.base_model || '') + '</td></tr>' +
        '<tr><td>Method</td><td>' + (cfg.method || 'lora').toUpperCase() + '</td></tr>' +
        '<tr><td>Dataset</td><td style="font-family:var(--font-mono);word-break:break-all">' + this.esc(cfg.dataset_path || '') + '</td></tr>' +
        '<tr><td>Epochs</td><td>' + (cfg.num_epochs || '?') + '</td></tr>' +
        '<tr><td>Batch Size</td><td>' + (cfg.batch_size || '?') + '</td></tr>' +
        '<tr><td>Learning Rate</td><td>' + (cfg.learning_rate || '?') + '</td></tr>' +
        '<tr><td>Max Seq Length</td><td>' + (cfg.max_seq_length || '?') + '</td></tr>' +
        (cfg.method === 'lora' ? '<tr><td>LoRA Rank</td><td>' + (cfg.lora_rank || '?') + '</td></tr><tr><td>LoRA Alpha</td><td>' + (cfg.lora_alpha || '?') + '</td></tr>' : '') +
        '<tr><td>Job ID</td><td style="font-family:var(--font-mono)">' + this.esc(jobData.job_id) + '</td></tr>' +
        '<tr><td>Started</td><td>' + started + '</td></tr><tr><td>Ended</td><td>' + ended + '</td></tr><tr><td>Duration</td><td>' + elapsed + '</td></tr>' +
      '</table>' + (isActive ? '<div style="margin-top:16px"><button class="btn btn-danger" id="training-cancel-job-btn">Cancel Job</button></div>' : '') + '</div>' +
      '<div class="training-right-col">' +
        (this.state.trainingLossData.length > 1 ? '<div class="card training-chart-card"><div class="card-header"><div class="card-title">Training Loss</div></div><div class="loss-chart-container"><canvas id="loss-chart" width="460" height="200"></canvas></div></div>' : '') +
        '<div class="card training-log-card"><div class="card-header"><div class="card-title">Logs</div><span class="log-line-count">' + (logsData?.total_lines || 0) + ' lines</span></div>' +
        '<div class="training-log-viewer" id="training-log-viewer">' + (logs.length > 0 ? logs.map(l => '<div class="log-line">' + this.esc(l) + '</div>').join('') : '<div class="log-empty">No logs yet</div>') + '</div></div>' +
      '</div></div></div>';
    container.innerHTML = html;
    document.getElementById('training-back-btn')?.addEventListener('click', () => { this.state.trainingView = 'list'; this.state.trainingDetailId = null; this.state.trainingLossData = []; this.renderTraining(); });
    document.getElementById('training-cancel-job-btn')?.addEventListener('click', async () => {
      if (!confirm('Cancel this training job?')) return;
      const resp = await fetch('/api/training/jobs/' + jobId, { method: 'DELETE' });
      if (resp.ok) { this.renderTraining(); } else { const err = await resp.json().catch(() => ({})); alert(err.error || 'Failed to cancel job'); }
    });
    const logViewer = document.getElementById('training-log-viewer');
    if (logViewer) logViewer.scrollTop = logViewer.scrollHeight;
    if (this.state.trainingLossData.length > 1) this.drawLossChart();
  },

  drawLossChart() {
    const canvas = document.getElementById('loss-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr; canvas.height = rect.height * dpr; ctx.scale(dpr, dpr);
    const w = rect.width, h = rect.height;
    const pad = { top: 12, right: 16, bottom: 28, left: 48 };
    const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;
    const data = this.state.trainingLossData;
    if (data.length < 2) return;
    const losses = data.map(d => d.loss);
    const maxLoss = Math.max(...losses) * 1.05, minLoss = Math.min(...losses) * 0.95;
    const maxProg = Math.max(...data.map(d => d.progress), 1);
    const toX = (prog) => pad.left + (prog / maxProg) * plotW;
    const toY = (loss) => pad.top + ((maxLoss - loss) / (maxLoss - minLoss || 1)) * plotH;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(45, 55, 72, 0.5)'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      const val = maxLoss - ((maxLoss - minLoss) / 4) * i;
      ctx.fillStyle = '#64748b'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right'; ctx.fillText(val.toFixed(3), pad.left - 6, y + 3);
    }
    ctx.fillStyle = '#64748b'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('0%', pad.left, h - 6); ctx.fillText(maxProg.toFixed(0) + '%', w - pad.right, h - 6);
    ctx.strokeStyle = '#4a90d9'; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.beginPath();
    data.forEach((d, i) => { const x = toX(d.progress), y = toY(d.loss); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
    const gradient = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
    gradient.addColorStop(0, 'rgba(74, 144, 217, 0.15)'); gradient.addColorStop(1, 'rgba(74, 144, 217, 0)');
    ctx.fillStyle = gradient; ctx.beginPath();
    data.forEach((d, i) => { const x = toX(d.progress), y = toY(d.loss); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.lineTo(toX(data[data.length - 1].progress), h - pad.bottom); ctx.lineTo(toX(data[0].progress), h - pad.bottom); ctx.closePath(); ctx.fill();
    const last = data[data.length - 1];
    ctx.fillStyle = '#4a90d9'; ctx.beginPath(); ctx.arc(toX(last.progress), toY(last.loss), 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#0a0e17'; ctx.lineWidth = 2; ctx.stroke();
  },

  // --- Utilities ---
  esc(str) { const div = document.createElement('div'); div.textContent = str || ''; return div.innerHTML; },
  formatUptime(seconds) { if (seconds < 60) return seconds + 's'; if (seconds < 3600) return Math.floor(seconds / 60) + 'm'; return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm'; },
  formatMarkdown(text) { if (!text) return ''; return text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>').replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>'); },
  skeletonCards(n) { return Array(n).fill('<div class="card"><div class="skeleton" style="height:48px;margin-bottom:8px"></div><div class="skeleton" style="height:14px;width:60%"></div></div>').join(''); },
};

document.addEventListener('DOMContentLoaded', () => AINode.init());
