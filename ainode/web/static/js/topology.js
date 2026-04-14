/* AINode Topology — NVIDIA-Styled Orbital Planet Visualization
 * Master node at center with pulsing rings + orbiting scan-dots.
 * Worker nodes orbit the master as planets with inward data pulses.
 */

class TopologyRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.width = 0;
    this.height = 0;
    this.time = 0;
    this.lastTs = 0;
    this.dt = 16;
    this.animFrameId = null;

    // Graph state
    this.nodes = [];
    this.edges = [];
    this.masterId = null;

    // Interaction state
    this.mouse = { x: 0, y: 0, down: false };
    this.dragNode = null;
    this.dragOffset = { x: 0, y: 0 };
    this.dragStartPos = { x: 0, y: 0 };
    this.hoveredNodeId = null;
    this.selectedNodeId = null;
    this.lastClickTime = 0;
    this.lastClickId = null;

    // Callbacks
    this.onNodeSelect = null;

    // Data pulse particles
    this.pulses = [];
    this.edgePulseTimers = {}; // edgeId -> ms accumulator

    // Config
    this.cfg = {
      masterRadius: 42,
      workerRadius: 32,
      orbitRadiusRatio: 0.32,
      orbitAngularVelocity: 0.0008, // rad/ms (counterclockwise)
      minHeight: 360,
      // Colors
      nvidiaGreen: '#76B900',
      nvidiaGreenBright: '#8ACE00',
      nodeBg: '#0a0a0a',
      nodeBgInner: '#141414',
      nodeBorder: '#76B900',
      nodeBorderStale: '#4a6b00',
      nodeBorderOffline: '#553333',
      textPrimary: '#E0E0E0',
      textSecondary: '#AAAAAA',
      textMuted: '#888888',
      trackBg: '#1f1f1f',
      dotGrid: '#141414',
      tempGreen: '#76B900',
      tempAmber: '#F5A623',
      tempRed: '#FF4444',
      pulseIntervalMs: 800,
      pulseDurationMs: 1500,
    };

    this._boundHandlers = {};
    this._resize();
    this._bindEvents();
    this._startLoop();
  }

  // --- Public API ---

  update(nodesArray) {
    if (!nodesArray || !Array.isArray(nodesArray)) return;

    const existingMap = {};
    this.nodes.forEach(n => { existingMap[n.id] = n; });

    // Determine master
    let masterNode = nodesArray.find(nd => nd.effective_role === 'master');
    if (!masterNode) masterNode = nodesArray.find(nd => nd.is_leader);
    const masterId = masterNode ? (masterNode.node_id || masterNode.id || 'local') : null;
    this.masterId = masterId;

    const cx = this.width / 2;
    const cy = this.height / 2;
    const orbitR = Math.min(this.width, this.height) * this.cfg.orbitRadiusRatio;

    const workers = nodesArray.filter(nd => (nd.node_id || nd.id || 'local') !== masterId);
    const workerCount = workers.length;

    const updatedNodes = [];
    let workerIdx = 0;

    nodesArray.forEach((nd) => {
      const id = nd.node_id || nd.id || 'local';
      let gn = existingMap[id];
      const isMaster = id === masterId;
      if (gn) {
        gn.data = nd;
        gn.isMaster = isMaster;
        gn.targetAlpha = 1;
      } else {
        gn = {
          id,
          data: nd,
          isMaster,
          x: cx,
          y: cy,
          orbitAngle: 0,
          pinned: false,
          alpha: 0,
          targetAlpha: 1,
          scale: 1,
          targetScale: 1,
        };
        // Initial angle for new worker
        if (!isMaster && workerCount > 0) {
          const i = workers.findIndex(w => (w.node_id || w.id || 'local') === id);
          gn.orbitAngle = (i / workerCount) * Math.PI * 2 - Math.PI / 2;
          gn.x = cx + Math.cos(gn.orbitAngle) * orbitR;
          gn.y = cy + Math.sin(gn.orbitAngle) * orbitR;
        }
      }
      if (!isMaster) workerIdx++;
      updatedNodes.push(gn);
    });

    this.nodes = updatedNodes;
    this._buildEdges();
  }

  destroy() {
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    this.animFrameId = null;
    this._unbindEvents();
  }

  resize() {
    this._resize();
  }

  getSelectedNodeId() {
    return this.selectedNodeId;
  }

  // --- Layout ---

  _buildEdges() {
    this.edges = [];
    const master = this.nodes.find(n => n.isMaster);
    if (!master) return;
    this.nodes.forEach(n => {
      if (n === master) return;
      const active = n.data.status === 'online' && master.data.status === 'online';
      const eid = master.id + '::' + n.id;
      this.edges.push({
        id: eid,
        source: n,      // worker
        target: master, // master (data flows worker -> master)
        active,
      });
    });
  }

  // --- Resize ---

  _resize() {
    if (!this.canvas || !this.canvas.parentElement) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width;
    this.height = Math.max(rect.height, this.cfg.minHeight);
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.canvas.style.width = this.width + 'px';
    this.canvas.style.height = this.height + 'px';
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  // --- Events ---

  _bindEvents() {
    const h = this._boundHandlers;

    h.mouseMove = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.mouse.x = e.clientX - rect.left;
      this.mouse.y = e.clientY - rect.top;
      if (this.dragNode) {
        this.dragNode.x = this.mouse.x - this.dragOffset.x;
        this.dragNode.y = this.mouse.y - this.dragOffset.y;
        this.dragNode.pinned = true;
        // Update orbit angle so if unpinned, it stays in place
        if (!this.dragNode.isMaster) {
          const master = this.nodes.find(nn => nn.isMaster);
          if (master) {
            this.dragNode.orbitAngle = Math.atan2(
              this.dragNode.y - master.y,
              this.dragNode.x - master.x
            );
          }
        }
      }
      this._updateHover();
    };

    h.mouseDown = (e) => {
      this.mouse.down = true;
      const node = this._getNodeAt(this.mouse.x, this.mouse.y);
      if (node) {
        this.dragNode = node;
        this.dragOffset.x = this.mouse.x - node.x;
        this.dragOffset.y = this.mouse.y - node.y;
        this.dragStartPos.x = this.mouse.x;
        this.dragStartPos.y = this.mouse.y;
        this.canvas.style.cursor = 'grabbing';
      }
    };

    h.mouseUp = () => {
      if (this.dragNode) {
        const dx = this.mouse.x - this.dragStartPos.x;
        const dy = this.mouse.y - this.dragStartPos.y;
        if (Math.abs(dx) < 4 && Math.abs(dy) < 4) {
          // Click — select
          this._selectNode(this.dragNode.id);
          // Don't pin on plain click
          this.dragNode.pinned = false;
        }
        this.dragNode = null;
        this.canvas.style.cursor = 'default';
      }
      this.mouse.down = false;
    };

    h.dblClick = () => {
      const node = this._getNodeAt(this.mouse.x, this.mouse.y);
      if (node) {
        node.pinned = false;
      }
    };

    h.click = () => {
      const node = this._getNodeAt(this.mouse.x, this.mouse.y);
      if (!node && !this.dragNode) {
        this._selectNode(null);
      }
    };

    h.mouseLeave = () => {
      this.hoveredNodeId = null;
      if (this.dragNode) { this.dragNode.pinned = false; this.dragNode = null; }
      this.mouse.down = false;
      this.canvas.style.cursor = 'default';
    };

    h.resize = () => {
      this._resize();
    };

    this.canvas.addEventListener('mousemove', h.mouseMove);
    this.canvas.addEventListener('mousedown', h.mouseDown);
    this.canvas.addEventListener('mouseup', h.mouseUp);
    this.canvas.addEventListener('mouseleave', h.mouseLeave);
    this.canvas.addEventListener('click', h.click);
    this.canvas.addEventListener('dblclick', h.dblClick);
    window.addEventListener('resize', h.resize);
  }

  _unbindEvents() {
    if (!this.canvas) return;
    const h = this._boundHandlers;
    this.canvas.removeEventListener('mousemove', h.mouseMove);
    this.canvas.removeEventListener('mousedown', h.mouseDown);
    this.canvas.removeEventListener('mouseup', h.mouseUp);
    this.canvas.removeEventListener('mouseleave', h.mouseLeave);
    this.canvas.removeEventListener('click', h.click);
    this.canvas.removeEventListener('dblclick', h.dblClick);
    window.removeEventListener('resize', h.resize);
  }

  _updateHover() {
    const node = this._getNodeAt(this.mouse.x, this.mouse.y);
    const newId = node ? node.id : null;
    if (newId !== this.hoveredNodeId) {
      this.hoveredNodeId = newId;
      this.canvas.style.cursor = newId ? 'grab' : 'default';
      if (this.dragNode) this.canvas.style.cursor = 'grabbing';
    }
    this.nodes.forEach(n => {
      n.targetScale = (n.id === this.hoveredNodeId) ? 1.08 : 1.0;
    });
  }

  _getNodeAt(mx, my) {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      const r = (n.isMaster ? this.cfg.masterRadius : this.cfg.workerRadius) * n.scale;
      const dx = mx - n.x;
      const dy = my - n.y;
      if (dx * dx + dy * dy <= r * r) {
        return n;
      }
    }
    return null;
  }

  _selectNode(id) {
    this.selectedNodeId = id;
    if (this.onNodeSelect) {
      const node = this.nodes.find(n => n.id === id);
      this.onNodeSelect(node ? node.data : null);
    }
  }

  // --- Simulation (orbital) ---

  _simulate() {
    const master = this.nodes.find(n => n.isMaster);
    const cx = this.width / 2;
    const cy = this.height / 2;
    const orbitR = Math.min(this.width, this.height) * this.cfg.orbitRadiusRatio;

    // Master snaps to center (unless pinned)
    if (master) {
      if (!master.pinned) {
        // Gentle breathing
        const bx = Math.sin(this.time * 0.7) * 0.8;
        const by = Math.cos(this.time * 0.5) * 0.8;
        master.x += ((cx + bx) - master.x) * 0.15;
        master.y += ((cy + by) - master.y) * 0.15;
      }
      master.alpha += (master.targetAlpha - master.alpha) * 0.05;
      master.scale += (master.targetScale - master.scale) * 0.1;
    }

    const mx = master ? master.x : cx;
    const my = master ? master.y : cy;

    this.nodes.forEach(n => {
      if (n.isMaster) return;
      if (!n.pinned) {
        n.orbitAngle += this.cfg.orbitAngularVelocity * this.dt;
        const tx = mx + Math.cos(n.orbitAngle) * orbitR;
        const ty = my + Math.sin(n.orbitAngle) * orbitR;
        // Smooth approach
        n.x += (tx - n.x) * 0.1;
        n.y += (ty - n.y) * 0.1;
      }
      n.alpha += (n.targetAlpha - n.alpha) * 0.05;
      n.scale += (n.targetScale - n.scale) * 0.1;
    });
  }

  // --- Pulse Particles ---

  _updatePulses() {
    const now = performance.now();
    // Per-edge spawn timer
    this.edges.forEach(edge => {
      if (!edge.active) return;
      const prev = this.edgePulseTimers[edge.id] || 0;
      if (now - prev >= this.cfg.pulseIntervalMs) {
        this.edgePulseTimers[edge.id] = now;
        this.pulses.push({
          edge,
          startTime: now,
          duration: this.cfg.pulseDurationMs,
          size: 2.2 + Math.random() * 1.2,
        });
      }
    });
    // Cull finished
    this.pulses = this.pulses.filter(p => now - p.startTime < p.duration);
  }

  // --- Drawing ---

  _draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    this._drawDotGrid();
    this._drawEdges();
    this._drawPulses();

    // Draw master last orbit effects behind its own circle
    const master = this.nodes.find(n => n.isMaster);
    if (master) this._drawMasterOrbitEffects(master);

    // Workers first, then master on top
    const workers = this.nodes.filter(n => !n.isMaster);
    const sorted = workers.sort((a, b) => {
      if (a.id === this.selectedNodeId) return 1;
      if (b.id === this.selectedNodeId) return -1;
      return 0;
    });
    sorted.forEach(n => this._drawNode(n));
    if (master) this._drawNode(master);

    // "Waiting for peers..." when alone
    if (master && workers.length === 0) {
      const ctx2 = this.ctx;
      ctx2.save();
      const textAlpha = 0.4 + Math.sin(this.time * 2) * 0.2;
      ctx2.font = '12px Inter, -apple-system, sans-serif';
      ctx2.fillStyle = this._rgba(this.cfg.textMuted, textAlpha);
      ctx2.textAlign = 'center';
      ctx2.textBaseline = 'top';
      ctx2.fillText('Waiting for peers…', master.x, master.y + this.cfg.masterRadius + 70);
      ctx2.restore();
    }

    // Hover tooltip
    if (this.hoveredNodeId) {
      const hn = this.nodes.find(n => n.id === this.hoveredNodeId);
      if (hn) this._drawTooltip(hn);
    }
  }

  _drawDotGrid() {
    const ctx = this.ctx;
    const spacing = 24;
    ctx.fillStyle = this.cfg.dotGrid;
    ctx.globalAlpha = 0.6;
    for (let x = spacing; x < this.width; x += spacing) {
      for (let y = spacing; y < this.height; y += spacing) {
        ctx.fillRect(x, y, 1, 1);
      }
    }
    ctx.globalAlpha = 1;
  }

  _drawEdges() {
    const ctx = this.ctx;
    this.edges.forEach(edge => {
      const a = edge.source;   // worker
      const b = edge.target;   // master
      const alpha = Math.min(a.alpha, b.alpha);
      if (alpha < 0.01) return;

      ctx.save();
      ctx.globalAlpha = alpha;

      const isHoverEdge = this.hoveredNodeId === a.id;
      const baseOpacity = edge.active ? 0.35 : 0.18;
      const opacity = isHoverEdge ? Math.min(1, baseOpacity + 0.35) : baseOpacity;
      ctx.strokeStyle = this._rgba(edge.active ? this.cfg.nvidiaGreen : this.cfg.nodeBorderStale, opacity);
      ctx.lineWidth = isHoverEdge ? 2 : 1.2;

      if (!edge.active) ctx.setLineDash([6, 6]);

      // Trim line so it doesn't draw over node circles
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const ux = dx / dist;
      const uy = dy / dist;
      const rA = this.cfg.workerRadius * a.scale + 4;
      const rB = this.cfg.masterRadius * b.scale + 4;
      const x1 = a.x + ux * rA;
      const y1 = a.y + uy * rA;
      const x2 = b.x - ux * rB;
      const y2 = b.y - uy * rB;

      if (isHoverEdge) {
        ctx.shadowColor = this.cfg.nvidiaGreen;
        ctx.shadowBlur = 10;
      }

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.shadowBlur = 0;

      ctx.restore();
    });
  }

  _drawPulses() {
    const ctx = this.ctx;
    const now = performance.now();
    this.pulses.forEach(p => {
      const a = p.edge.source;   // worker
      const b = p.edge.target;   // master
      const t = Math.min(1, (now - p.startTime) / p.duration);
      // t=0 at worker, t=1 at master (inward)
      const x = a.x + (b.x - a.x) * t;
      const y = a.y + (b.y - a.y) * t;

      // Brighter near master (final 30%)
      const nearMaster = t > 0.7 ? (t - 0.7) / 0.3 : 0;
      const coreAlpha = 0.65 + nearMaster * 0.35;
      const glowBlur = 8 + nearMaster * 12;
      const size = p.size + nearMaster * 1.5;

      ctx.save();
      ctx.globalAlpha = coreAlpha * 0.5;
      ctx.shadowColor = this.cfg.nvidiaGreenBright;
      ctx.shadowBlur = glowBlur;
      ctx.fillStyle = this.cfg.nvidiaGreen;
      ctx.beginPath();
      ctx.arc(x, y, size + 1.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.globalAlpha = coreAlpha;
      ctx.fillStyle = this.cfg.nvidiaGreenBright;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  }

  _drawMasterOrbitEffects(n) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = n.alpha * 0.9;

    const baseR = this.cfg.masterRadius;

    // Pulsing concentric rings (subtle)
    const phase = (this.time * 1.5) % (Math.PI * 2);
    for (let i = 0; i < 3; i++) {
      const rPhase = phase + i * 1.2;
      const ringR = baseR + 18 + i * 22 + Math.sin(rPhase) * 6;
      const ringAlpha = (0.09 - i * 0.025) + Math.sin(rPhase) * 0.025;
      ctx.strokeStyle = this._rgba(this.cfg.nvidiaGreen, Math.max(0, ringAlpha));
      ctx.lineWidth = 1.4 - i * 0.3;
      ctx.beginPath();
      ctx.arc(n.x, n.y, ringR, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Orbiting scan dots
    const scanR = baseR + 30;
    for (let i = 0; i < 3; i++) {
      const dotAngle = this.time * 0.8 + (i * Math.PI * 2 / 3);
      const dx = n.x + Math.cos(dotAngle) * scanR;
      const dy = n.y + Math.sin(dotAngle) * scanR;
      const dotAlpha = 0.35 + Math.sin(this.time * 3 + i) * 0.15;
      ctx.fillStyle = this._rgba(this.cfg.nvidiaGreen, dotAlpha);
      ctx.shadowColor = this.cfg.nvidiaGreen;
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.arc(dx, dy, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    ctx.restore();
  }

  _drawNode(n) {
    const ctx = this.ctx;
    const d = n.data;
    const cfg = this.cfg;
    const isHovered = this.hoveredNodeId === n.id;
    const isSelected = this.selectedNodeId === n.id;
    const isOnline = d.status === 'online';
    const isStale = d.status === 'stale';
    const isMaster = n.isMaster;
    const baseR = isMaster ? cfg.masterRadius : cfg.workerRadius;
    const r = baseR * n.scale;

    ctx.save();
    ctx.globalAlpha = n.alpha;

    // --- Outer glow halo ---
    const haloColor = isOnline ? cfg.nvidiaGreen : (isStale ? cfg.tempAmber : cfg.tempRed);
    const haloAlpha = (isMaster ? 0.35 : 0.22) + (isHovered ? 0.15 : 0) + (isSelected ? 0.15 : 0);
    const haloGrad = ctx.createRadialGradient(n.x, n.y, r * 0.9, n.x, n.y, r + (isMaster ? 16 : 10));
    haloGrad.addColorStop(0, this._rgba(haloColor, haloAlpha));
    haloGrad.addColorStop(1, this._rgba(haloColor, 0));
    ctx.fillStyle = haloGrad;
    ctx.beginPath();
    ctx.arc(n.x, n.y, r + (isMaster ? 16 : 10), 0, Math.PI * 2);
    ctx.fill();

    // --- Selected outer ring ---
    if (isSelected) {
      ctx.strokeStyle = this._rgba(cfg.nvidiaGreen, 0.5 + Math.sin(this.time * 3) * 0.2);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 6, 0, Math.PI * 2);
      ctx.stroke();
    }

    // --- Main planet circle with radial gradient ---
    const coreGrad = ctx.createRadialGradient(
      n.x - r * 0.3, n.y - r * 0.3, r * 0.1,
      n.x, n.y, r
    );
    coreGrad.addColorStop(0, cfg.nodeBgInner);
    coreGrad.addColorStop(1, cfg.nodeBg);
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    ctx.fill();

    // --- Border ---
    const borderColor = isOnline
      ? (isHovered ? cfg.nvidiaGreenBright : cfg.nvidiaGreen)
      : isStale ? cfg.nodeBorderStale : cfg.nodeBorderOffline;
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = isMaster ? 3 : 2;
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    ctx.stroke();

    // --- Memory ring (arc on outside of border) ---
    const memPct = Math.max(0, Math.min(100, d.gpu_memory_used_pct || 0));
    if (memPct > 0) {
      const memR = r + 4;
      const startA = -Math.PI / 2;
      const endA = startA + (memPct / 100) * Math.PI * 2;
      let memColor = cfg.tempGreen;
      if (memPct >= 85) memColor = cfg.tempRed;
      else if (memPct >= 65) memColor = cfg.tempAmber;
      ctx.strokeStyle = memColor;
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.arc(n.x, n.y, memR, startA, endA);
      ctx.stroke();
      ctx.lineCap = 'butt';
    }

    // --- Inside: hostname / gpu short / status dot ---
    const hostname = this._shortHost(d.node_name || d.node_id || 'Node');
    const gpuShort = this._shortGpu(d.gpu_name || '');
    const innerTopY = n.y - (isMaster ? 12 : 9);

    ctx.fillStyle = cfg.textPrimary;
    ctx.font = 'bold ' + (isMaster ? 12 : 11) + 'px Inter, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(this._truncate(hostname, isMaster ? 10 : 8), n.x, innerTopY);

    if (gpuShort) {
      ctx.fillStyle = cfg.textSecondary;
      ctx.font = (isMaster ? 10 : 9) + 'px JetBrains Mono, Fira Code, monospace';
      ctx.fillText(gpuShort, n.x, innerTopY + (isMaster ? 14 : 12));
    }

    // Status pulse dot inside bottom
    const statusColor = isOnline ? cfg.nvidiaGreen : isStale ? cfg.tempAmber : cfg.tempRed;
    const dotY = n.y + (isMaster ? 20 : 15);
    if (isOnline) {
      const pulseAlpha = 0.25 + Math.sin(this.time * 4) * 0.15;
      const pulseR = 4 + Math.sin(this.time * 4) * 1.5;
      ctx.fillStyle = this._rgba(statusColor, pulseAlpha);
      ctx.beginPath();
      ctx.arc(n.x, dotY, pulseR, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = statusColor;
    ctx.beginPath();
    ctx.arc(n.x, dotY, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // --- Master crown ---
    if (isMaster) {
      ctx.save();
      ctx.shadowColor = cfg.nvidiaGreen;
      ctx.shadowBlur = 10;
      ctx.fillStyle = cfg.nvidiaGreenBright;
      ctx.font = 'bold 16px Inter, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('\u2726', n.x, n.y - r - 14);
      ctx.restore();
    }

    // --- Below-circle labels ---
    const belowY = n.y + r + (isMaster ? 14 : 12);
    const memTotal = d.gpu_memory_gb || 0;
    const memUsed = memTotal > 0 ? (memTotal * memPct / 100) : 0;
    let memText;
    if (memTotal > 0) {
      memText = memUsed.toFixed(1) + '/' + memTotal.toFixed(0) + ' GB (' + Math.round(memPct) + '%)';
    } else {
      memText = '—';
    }
    ctx.fillStyle = cfg.textMuted;
    ctx.font = '10px JetBrains Mono, Fira Code, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(memText, n.x, belowY);

    if (d.model && d.model !== 'none') {
      ctx.fillStyle = cfg.nvidiaGreen;
      ctx.font = '9px JetBrains Mono, Fira Code, monospace';
      ctx.fillText(this._truncate(d.model, 24), n.x, belowY + 14);
    }

    ctx.restore();
  }

  _drawTooltip(n) {
    const ctx = this.ctx;
    const d = n.data;
    const lines = [];
    lines.push((d.node_name || d.node_id || 'Node'));
    if (d.gpu_name) lines.push(d.gpu_name);
    const parts = [];
    if (d.gpu_utilization != null) parts.push('Util ' + Math.round(d.gpu_utilization) + '%');
    if (d.gpu_temp != null) parts.push(Math.round(d.gpu_temp) + '°C');
    if (d.gpu_wattage != null) parts.push(Math.round(d.gpu_wattage) + 'W');
    if (parts.length) lines.push(parts.join('  •  '));
    if (d.effective_role) lines.push('Role: ' + d.effective_role);

    const padding = 8;
    const lineH = 14;
    ctx.font = '11px Inter, -apple-system, sans-serif';
    let maxW = 0;
    lines.forEach(l => { maxW = Math.max(maxW, ctx.measureText(l).width); });
    const w = maxW + padding * 2;
    const h = lines.length * lineH + padding * 2;

    let tx = this.mouse.x + 14;
    let ty = this.mouse.y + 14;
    if (tx + w > this.width - 4) tx = this.mouse.x - w - 14;
    if (ty + h > this.height - 4) ty = this.mouse.y - h - 14;

    ctx.save();
    ctx.fillStyle = 'rgba(10,10,10,0.92)';
    ctx.strokeStyle = this._rgba(this.cfg.nvidiaGreen, 0.5);
    ctx.lineWidth = 1;
    this._roundRect(tx, ty, w, h, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = this.cfg.textPrimary;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    lines.forEach((l, i) => {
      ctx.fillStyle = i === 0 ? this.cfg.nvidiaGreenBright : this.cfg.textPrimary;
      ctx.font = (i === 0 ? 'bold ' : '') + '11px Inter, -apple-system, sans-serif';
      ctx.fillText(l, tx + padding, ty + padding + i * lineH);
    });
    ctx.restore();
  }

  // --- Utilities ---

  _roundRect(x, y, w, h, r) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  _rgba(hex, alpha) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }

  _truncate(str, max) {
    if (!str) return '';
    return str.length > max ? str.slice(0, max - 1) + '\u2026' : str;
  }

  _shortHost(name) {
    if (!name) return 'node';
    // Strip domain and take first chunk
    let s = String(name).split('.')[0];
    if (s.length <= 10) return s;
    return s.slice(0, 8);
  }

  _shortGpu(name) {
    if (!name) return '';
    // "NVIDIA GB10" -> "GB10", "NVIDIA GeForce RTX 4090" -> "RTX 4090"
    let s = String(name).replace(/^NVIDIA\s+/i, '').replace(/^GeForce\s+/i, '');
    return this._truncate(s, 12);
  }

  // --- Animation Loop ---

  _startLoop() {
    const frame = (ts) => {
      if (this.lastTs === 0) this.lastTs = ts;
      this.dt = Math.min(64, ts - this.lastTs);
      this.lastTs = ts;
      this.time = ts * 0.002;
      this._simulate();
      this._updatePulses();
      this._draw();
      this.animFrameId = requestAnimationFrame(frame);
    };
    this.animFrameId = requestAnimationFrame(frame);
  }
}

// Backward-compatible wrapper: app.js uses `new Topology(canvas)` and `.update()`
const Topology = function(canvas) {
  const renderer = new TopologyRenderer(canvas);
  this.update = function(nodesArray) { renderer.update(nodesArray); };
  this.destroy = function() { renderer.destroy(); };
  this.resize = function() { renderer.resize(); };
  this.getSelectedNodeId = function() { return renderer.getSelectedNodeId(); };
  this.onNodeSelect = null;
  Object.defineProperty(this, 'onNodeSelect', {
    set: function(fn) { renderer.onNodeSelect = fn; },
    get: function() { return renderer.onNodeSelect; },
  });
};
