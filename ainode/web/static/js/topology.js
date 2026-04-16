/* ============================================================
 * AINode Topology — Hub & Spoke
 *
 * Master node sits at canvas center with the original "scanning"
 * pulsing concentric rings + orbiting scan dots animation.
 * Worker nodes are placed STATICALLY on the circumference of an
 * outer ring around the master. They pulsate gently in place but
 * do NOT rotate or move.
 *
 * Each worker has an "electricity" connection line to the master
 * — a soft baseline glow with a bright pulse traveling worker → master.
 *
 * Nodes are circular ("planets"), not rectangles.
 * ============================================================ */

(function (global) {
  'use strict';

  const CFG = {
    nvidiaGreen: '#76B900',
    nvidiaGreenBright: '#8ACE00',
    nvidiaGreenDim: '#4a7a00',
    bgGrid: '#151515',
    textPrimary: '#E0E0E0',
    textSecondary: '#888888',
    textMuted: '#555555',
    amber: '#FFB800',
    red: '#FF3333',
    masterRadius: 56,
    workerRadius: 40,
    orbitRatio: 0.32,
    minHeight: 380,
    fontSans: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    fontMono: 'JetBrains Mono, Fira Code, SF Mono, monospace',
  };

  class TopologyRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.dpr = window.devicePixelRatio || 1;
      this.width = 0;
      this.height = 0;
      this.nodes = [];
      this.master = null;
      this.time = 0;
      this.lastFrame = 0;
      this.hoverIdx = -1;
      this.selectedId = null;
      this.mouse = { x: -1, y: -1 };
      this.onNodeSelect = null;

      // Loading / fade-in state
      this._prevNodeIds = new Set();   // tracks which nodes were present last frame
      this._nodeAlpha = {};            // per-node fade-in alpha (0 → 1)
      this._masterTransition = 0;      // 0 = fully loading, 1 = fully real
      this._engineReady = false;       // set true when engine comes online

      this._bind();
      this._resize();
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
      window.addEventListener('resize', () => this._resize());
    }

    update(nodesArray, engineReady) {
      const incoming = (nodesArray || []).map((data) => ({
        data,
        id: data.node_id,
        x: 0, y: 0,
        pulse: Math.random() * Math.PI * 2,
      }));

      const masterIdx = incoming.findIndex(
        (n) => n.data.effective_role === 'master' || n.data.is_leader
      );
      if (masterIdx >= 0) {
        const m = incoming.splice(masterIdx, 1)[0];
        m.role = 'master';
        incoming.unshift(m);
      } else if (incoming.length > 0) {
        incoming[0].role = 'master';
      }
      incoming.forEach((n) => { if (!n.role) n.role = 'worker'; });

      // Track which nodes are new (get fade-in alpha starting at 0)
      incoming.forEach((n) => {
        if (!this._prevNodeIds.has(n.id)) {
          this._nodeAlpha[n.id] = n.role === 'master' ? 1 : 0; // master starts visible
        }
      });
      this._prevNodeIds = new Set(incoming.map((n) => n.id));

      // Engine-ready transition: loading circle → real master node
      const wasReady = this._engineReady;
      this._engineReady = !!engineReady;
      if (!wasReady && this._engineReady) {
        this._masterTransition = 0; // start cross-fade
      }

      this.nodes = incoming;
      this.master = this.nodes[0] || null;
      this._layout();
    }

    _layout() {
      if (!this.master) return;
      const cx = this.width / 2;
      const cy = this.height / 2;
      this.master.x = cx;
      this.master.y = cy;

      const workers = this.nodes.filter((n) => n.role === 'worker');
      if (workers.length === 0) return;

      const orbitR = Math.min(this.width, this.height) * CFG.orbitRatio;
      workers.forEach((w, i) => {
        const angle = -Math.PI / 2 + (i / workers.length) * Math.PI * 2;
        w.x = cx + Math.cos(angle) * orbitR;
        w.y = cy + Math.sin(angle) * orbitR;
        w.angle = angle;
      });
    }

    _resize() {
      const parent = this.canvas.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      this.width = rect.width;
      this.height = Math.max(rect.height, CFG.minHeight);
      this.canvas.width = this.width * this.dpr;
      this.canvas.height = this.height * this.dpr;
      this.canvas.style.width = this.width + 'px';
      this.canvas.style.height = this.height + 'px';
      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      this._layout();
    }

    _bind() {
      this.canvas.addEventListener('mousemove', (e) => {
        const r = this.canvas.getBoundingClientRect();
        this.mouse.x = e.clientX - r.left;
        this.mouse.y = e.clientY - r.top;
        this.hoverIdx = this._nodeAt(this.mouse.x, this.mouse.y);
        this.canvas.style.cursor = this.hoverIdx >= 0 ? 'pointer' : 'default';
      });
      this.canvas.addEventListener('mouseleave', () => {
        this.hoverIdx = -1;
        this.mouse.x = -1; this.mouse.y = -1;
      });
      this.canvas.addEventListener('click', (e) => {
        const idx = this._nodeAt(e.offsetX, e.offsetY);
        if (idx >= 0) {
          this.selectedId = this.nodes[idx].id;
          if (this.onNodeSelect) this.onNodeSelect(this.nodes[idx].data);
        } else {
          this.selectedId = null;
        }
      });
    }

    _nodeAt(x, y) {
      for (let i = this.nodes.length - 1; i >= 0; i--) {
        const n = this.nodes[i];
        const r = n.role === 'master' ? CFG.masterRadius : CFG.workerRadius;
        const dx = x - n.x; const dy = y - n.y;
        if (dx * dx + dy * dy <= r * r) return i;
      }
      return -1;
    }

    _loop(ts) {
      const dt = this.lastFrame ? (ts - this.lastFrame) / 1000 : 0;
      this.lastFrame = ts;
      this.time += dt;
      this._render();
      requestAnimationFrame(this._loop);
    }

    _render() {
      const ctx = this.ctx;
      const dt = 0.016; // approx frame delta for transitions

      ctx.clearRect(0, 0, this.width, this.height);
      this._drawGrid();

      if (!this.master) {
        // No nodes at all — show the loading animation
        this._drawLoading();
        return;
      }

      // Advance master transition (loading → real): takes ~0.8s
      if (this._engineReady && this._masterTransition < 1) {
        this._masterTransition = Math.min(1, this._masterTransition + dt / 0.8);
      } else if (!this._engineReady && this.master) {
        // Engine not ready — stay in loading state
        this._masterTransition = Math.max(0, this._masterTransition - dt / 0.5);
      }

      // Advance per-worker fade-in
      this.nodes.forEach((n) => {
        if (n.role === 'worker') {
          if (this._nodeAlpha[n.id] === undefined) this._nodeAlpha[n.id] = 0;
          this._nodeAlpha[n.id] = Math.min(1, (this._nodeAlpha[n.id] || 0) + dt / 1.2);
        }
      });

      // Master rings
      this._drawMasterRings();

      // Connection lines (fade in with their workers)
      this.nodes.forEach((n) => {
        if (n.role === 'worker') {
          const a = this._nodeAlpha[n.id] || 0;
          if (a > 0) this._drawConnection(n, this.master, a);
        }
      });

      // Nodes — master always visible once discovered; workers fade in
      this.nodes.forEach((n, i) => {
        const alpha = n.role === 'master' ? 1 : (this._nodeAlpha[n.id] || 0);
        if (alpha > 0.01) this._drawNode(n, i === this.hoverIdx, alpha);
      });

      // Loading overlay on master while engine isn't ready (fades out)
      if (this._masterTransition < 1 && this.master) {
        this._drawMasterLoadingOverlay(1 - this._masterTransition);
      }

      if (this.hoverIdx >= 0) this._drawTooltip(this.nodes[this.hoverIdx]);
      if (this.nodes.length === 1) this._drawWaitingLabel();
    }

    _drawGrid() {
      const ctx = this.ctx;
      ctx.save();
      ctx.fillStyle = CFG.bgGrid;
      const step = 22;
      for (let x = step / 2; x < this.width; x += step) {
        for (let y = step / 2; y < this.height; y += step) {
          ctx.beginPath();
          ctx.arc(x, y, 0.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
    }

    _drawEmpty() {
      this._drawLoading();
    }

    _drawLoading() {
      // Shown when no nodes have been discovered yet.
      // Renders a pulsating master-sized circle in the center that says
      // "Loading..." — exactly like the real master node but ghosted/breathing.
      const ctx = this.ctx;
      const cx = this.width / 2;
      const cy = this.height / 2;
      const r = CFG.masterRadius;
      const t = this.time;

      ctx.save();

      // Expanding ghost rings (same as master rings but dimmer)
      for (let i = 0; i < 4; i++) {
        const phase = (t * 0.6 + i * 0.55) % 2.0;
        const frac = phase / 2.0;
        const ringR = r + 10 + frac * 90;
        const alpha = (1 - frac) * 0.22 - i * 0.03;
        if (alpha <= 0) continue;
        ctx.strokeStyle = this._rgba(CFG.nvidiaGreenDim, alpha);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Breathing scale
      const breathe = 1 + Math.sin(t * 1.8) * 0.05;
      ctx.translate(cx, cy);
      ctx.scale(breathe, breathe);

      // Halo
      const haloR = r + 14;
      const halo = ctx.createRadialGradient(0, 0, r * 0.7, 0, 0, haloR);
      halo.addColorStop(0, this._rgba(CFG.nvidiaGreenDim, 0.18));
      halo.addColorStop(1, this._rgba(CFG.nvidiaGreenDim, 0));
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(0, 0, haloR, 0, Math.PI * 2);
      ctx.fill();

      // Circle fill
      const fill = ctx.createRadialGradient(0, -r * 0.3, 2, 0, 0, r);
      fill.addColorStop(0, '#111');
      fill.addColorStop(1, '#080808');
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fill();

      // Dashed border — indicates "pending"
      const borderAlpha = 0.35 + Math.sin(t * 2.5) * 0.2;
      ctx.strokeStyle = this._rgba(CFG.nvidiaGreenDim, borderAlpha);
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // "Loading..." text with animated dots
      const dots = '.'.repeat(1 + Math.floor(t * 2) % 3);
      ctx.fillStyle = this._rgba(CFG.textSecondary, 0.7 + Math.sin(t * 2) * 0.2);
      ctx.font = 'bold 11px ' + CFG.fontSans;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Loading' + dots, 0, -5);

      // Spinning arc inside to show activity
      const arcStart = t * 2.4;
      const arcLen = Math.PI * 0.7 + Math.sin(t * 1.2) * 0.3;
      ctx.strokeStyle = this._rgba(CFG.nvidiaGreen, 0.4);
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.55, arcStart, arcStart + arcLen);
      ctx.stroke();

      ctx.restore();
    }

    _drawLoadingGhost(alpha) {
      // Legacy — not used, kept for safety
    }

    _drawMasterLoadingOverlay(alpha) {
      // Shown on top of the real master node while engine is starting up.
      // Keeps the node name/GPU visible but adds a spinning arc + dim veil.
      if (alpha <= 0 || !this.master) return;
      const ctx = this.ctx;
      const m = this.master;
      const r = CFG.masterRadius;
      const t = this.time;

      ctx.save();
      ctx.globalAlpha = alpha * 0.75;
      ctx.translate(m.x, m.y);

      // Dim veil over the node interior
      const veil = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
      veil.addColorStop(0, 'rgba(0,0,0,0.55)');
      veil.addColorStop(1, 'rgba(0,0,0,0.2)');
      ctx.fillStyle = veil;
      ctx.beginPath();
      ctx.arc(0, 0, r - 3, 0, Math.PI * 2);
      ctx.fill();

      // Spinning arc inside the circle
      const arcStart = t * 2.8;
      const arcLen = Math.PI * 0.65 + Math.sin(t * 1.4) * 0.25;
      ctx.strokeStyle = this._rgba(CFG.nvidiaGreen, 0.7);
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.52, arcStart, arcStart + arcLen);
      ctx.stroke();

      // Small "starting up" label — doesn't replace the node name
      const dots = '.'.repeat(1 + Math.floor(t * 2) % 3);
      ctx.globalAlpha = alpha * 0.85;
      ctx.fillStyle = this._rgba(CFG.textSecondary, 0.9);
      ctx.font = '9px ' + CFG.fontMono;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('starting' + dots, 0, 18);

      ctx.restore();
    }

    _drawMasterRings() {
      const ctx = this.ctx;
      const m = this.master;
      const baseR = CFG.masterRadius;
      ctx.save();

      // Expanding concentric rings
      for (let i = 0; i < 4; i++) {
        const phase = (this.time * 0.7 + i * 0.5) % 2.0;
        const t = phase / 2.0;
        const r = baseR + 12 + t * 110;
        const alpha = (1 - t) * 0.35 - i * 0.04;
        if (alpha <= 0) continue;
        ctx.strokeStyle = this._rgba(CFG.nvidiaGreen, alpha);
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.arc(m.x, m.y, r, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Orbiting scan dots
      const dotR = baseR + 22;
      for (let i = 0; i < 3; i++) {
        const a = this.time * 0.8 + (i * Math.PI * 2 / 3);
        const dx = m.x + Math.cos(a) * dotR;
        const dy = m.y + Math.sin(a) * dotR;
        const dotAlpha = 0.4 + Math.sin(this.time * 3 + i) * 0.2;
        ctx.fillStyle = this._rgba(CFG.nvidiaGreen, dotAlpha);
        ctx.beginPath();
        ctx.arc(dx, dy, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();
    }

    _drawConnection(worker, master, alpha) {
      if (alpha !== undefined && alpha < 0.01) return;
      const ctx = this.ctx;
      const wOnline = (worker.data.status || 'online') === 'online';
      const dx = master.x - worker.x;
      const dy = master.y - worker.y;
      const dist = Math.hypot(dx, dy) || 1;
      const ux = dx / dist, uy = dy / dist;
      const sx = worker.x + ux * CFG.workerRadius;
      const sy = worker.y + uy * CFG.workerRadius;
      const ex = master.x - ux * CFG.masterRadius;
      const ey = master.y - uy * CFG.masterRadius;

      ctx.save();
      if (alpha !== undefined) ctx.globalAlpha = alpha;

      if (!wOnline) {
        ctx.strokeStyle = this._rgba(CFG.textMuted, 0.4);
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 6]);
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.restore();
        return;
      }

      // Soft baseline glow
      ctx.strokeStyle = this._rgba(CFG.nvidiaGreen, 0.35);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();

      // Electric pulse traveling worker → master
      const cycleSec = 1.6;
      const offset = (worker.angle || 0) * 0.15;
      const t = (((this.time + offset) % cycleSec) / cycleSec);
      const segLen = 0.28;
      const t0 = Math.max(0, t - segLen);
      const t1 = t;
      const px0 = sx + (ex - sx) * t0;
      const py0 = sy + (ey - sy) * t0;
      const px1 = sx + (ex - sx) * t1;
      const py1 = sy + (ey - sy) * t1;

      const grad = ctx.createLinearGradient(px0, py0, px1, py1);
      grad.addColorStop(0, this._rgba(CFG.nvidiaGreen, 0));
      grad.addColorStop(0.5, this._rgba(CFG.nvidiaGreenBright, 0.95));
      grad.addColorStop(1, this._rgba('#ffffff', 0.85));

      ctx.strokeStyle = grad;
      ctx.lineWidth = 2.6;
      ctx.shadowColor = CFG.nvidiaGreen;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(px0, py0);
      ctx.lineTo(px1, py1);
      ctx.stroke();

      ctx.restore();
    }

    _drawNode(n, isHovered, alpha) {
      const ctx = this.ctx;
      const isMaster = n.role === 'master';
      const r = isMaster ? CFG.masterRadius : CFG.workerRadius;
      const isOnline = (n.data.status || 'online') === 'online';
      const stroke = isOnline
        ? (isMaster ? CFG.nvidiaGreenBright : CFG.nvidiaGreen)
        : ((n.data.status === 'stale') ? CFG.amber : CFG.red);

      // Gentle in-place pulse
      const pulseAmp = isMaster ? 0.04 : 0.06;
      const scale = 1 + Math.sin(this.time * 2 + n.pulse) * pulseAmp + (isHovered ? 0.06 : 0);

      ctx.save();
      if (alpha !== undefined && alpha < 1) ctx.globalAlpha = alpha;
      ctx.translate(n.x, n.y);
      ctx.scale(scale, scale);

      // Outer halo
      const haloR = r + 14;
      const halo = ctx.createRadialGradient(0, 0, r * 0.7, 0, 0, haloR);
      halo.addColorStop(0, this._rgba(stroke, isMaster ? 0.32 : 0.22));
      halo.addColorStop(1, this._rgba(stroke, 0));
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(0, 0, haloR, 0, Math.PI * 2);
      ctx.fill();

      // Memory ring on border
      const memPct = Number(n.data.gpu_memory_used_pct || 0);
      if (memPct > 0) {
        const memColor = memPct > 90 ? CFG.red : memPct > 70 ? CFG.amber : CFG.nvidiaGreen;
        ctx.strokeStyle = this._rgba(memColor, 0.85);
        ctx.lineWidth = 3.5;
        ctx.beginPath();
        ctx.arc(0, 0, r + 4, -Math.PI / 2, -Math.PI / 2 + (memPct / 100) * Math.PI * 2);
        ctx.stroke();
      }

      // Main circle fill
      const fill = ctx.createRadialGradient(0, -r * 0.3, 2, 0, 0, r);
      fill.addColorStop(0, '#1a1a1a');
      fill.addColorStop(1, '#0a0a0a');
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fill();

      // Border
      ctx.strokeStyle = stroke;
      ctx.lineWidth = isMaster ? 3 : 2;
      ctx.shadowColor = stroke;
      ctx.shadowBlur = isMaster ? 14 : 8;
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Selected
      if (this.selectedId === n.id) {
        ctx.strokeStyle = this._rgba(CFG.nvidiaGreenBright, 0.85);
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.arc(0, 0, r + 9, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Inside text
      const name = this._shortName(n.data.node_name || n.data.node_id, isMaster ? 12 : 10);
      ctx.fillStyle = CFG.textPrimary;
      ctx.font = (isMaster ? 'bold 12px ' : 'bold 11px ') + CFG.fontSans;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(name, 0, -6);

      const gpuShort = this._shortGpuName(n.data.gpu_name || '');
      ctx.fillStyle = CFG.textSecondary;
      ctx.font = '9px ' + CFG.fontMono;
      ctx.fillText(gpuShort, 0, 8);

      // Inner status dot
      const dotColor = isOnline ? CFG.nvidiaGreen : CFG.red;
      const dotPulse = 0.5 + Math.sin(this.time * 3.5 + n.pulse) * 0.4;
      ctx.fillStyle = this._rgba(dotColor, dotPulse);
      ctx.beginPath();
      ctx.arc(0, r * 0.45, 2.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();

      // Crown on master (unscaled)
      if (isMaster) {
        ctx.save();
        ctx.fillStyle = CFG.nvidiaGreenBright;
        ctx.shadowColor = CFG.nvidiaGreen;
        ctx.shadowBlur = 12;
        ctx.font = 'bold 18px ' + CFG.fontSans;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('✦', n.x, n.y - r - 14);
        ctx.restore();
      }

      // Below labels
      const labelY = n.y + r + 22;
      const memTotal = Number(n.data.gpu_memory_gb || 0).toFixed(0);
      const memUsed = (memTotal * memPct / 100).toFixed(1);
      const memText = memUsed + '/' + memTotal + ' GB (' + Math.round(memPct) + '%)';

      ctx.save();
      ctx.fillStyle = CFG.textSecondary;
      ctx.font = '10px ' + CFG.fontMono;
      ctx.textAlign = 'center';
      ctx.fillText(memText, n.x, labelY);

      const model = n.data.model;
      if (model) {
        ctx.fillStyle = CFG.nvidiaGreen;
        ctx.font = '9px ' + CFG.fontMono;
        ctx.fillText(this._shortText(model, 28), n.x, labelY + 14);
      }

      ctx.fillStyle = CFG.textMuted;
      ctx.font = 'bold 8px ' + CFG.fontSans;
      ctx.fillText(isMaster ? 'MASTER' : 'WORKER', n.x, n.y - r - 28);
      ctx.restore();
    }

    _drawWaitingLabel() {
      const ctx = this.ctx;
      const m = this.master;
      const labelAlpha = 0.4 + Math.sin(this.time * 1.6) * 0.2;
      ctx.save();
      ctx.fillStyle = this._rgba(CFG.textMuted, labelAlpha);
      ctx.font = '12px ' + CFG.fontSans;
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for peers...', m.x, m.y + CFG.masterRadius + 78);
      ctx.restore();
    }

    _drawTooltip(n) {
      const ctx = this.ctx;
      const lines = [
        n.data.node_name || n.data.node_id,
        (n.data.gpu_name || '?'),
      ];
      const util = Number(n.data.gpu_utilization || 0);
      const temp = Number(n.data.gpu_temp || 0);
      const watts = Number(n.data.gpu_wattage || 0);
      if (util > 0) lines.push('Util ' + util.toFixed(0) + '%');
      if (temp > 0) lines.push('Temp ' + temp.toFixed(0) + '°C');
      if (watts > 0) lines.push(watts.toFixed(0) + ' W');
      lines.push((n.role || '').toUpperCase());

      const padX = 10, padY = 8, lineH = 14;
      ctx.save();
      ctx.font = '11px ' + CFG.fontMono;
      const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + padX * 2;
      const h = lines.length * lineH + padY * 2;
      let x = this.mouse.x + 14;
      let y = this.mouse.y + 14;
      if (x + w > this.width - 4) x = this.mouse.x - w - 14;
      if (y + h > this.height - 4) y = this.mouse.y - h - 14;

      ctx.fillStyle = '#0a0a0a';
      ctx.strokeStyle = CFG.nvidiaGreenDim;
      ctx.lineWidth = 1;
      this._roundRect(x, y, w, h, 6);
      ctx.fill();
      ctx.stroke();

      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      lines.forEach((line, i) => {
        ctx.fillStyle = i === 0 ? CFG.textPrimary :
                        i === lines.length - 1 ? CFG.nvidiaGreen : CFG.textSecondary;
        ctx.fillText(line, x + padX, y + padY + i * lineH);
      });
      ctx.restore();
    }

    _shortName(s, maxLen) {
      s = String(s || '');
      if (s.length <= maxLen) return s;
      return s.slice(0, maxLen - 1) + '…';
    }

    _shortGpuName(s) {
      const m = String(s).match(/(GB10|H100|A100|L40S|L40|H200|RTX\s*\d+|GH200|B200)/i);
      return m ? m[0].replace(/\s+/g, '') : s.replace('NVIDIA', '').trim();
    }

    _shortText(s, maxLen) {
      s = String(s || '');
      if (s.length <= maxLen) return s;
      return s.slice(0, maxLen - 1) + '…';
    }

    _roundRect(x, y, w, h, r) {
      const ctx = this.ctx;
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    _rgba(hex, alpha) {
      const h = hex.replace('#', '');
      const bigint = parseInt(h.length === 3
        ? h.split('').map((c) => c + c).join('')
        : h, 16);
      const r = (bigint >> 16) & 255;
      const g = (bigint >> 8) & 255;
      const b = bigint & 255;
      return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    }
  }

  function Topology(canvas) {
    const renderer = new TopologyRenderer(canvas);
    return {
      update(nodes) { renderer.update(nodes); },
      destroy() {},
      get onNodeSelect() { return renderer.onNodeSelect; },
      set onNodeSelect(fn) { renderer.onNodeSelect = fn; },
    };
  }

  global.Topology = Topology;
  global.TopologyRenderer = TopologyRenderer;
})(typeof window !== 'undefined' ? window : this);
