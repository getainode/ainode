/* AINode Topology — NVIDIA-Styled Interactive Node Visualization
 * Premium canvas-based topology renderer with force-directed layout,
 * animated data pulses, and drag physics.
 */

class TopologyRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.width = 0;
    this.height = 0;
    this.time = 0;
    this.animFrameId = null;

    // Graph state
    this.nodes = [];
    this.edges = [];

    // Interaction state
    this.mouse = { x: 0, y: 0, down: false };
    this.dragNode = null;
    this.dragOffset = { x: 0, y: 0 };
    this.dragStartPos = { x: 0, y: 0 };
    this.hoveredNodeId = null;
    this.selectedNodeId = null;

    // Callbacks
    this.onNodeSelect = null;

    // Data pulse particles
    this.pulses = [];

    // Config
    this.cfg = {
      nodeWidth: 180,
      nodeHeight: 120,
      cornerRadius: 12,
      padding: 60,
      minHeight: 360,
      // Physics
      repulsion: 9000,
      attraction: 0.004,
      centerGravity: 0.025,
      damping: 0.86,
      breathAmplitude: 0.3,
      // Colors
      nvidiaGreen: '#76B900',
      nvidiaGreenBright: '#8ACE00',
      nodeBg: '#111111',
      nodeBorder: '#76B900',
      nodeBorderOffline: '#333333',
      textPrimary: '#E0E0E0',
      textSecondary: '#AAAAAA',
      textMuted: '#888888',
      trackBg: '#1f1f1f',
      dotGrid: '#1a1a1a',
      tempGreen: '#76B900',
      tempAmber: '#F5A623',
      tempRed: '#FF4444',
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

    const updatedNodes = [];
    nodesArray.forEach((nd, i) => {
      const id = nd.node_id || nd.id || 'local';
      let gn = existingMap[id];
      if (gn) {
        gn.data = nd;
        gn.targetAlpha = 1;
      } else {
        gn = {
          id,
          data: nd,
          x: this.width / 2 + (Math.random() - 0.5) * 120,
          y: this.height / 2 + (Math.random() - 0.5) * 120,
          vx: 0,
          vy: 0,
          pinned: false,
          alpha: 0,
          targetAlpha: 1,
          scale: 1,
          targetScale: 1,
        };
      }
      updatedNodes.push(gn);
    });

    this.nodes = updatedNodes;
    this._buildEdges();

    if (this.nodes.length === 1) {
      const n = this.nodes[0];
      if (!n.pinned) {
        n.x = this.width / 2;
        n.y = this.height / 2;
      }
    } else if (this.nodes.length <= 4) {
      this._layoutCircle(false);
    }
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
    for (let i = 0; i < this.nodes.length; i++) {
      for (let j = i + 1; j < this.nodes.length; j++) {
        const a = this.nodes[i];
        const b = this.nodes[j];
        const aOnline = a.data.status === 'online';
        const bOnline = b.data.status === 'online';
        this.edges.push({
          source: a,
          target: b,
          active: aOnline && bOnline,
        });
      }
    }
  }

  _layoutCircle(force) {
    const cx = this.width / 2;
    const cy = this.height / 2;
    const radius = Math.min(this.width, this.height) * 0.25;
    this.nodes.forEach((n, i) => {
      if (force || (n.alpha < 0.1 && !n.pinned)) {
        const angle = (i / this.nodes.length) * Math.PI * 2 - Math.PI / 2;
        n.x = cx + Math.cos(angle) * radius;
        n.y = cy + Math.sin(angle) * radius;
      }
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
        this.dragNode.vx = 0;
        this.dragNode.vy = 0;
        this.dragNode.pinned = true;
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
          this._selectNode(this.dragNode.id);
        }
        this.dragNode.pinned = false;
        this.dragNode = null;
        this.canvas.style.cursor = 'default';
      }
      this.mouse.down = false;
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
      this.nodes.forEach(n => {
        n.x = Math.min(Math.max(n.x, this.cfg.padding), this.width - this.cfg.padding);
        n.y = Math.min(Math.max(n.y, this.cfg.padding), this.height - this.cfg.padding);
      });
    };

    this.canvas.addEventListener('mousemove', h.mouseMove);
    this.canvas.addEventListener('mousedown', h.mouseDown);
    this.canvas.addEventListener('mouseup', h.mouseUp);
    this.canvas.addEventListener('mouseleave', h.mouseLeave);
    this.canvas.addEventListener('click', h.click);
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
    // Update target scales
    this.nodes.forEach(n => {
      n.targetScale = (n.id === this.hoveredNodeId) ? 1.02 : 1.0;
    });
  }

  _getNodeAt(mx, my) {
    const nw = this.cfg.nodeWidth;
    const nh = this.cfg.nodeHeight;
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      const hw = nw / 2 * n.scale;
      const hh = nh / 2 * n.scale;
      if (mx >= n.x - hw && mx <= n.x + hw && my >= n.y - hh && my <= n.y + hh) {
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

  // --- Physics ---

  _simulate() {
    const N = this.nodes.length;
    if (N === 0) return;

    if (N === 1) {
      const n = this.nodes[0];
      if (!n.pinned) {
        // Gentle breathing motion around center
        const bx = Math.sin(this.time * 0.7) * this.cfg.breathAmplitude;
        const by = Math.cos(this.time * 0.5) * this.cfg.breathAmplitude;
        n.vx += (this.width / 2 + bx - n.x) * 0.02;
        n.vy += (this.height / 2 + by - n.y) * 0.02;
        n.vx *= this.cfg.damping;
        n.vy *= this.cfg.damping;
        n.x += n.vx;
        n.y += n.vy;
      }
      n.alpha += (n.targetAlpha - n.alpha) * 0.05;
      n.scale += (n.targetScale - n.scale) * 0.1;
      return;
    }

    // Repulsion
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const a = this.nodes[i];
        const b = this.nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 1) dist = 1;
        const force = this.cfg.repulsion / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        if (!a.pinned) { a.vx -= fx; a.vy -= fy; }
        if (!b.pinned) { b.vx += fx; b.vy += fy; }
      }
    }

    // Attraction along edges
    this.edges.forEach(edge => {
      const a = edge.source;
      const b = edge.target;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const idealDist = 300;
      const force = (dist - idealDist) * this.cfg.attraction;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      if (!a.pinned) { a.vx += fx; a.vy += fy; }
      if (!b.pinned) { b.vx -= fx; b.vy -= fy; }
    });

    // Center gravity + breathing
    const cx = this.width / 2;
    const cy = this.height / 2;
    this.nodes.forEach(n => {
      if (n.pinned) {
        n.alpha += (n.targetAlpha - n.alpha) * 0.05;
        n.scale += (n.targetScale - n.scale) * 0.1;
        return;
      }
      // Gentle breathing offset per node
      const bx = Math.sin(this.time * 0.5 + n.x * 0.01) * this.cfg.breathAmplitude;
      const by = Math.cos(this.time * 0.4 + n.y * 0.01) * this.cfg.breathAmplitude;
      n.vx += (cx + bx - n.x) * this.cfg.centerGravity;
      n.vy += (cy + by - n.y) * this.cfg.centerGravity;
      n.vx *= this.cfg.damping;
      n.vy *= this.cfg.damping;
      n.x += n.vx;
      n.y += n.vy;
      // Bounds
      const hw = this.cfg.nodeWidth / 2 + 10;
      const hh = this.cfg.nodeHeight / 2 + 10;
      n.x = Math.max(hw, Math.min(this.width - hw, n.x));
      n.y = Math.max(hh, Math.min(this.height - hh, n.y));
      // Animations
      n.alpha += (n.targetAlpha - n.alpha) * 0.05;
      n.scale += (n.targetScale - n.scale) * 0.1;
    });
  }

  // --- Pulse Particles ---

  _updatePulses() {
    // Spawn pulses on active edges
    if (this.edges.length > 0 && Math.random() < 0.03) {
      const edge = this.edges[Math.floor(Math.random() * this.edges.length)];
      if (edge.active) {
        this.pulses.push({
          edge,
          t: 0,
          speed: 0.005 + Math.random() * 0.005,
          size: 2 + Math.random() * 1.5,
        });
      }
    }
    // Update existing pulses
    this.pulses = this.pulses.filter(p => {
      p.t += p.speed;
      return p.t < 1;
    });
  }

  // --- Drawing ---

  _draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    this._drawDotGrid();
    this._drawEdges();
    this._drawPulses();

    // Sort nodes: selected on top
    const sorted = [...this.nodes].sort((a, b) => {
      if (a.id === this.selectedNodeId) return 1;
      if (b.id === this.selectedNodeId) return -1;
      return 0;
    });
    sorted.forEach(n => this._drawNode(n));

    if (this.nodes.length === 1) {
      this._drawSingleNodeEffects(this.nodes[0]);
    }
  }

  _drawDotGrid() {
    const ctx = this.ctx;
    const spacing = 20;
    ctx.fillStyle = this.cfg.dotGrid;
    for (let x = spacing; x < this.width; x += spacing) {
      for (let y = spacing; y < this.height; y += spacing) {
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }

  _drawEdges() {
    const ctx = this.ctx;
    this.edges.forEach(edge => {
      const a = edge.source;
      const b = edge.target;
      const alpha = Math.min(a.alpha, b.alpha);
      if (alpha < 0.01) return;

      ctx.save();
      ctx.globalAlpha = alpha;

      const opacity = edge.active ? 0.4 : 0.2;
      ctx.strokeStyle = this._rgba(this.cfg.nvidiaGreen, opacity);
      ctx.lineWidth = 1.5;

      if (!edge.active) {
        ctx.setLineDash([6, 6]);
      }

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Direction chevron at midpoint
      if (edge.active) {
        this._drawChevron(a.x, a.y, b.x, b.y, alpha * 0.6);
      }

      ctx.restore();
    });
  }

  _drawChevron(x1, y1, x2, y2, alpha) {
    const ctx = this.ctx;
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const size = 6;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = this.cfg.nvidiaGreen;
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(
      mx - Math.cos(angle - 0.5) * size,
      my - Math.sin(angle - 0.5) * size
    );
    ctx.lineTo(mx, my);
    ctx.lineTo(
      mx - Math.cos(angle + 0.5) * size,
      my - Math.sin(angle + 0.5) * size
    );
    ctx.stroke();

    // Second chevron offset forward
    const ox = Math.cos(angle) * 10;
    const oy = Math.sin(angle) * 10;
    ctx.beginPath();
    ctx.moveTo(
      mx + ox - Math.cos(angle - 0.5) * size,
      my + oy - Math.sin(angle - 0.5) * size
    );
    ctx.lineTo(mx + ox, my + oy);
    ctx.lineTo(
      mx + ox - Math.cos(angle + 0.5) * size,
      my + oy - Math.sin(angle + 0.5) * size
    );
    ctx.stroke();

    ctx.restore();
  }

  _drawPulses() {
    const ctx = this.ctx;
    this.pulses.forEach(p => {
      const a = p.edge.source;
      const b = p.edge.target;
      const x = a.x + (b.x - a.x) * p.t;
      const y = a.y + (b.y - a.y) * p.t;
      // Fade in/out at edges
      const fadeAlpha = Math.min(p.t * 5, 1) * Math.min((1 - p.t) * 5, 1);

      // Glow
      ctx.save();
      ctx.globalAlpha = fadeAlpha * 0.4;
      ctx.shadowColor = this.cfg.nvidiaGreen;
      ctx.shadowBlur = 8;
      ctx.fillStyle = this.cfg.nvidiaGreen;
      ctx.beginPath();
      ctx.arc(x, y, p.size + 1, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Core dot
      ctx.globalAlpha = fadeAlpha * 0.9;
      ctx.fillStyle = this.cfg.nvidiaGreenBright;
      ctx.beginPath();
      ctx.arc(x, y, p.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  }

  _drawNode(n) {
    const ctx = this.ctx;
    const d = n.data;
    const cfg = this.cfg;
    const isHovered = this.hoveredNodeId === n.id;
    const isSelected = this.selectedNodeId === n.id;
    const isLeader = d.is_leader;
    const isOnline = d.status === 'online';
    const nw = cfg.nodeWidth;
    const nh = cfg.nodeHeight;
    const r = cfg.cornerRadius;

    ctx.save();
    ctx.globalAlpha = n.alpha;

    // Apply scale transform around node center
    ctx.translate(n.x, n.y);
    ctx.scale(n.scale, n.scale);

    const x = -nw / 2;
    const y = -nh / 2;

    // Hover/selected glow
    if (isHovered || isSelected) {
      const glowColor = this.cfg.nvidiaGreen;
      const glowAlpha = isSelected ? 0.25 : 0.15;
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = isSelected ? 20 : 12;
      ctx.fillStyle = this._rgba(glowColor, glowAlpha);
      this._roundRect(x - 2, y - 2, nw + 4, nh + 4, r + 1);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Selected outer glow ring
    if (isSelected) {
      ctx.strokeStyle = this._rgba(this.cfg.nvidiaGreen, 0.3 + Math.sin(this.time * 3) * 0.1);
      ctx.lineWidth = 1;
      this._roundRect(x - 5, y - 5, nw + 10, nh + 10, r + 3);
      ctx.stroke();
    }

    // Node background
    ctx.fillStyle = cfg.nodeBg;
    this._roundRect(x, y, nw, nh, r);
    ctx.fill();

    // Border
    const borderColor = isOnline
      ? (isHovered ? cfg.nvidiaGreenBright : cfg.nvidiaGreen)
      : cfg.nodeBorderOffline;
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = isSelected ? 2 : 1;
    this._roundRect(x, y, nw, nh, r);
    ctx.stroke();

    // --- Content ---
    const leftPad = x + 12;
    const rightEdge = x + nw - 12;

    // Leader badge (top-right)
    if (isLeader) {
      ctx.fillStyle = this.cfg.nvidiaGreen;
      ctx.font = '12px Inter, -apple-system, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText('\u2605', rightEdge, y + 8);
    }

    // Hostname (bold, left)
    ctx.fillStyle = cfg.textPrimary;
    ctx.font = 'bold 14px Inter, -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    const hostname = this._truncate(d.node_name || d.node_id || 'Node', 16);
    ctx.fillText(hostname, leftPad, y + 10);

    // GPU utilization % (bold, right side)
    const util = d.gpu_utilization != null ? d.gpu_utilization : (d.gpu_utilization_pct != null ? d.gpu_utilization_pct : null);
    if (util != null) {
      ctx.fillStyle = cfg.nvidiaGreen;
      ctx.font = 'bold 18px Inter, -apple-system, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText(Math.round(util) + '%', rightEdge, y + 8);
    }

    // GPU name (secondary text)
    ctx.fillStyle = cfg.textSecondary;
    ctx.font = '10px Inter, -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    const gpuName = this._truncate(d.gpu_name || 'Unknown GPU', 24);
    ctx.fillText(gpuName, leftPad, y + 30);

    // Temperature with color coding
    const temp = d.gpu_temp;
    if (temp != null) {
      let tempColor = cfg.tempGreen;
      if (temp >= 80) tempColor = cfg.tempRed;
      else if (temp >= 60) tempColor = cfg.tempAmber;

      ctx.fillStyle = tempColor;
      ctx.font = '11px Inter, -apple-system, sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(Math.round(temp) + '\u00B0C', leftPad, y + 46);
    }

    // Wattage (small text, next to temp)
    const wattage = d.gpu_wattage;
    if (wattage != null) {
      ctx.fillStyle = cfg.textMuted;
      ctx.font = '10px Inter, -apple-system, sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      const tempOffset = temp != null ? 45 : 0;
      ctx.fillText(Math.round(wattage) + 'W', leftPad + tempOffset, y + 47);
    }

    // Model name if loaded
    if (d.model && d.model !== 'none') {
      ctx.fillStyle = cfg.textMuted;
      ctx.font = '9px JetBrains Mono, Fira Code, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText(this._truncate(d.model, 18), rightEdge, y + 47);
    }

    // Memory bar
    const barX = leftPad;
    const barY = y + 66;
    const barW = nw - 24;
    const barH = 4;
    const memPct = d.gpu_memory_used_pct || 0;

    // Track
    ctx.fillStyle = cfg.trackBg;
    this._roundRect(barX, barY, barW, barH, 2);
    ctx.fill();

    // Fill
    if (memPct > 0) {
      ctx.fillStyle = cfg.nvidiaGreen;
      const fillW = Math.max(2, barW * (memPct / 100));
      this._roundRect(barX, barY, fillW, barH, 2);
      ctx.fill();
    }

    // Memory text below bar
    const memTotal = d.gpu_memory_gb || 0;
    const memUsed = memTotal > 0 ? (memTotal * memPct / 100) : 0;
    const memText = memUsed.toFixed(1) + 'GB/' + memTotal + 'GB (' + Math.round(memPct) + '%)';
    ctx.fillStyle = cfg.textMuted;
    ctx.font = '9px JetBrains Mono, Fira Code, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(memText, leftPad, barY + 8);

    // Shard role badge if present
    if (d.shard_role) {
      const badgeY = barY + 22;
      const badgeColor = d.shard_role === 'head' ? cfg.nvidiaGreen : '#8B5CF6';
      ctx.fillStyle = this._rgba(badgeColor, 0.15);
      this._roundRect(leftPad, badgeY, nw - 24, 16, 3);
      ctx.fill();
      ctx.fillStyle = badgeColor;
      ctx.font = 'bold 8px Inter, -apple-system, sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(
        d.shard_role.toUpperCase() + ' | Layers ' + (d.shard_layers || 'all'),
        leftPad + 4,
        badgeY + 3
      );
    }

    // Status indicator dot (bottom-right)
    const statusColor = isOnline ? cfg.nvidiaGreen : d.status === 'stale' ? cfg.tempAmber : cfg.tempRed;
    const dotX = rightEdge - 2;
    const dotY = y + nh - 12;

    if (isOnline) {
      // Pulse ring
      const pulseAlpha = 0.2 + Math.sin(this.time * 4) * 0.15;
      const pulseR = 5 + Math.sin(this.time * 4) * 2;
      ctx.fillStyle = this._rgba(statusColor, pulseAlpha);
      ctx.beginPath();
      ctx.arc(dotX, dotY, pulseR, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = statusColor;
    ctx.beginPath();
    ctx.arc(dotX, dotY, 3, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  _drawSingleNodeEffects(n) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = n.alpha;

    // Pulsing green rings
    const phase = (this.time * 1.5) % (Math.PI * 2);
    for (let i = 0; i < 3; i++) {
      const rPhase = phase + i * 1.2;
      const ringR = this.cfg.nodeWidth * 0.7 + i * 30 + Math.sin(rPhase) * 8;
      const ringAlpha = (0.08 - i * 0.02) + Math.sin(rPhase) * 0.03;
      ctx.strokeStyle = this._rgba(this.cfg.nvidiaGreen, Math.max(0, ringAlpha));
      ctx.lineWidth = 1.5 - i * 0.3;
      ctx.beginPath();
      ctx.arc(n.x, n.y, ringR, 0, Math.PI * 2);
      ctx.stroke();
    }

    // "Scanning for peers..." subtitle
    const textAlpha = 0.4 + Math.sin(this.time * 2) * 0.2;
    ctx.font = '12px Inter, -apple-system, sans-serif';
    ctx.fillStyle = this._rgba(this.cfg.textMuted, textAlpha);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Scanning for peers...', n.x, n.y + this.cfg.nodeHeight / 2 + 24);

    // Orbiting scan dots
    for (let i = 0; i < 3; i++) {
      const dotAngle = this.time * 0.8 + (i * Math.PI * 2 / 3);
      const dotR = this.cfg.nodeWidth * 0.8;
      const dx = n.x + Math.cos(dotAngle) * dotR;
      const dy = n.y + Math.sin(dotAngle) * dotR;
      const dotAlpha = 0.2 + Math.sin(this.time * 3 + i) * 0.1;
      ctx.fillStyle = this._rgba(this.cfg.nvidiaGreen, dotAlpha);
      ctx.beginPath();
      ctx.arc(dx, dy, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

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

  // --- Animation Loop ---

  _startLoop() {
    const frame = (ts) => {
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
  // Proxy onNodeSelect callback
  const self = this;
  Object.defineProperty(this, 'onNodeSelect', {
    set: function(fn) { renderer.onNodeSelect = fn; },
    get: function() { return renderer.onNodeSelect; },
  });
};
