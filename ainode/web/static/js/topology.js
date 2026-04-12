/* AINode Topology — Interactive Force-Directed Node Graph */

const Topology = (() => {
  // --- Configuration ---
  const CONFIG = {
    nodeWidth: 220,
    nodeHeight: 88,
    nodeRadius: 14,
    minHeight: 360,
    padding: 60,
    // Physics
    repulsionStrength: 8000,
    attractionStrength: 0.005,
    centerGravity: 0.03,
    damping: 0.88,
    minVelocity: 0.01,
    // Appearance
    connectionWidth: 2,
    pulseSpeed: 0.002,
    hoverExpandHeight: 56,
    // Colors (read from CSS vars at init)
    colors: {},
  };

  // --- State ---
  let canvas = null;
  let ctx = null;
  let width = 0;
  let height = 0;
  let dpr = 1;
  let animFrameId = null;
  let time = 0;
  let selectedNodeId = null;
  let hoveredNodeId = null;
  let onSelectCallback = null;

  // Graph data
  let graphNodes = [];
  let graphEdges = [];

  // Mouse state
  let mouse = { x: 0, y: 0, down: false };
  let dragNode = null;
  let dragOffset = { x: 0, y: 0 };

  // --- Color Helpers ---
  function readCSSColors() {
    const style = getComputedStyle(document.documentElement);
    const get = (v) => style.getPropertyValue(v).trim();
    CONFIG.colors = {
      bgPrimary: get('--bg-primary') || '#0a0e17',
      bgCard: get('--bg-card') || '#1a2332',
      bgCardHover: get('--bg-card-hover') || '#1e2a3d',
      border: get('--border') || '#2d3748',
      borderActive: get('--border-active') || '#4a90d9',
      textPrimary: get('--text-primary') || '#e2e8f0',
      textSecondary: get('--text-secondary') || '#94a3b8',
      textMuted: get('--text-muted') || '#64748b',
      accent: get('--accent') || '#4a90d9',
      green: get('--green') || '#10b981',
      yellow: get('--yellow') || '#f59e0b',
      red: get('--red') || '#ef4444',
      purple: get('--purple') || '#8b5cf6',
    };
  }

  function hexToRgba(hex, alpha) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
    const r = parseInt(hex.substring(0,2), 16);
    const g = parseInt(hex.substring(2,4), 16);
    const b = parseInt(hex.substring(4,6), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // --- Initialization ---
  function init(canvasEl, onSelect) {
    canvas = canvasEl;
    ctx = canvas.getContext('2d');
    onSelectCallback = onSelect;
    dpr = window.devicePixelRatio || 1;
    readCSSColors();
    resize();
    bindEvents();
    startLoop();
  }

  function destroy() {
    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = null;
    unbindEvents();
  }

  function resize() {
    if (!canvas || !canvas.parentElement) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    width = rect.width;
    height = Math.max(rect.height, CONFIG.minHeight);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // --- Event Binding ---
  let boundHandlers = {};

  function bindEvents() {
    boundHandlers.mouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
      if (dragNode) {
        dragNode.x = mouse.x - dragOffset.x;
        dragNode.y = mouse.y - dragOffset.y;
        dragNode.vx = 0;
        dragNode.vy = 0;
        dragNode.pinned = true;
      }
      updateHover();
    };
    boundHandlers.mouseDown = (e) => {
      mouse.down = true;
      const node = getNodeAt(mouse.x, mouse.y);
      if (node) {
        dragNode = node;
        dragOffset.x = mouse.x - node.x;
        dragOffset.y = mouse.y - node.y;
        canvas.style.cursor = 'grabbing';
      }
    };
    boundHandlers.mouseUp = () => {
      if (dragNode) {
        // If barely moved, treat as click
        const dx = mouse.x - (dragNode.x + dragOffset.x);
        const dy = mouse.y - (dragNode.y + dragOffset.y);
        if (Math.abs(dx) < 3 && Math.abs(dy) < 3) {
          selectNode(dragNode.id);
        }
        dragNode.pinned = false;
        dragNode = null;
        canvas.style.cursor = 'default';
      }
      mouse.down = false;
    };
    boundHandlers.resize = () => {
      resize();
      // Re-center nodes if needed
      graphNodes.forEach(n => {
        n.x = Math.min(Math.max(n.x, CONFIG.padding), width - CONFIG.padding);
        n.y = Math.min(Math.max(n.y, CONFIG.padding), height - CONFIG.padding);
      });
    };
    boundHandlers.click = (e) => {
      const node = getNodeAt(mouse.x, mouse.y);
      if (!node && !dragNode) {
        selectNode(null);
      }
    };

    canvas.addEventListener('mousemove', boundHandlers.mouseMove);
    canvas.addEventListener('mousedown', boundHandlers.mouseDown);
    canvas.addEventListener('mouseup', boundHandlers.mouseUp);
    canvas.addEventListener('mouseleave', () => {
      hoveredNodeId = null;
      if (dragNode) { dragNode.pinned = false; dragNode = null; }
      mouse.down = false;
      canvas.style.cursor = 'default';
    });
    canvas.addEventListener('click', boundHandlers.click);
    window.addEventListener('resize', boundHandlers.resize);
  }

  function unbindEvents() {
    if (!canvas) return;
    canvas.removeEventListener('mousemove', boundHandlers.mouseMove);
    canvas.removeEventListener('mousedown', boundHandlers.mouseDown);
    canvas.removeEventListener('mouseup', boundHandlers.mouseUp);
    canvas.removeEventListener('click', boundHandlers.click);
    window.removeEventListener('resize', boundHandlers.resize);
  }

  function updateHover() {
    const node = getNodeAt(mouse.x, mouse.y);
    const newHovered = node ? node.id : null;
    if (newHovered !== hoveredNodeId) {
      hoveredNodeId = newHovered;
      canvas.style.cursor = newHovered ? 'grab' : 'default';
      if (dragNode) canvas.style.cursor = 'grabbing';
    }
  }

  function getNodeAt(mx, my) {
    // Reverse order so topmost (last drawn) is checked first
    for (let i = graphNodes.length - 1; i >= 0; i--) {
      const n = graphNodes[i];
      const nw = CONFIG.nodeWidth;
      const nh = getNodeHeight(n);
      if (mx >= n.x - nw/2 && mx <= n.x + nw/2 &&
          my >= n.y - nh/2 && my <= n.y + nh/2) {
        return n;
      }
    }
    return null;
  }

  function getNodeHeight(n) {
    let h = CONFIG.nodeHeight;
    if (n.data && n.data.shard_role) h += 22;
    if (hoveredNodeId === n.id) h += CONFIG.hoverExpandHeight + (n.data && n.data.sharded_model ? 16 : 0);
    return h;
  }

  function selectNode(id) {
    selectedNodeId = id;
    if (onSelectCallback) {
      const node = graphNodes.find(n => n.id === id);
      onSelectCallback(node ? node.data : null);
    }
  }

  // --- Data Update ---
  function update(nodesData) {
    if (!nodesData || !Array.isArray(nodesData)) return;

    const existingMap = {};
    graphNodes.forEach(n => { existingMap[n.id] = n; });

    const newIds = new Set(nodesData.map(n => n.node_id || n.id || 'local'));
    const updatedNodes = [];

    nodesData.forEach((nd, i) => {
      const id = nd.node_id || nd.id || 'local';
      let gn = existingMap[id];
      if (gn) {
        // Update data, keep position
        gn.data = nd;
      } else {
        // New node — place with initial position
        gn = {
          id,
          data: nd,
          x: width / 2 + (Math.random() - 0.5) * 100,
          y: height / 2 + (Math.random() - 0.5) * 100,
          vx: 0,
          vy: 0,
          pinned: false,
          targetAlpha: 1,
          alpha: 0, // fade in
        };
      }
      updatedNodes.push(gn);
    });

    graphNodes = updatedNodes;

    // Build edges: all-to-all mesh for discovered cluster
    graphEdges = [];
    for (let i = 0; i < graphNodes.length; i++) {
      for (let j = i + 1; j < graphNodes.length; j++) {
        const a = graphNodes[i];
        const b = graphNodes[j];
        const aOnline = a.data.status === 'online';
        const bOnline = b.data.status === 'online';
        const bothSharded = !!(a.data.shard_role && b.data.shard_role);
        graphEdges.push({
          source: a,
          target: b,
          solid: aOnline && bOnline,
          label: bothSharded ? 'SHARD' : 'LAN',
        });
      }
    }

    // If single node, position at center
    if (graphNodes.length === 1) {
      const n = graphNodes[0];
      if (!n.pinned) {
        n.x = width / 2;
        n.y = height / 2;
      }
    }
    // For 2-4 nodes, arrange in a circle initially if new
    else if (graphNodes.length <= 4) {
      layoutCircle(false);
    }
  }

  function layoutCircle(force) {
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.25;
    graphNodes.forEach((n, i) => {
      if (force || (n.alpha < 0.1 && !n.pinned)) {
        const angle = (i / graphNodes.length) * Math.PI * 2 - Math.PI / 2;
        n.x = cx + Math.cos(angle) * radius;
        n.y = cy + Math.sin(angle) * radius;
      }
    });
  }

  // --- Physics ---
  function simulate() {
    const N = graphNodes.length;
    if (N === 0) return;
    if (N === 1) {
      // Single node: gently drift to center
      const n = graphNodes[0];
      if (!n.pinned) {
        n.vx += (width / 2 - n.x) * 0.02;
        n.vy += (height / 2 - n.y) * 0.02;
        n.vx *= CONFIG.damping;
        n.vy *= CONFIG.damping;
        n.x += n.vx;
        n.y += n.vy;
      }
      n.alpha += (n.targetAlpha - n.alpha) * 0.05;
      return;
    }

    // Repulsion between all node pairs
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const a = graphNodes[i];
        const b = graphNodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 1) dist = 1;
        const force = CONFIG.repulsionStrength / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        if (!a.pinned) { a.vx -= fx; a.vy -= fy; }
        if (!b.pinned) { b.vx += fx; b.vy += fy; }
      }
    }

    // Attraction along edges
    graphEdges.forEach(edge => {
      const a = edge.source;
      const b = edge.target;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const idealDist = 280;
      const force = (dist - idealDist) * CONFIG.attractionStrength;
      const fx = (dx / (dist || 1)) * force;
      const fy = (dy / (dist || 1)) * force;
      if (!a.pinned) { a.vx += fx; a.vy += fy; }
      if (!b.pinned) { b.vx -= fx; b.vy -= fy; }
    });

    // Center gravity
    const cx = width / 2;
    const cy = height / 2;
    graphNodes.forEach(n => {
      if (n.pinned) return;
      n.vx += (cx - n.x) * CONFIG.centerGravity;
      n.vy += (cy - n.y) * CONFIG.centerGravity;
      // Damping
      n.vx *= CONFIG.damping;
      n.vy *= CONFIG.damping;
      // Apply
      n.x += n.vx;
      n.y += n.vy;
      // Bounds
      const hw = CONFIG.nodeWidth / 2 + 10;
      const hh = CONFIG.nodeHeight / 2 + 10;
      n.x = Math.max(hw, Math.min(width - hw, n.x));
      n.y = Math.max(hh, Math.min(height - hh, n.y));
      // Fade in
      n.alpha += (n.targetAlpha - n.alpha) * 0.05;
    });
  }

  // --- Rendering ---
  function draw() {
    const c = CONFIG.colors;
    ctx.clearRect(0, 0, width, height);

    // Background subtle grid
    drawGrid();

    // Draw edges
    graphEdges.forEach(edge => drawEdge(edge));

    // Draw nodes (selected on top)
    const sorted = [...graphNodes].sort((a, b) => {
      if (a.id === selectedNodeId) return 1;
      if (b.id === selectedNodeId) return -1;
      return 0;
    });
    sorted.forEach(n => drawNode(n));

    // Single node: pulsing ring and subtitle
    if (graphNodes.length === 1) {
      drawSingleNodeEffects(graphNodes[0]);
    }
  }

  function drawGrid() {
    const c = CONFIG.colors;
    ctx.strokeStyle = hexToRgba(c.border, 0.15);
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = gridSize; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = gridSize; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }

  function drawEdge(edge) {
    const c = CONFIG.colors;
    const a = edge.source;
    const b = edge.target;
    const alpha = Math.min(a.alpha, b.alpha) * 0.6;
    if (alpha < 0.01) return;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = edge.solid ? hexToRgba(c.accent, 0.4) : hexToRgba(c.textMuted, 0.3);
    ctx.lineWidth = CONFIG.connectionWidth;
    if (!edge.solid) {
      ctx.setLineDash([6, 4]);
    }
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Connection label at midpoint
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;
    ctx.font = '10px Inter, -apple-system, sans-serif';
    ctx.fillStyle = hexToRgba(c.textMuted, 0.7);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    // Background pill for label
    const labelText = edge.label || 'LAN';
    const tw = ctx.measureText(labelText).width + 10;
    ctx.fillStyle = hexToRgba(c.bgPrimary, 0.85);
    roundRect(ctx, mx - tw/2, my - 8, tw, 16, 4);
    ctx.fill();
    ctx.fillStyle = hexToRgba(c.textMuted, 0.7);
    ctx.fillText(labelText, mx, my);

    ctx.restore();
  }

  function drawNode(n) {
    const c = CONFIG.colors;
    const d = n.data;
    const isHovered = hoveredNodeId === n.id;
    const isSelected = selectedNodeId === n.id;
    const isLeader = d.is_leader;
    const nw = CONFIG.nodeWidth;
    const nh = getNodeHeight(n);
    const x = n.x - nw / 2;
    const y = n.y - nh / 2;
    const r = CONFIG.nodeRadius;

    ctx.save();
    ctx.globalAlpha = n.alpha;

    // Glow for leader
    if (isLeader) {
      const glowPulse = 0.15 + Math.sin(time * 3) * 0.08;
      ctx.shadowColor = c.accent;
      ctx.shadowBlur = 20 + Math.sin(time * 3) * 5;
      ctx.fillStyle = hexToRgba(c.accent, glowPulse);
      roundRect(ctx, x - 3, y - 3, nw + 6, nh + 6, r + 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Selected glow
    if (isSelected) {
      ctx.shadowColor = c.accent;
      ctx.shadowBlur = 16;
    }

    // Node background
    ctx.fillStyle = isHovered ? c.bgCardHover : c.bgCard;
    roundRect(ctx, x, y, nw, nh, r);
    ctx.fill();

    // Border
    ctx.strokeStyle = isLeader ? c.accent : isSelected ? c.borderActive : c.border;
    ctx.lineWidth = isLeader || isSelected ? 2 : 1;
    roundRect(ctx, x, y, nw, nh, r);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Status indicator dot with pulse
    const statusColor = d.status === 'online' ? c.green : d.status === 'starting' ? c.yellow : c.red;
    const dotX = x + nw - 16;
    const dotY = y + 16;
    // Pulse ring for online
    if (d.status === 'online') {
      const pulseAlpha = 0.3 + Math.sin(time * 4) * 0.2;
      const pulseR = 6 + Math.sin(time * 4) * 3;
      ctx.fillStyle = hexToRgba(statusColor, pulseAlpha);
      ctx.beginPath();
      ctx.arc(dotX, dotY, pulseR, 0, Math.PI * 2);
      ctx.fill();
    }
    // Solid dot
    ctx.fillStyle = statusColor;
    ctx.beginPath();
    ctx.arc(dotX, dotY, 4, 0, Math.PI * 2);
    ctx.fill();

    // Leader crown indicator
    if (isLeader) {
      ctx.fillStyle = c.accent;
      ctx.font = 'bold 9px Inter, -apple-system, sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      // Crown icon as text
      const crownX = x + 10;
      const crownY = y + 6;
      ctx.fillStyle = hexToRgba(c.accent, 0.8);
      ctx.font = '11px Inter, -apple-system, sans-serif';
      ctx.fillText('\u2605', crownX, crownY); // star
      ctx.fillStyle = c.accent;
      ctx.font = 'bold 9px Inter, -apple-system, sans-serif';
      ctx.fillText('LEADER', crownX + 14, crownY + 1);
    }

    // Node name
    const nameY = isLeader ? y + 22 : y + 14;
    ctx.fillStyle = c.textPrimary;
    ctx.font = 'bold 13px Inter, -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    const name = truncate(d.node_name || d.node_id || 'Node', 22);
    ctx.fillText(name, x + 12, nameY);

    // GPU name
    ctx.fillStyle = c.textSecondary;
    ctx.font = '11px Inter, -apple-system, sans-serif';
    const gpuText = truncate(d.gpu_name || 'Unknown GPU', 28);
    ctx.fillText(gpuText, x + 12, nameY + 18);

    // Memory + model line
    const memText = (d.gpu_memory_gb || '?') + ' GB';
    const modelText = truncate(d.model || 'no model', 18);
    ctx.fillStyle = c.textMuted;
    ctx.font = '10px JetBrains Mono, Fira Code, monospace';
    ctx.fillText(memText + '  |  ' + modelText, x + 12, nameY + 34);

    // Shard role badge (visible without hover)
    if (d.shard_role) {
      const badgeY = nameY + 50;
      const badgeColor = d.shard_role === 'head' ? c.green : c.purple;
      ctx.fillStyle = hexToRgba(badgeColor, 0.15);
      roundRect(ctx, x + 10, badgeY - 2, nw - 20, 18, 4);
      ctx.fill();
      ctx.font = 'bold 9px Inter, -apple-system, sans-serif';
      ctx.fillStyle = badgeColor;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(d.shard_role.toUpperCase() + ' | Layers ' + (d.shard_layers || 'all') + ' | ~' + (d.shard_memory_gb || '?') + ' GB', x + 14, badgeY + 1);
    }

    // Hover expanded info
    if (isHovered) {
      const expandY = d.shard_role ? nameY + 72 : nameY + 52;
      ctx.fillStyle = hexToRgba(c.border, 0.5);
      ctx.fillRect(x + 12, expandY - 4, nw - 24, 1);

      ctx.font = '10px Inter, -apple-system, sans-serif';
      ctx.fillStyle = c.textSecondary;
      const gpuUtil = d.gpu_utilization_pct != null ? d.gpu_utilization_pct + '%' : 'N/A';
      const memUsed = d.gpu_memory_used_pct != null ? d.gpu_memory_used_pct + '%' : 'N/A';
      const memLabel = d.unified_memory ? 'unified' : 'VRAM';
      const uptime = d.uptime_seconds ? formatUptime(d.uptime_seconds) : 'N/A';
      ctx.fillText('GPU: ' + gpuUtil + '   Mem: ' + memUsed + ' ' + memLabel, x + 12, expandY + 4);
      ctx.fillText('Uptime: ' + uptime + '   Status: ' + (d.status || 'unknown'), x + 12, expandY + 20);

      // Memory bar
      const barX = x + 12;
      const barY = expandY + 36;
      const barW = nw - 24;
      const barH = 4;
      const pct = d.gpu_memory_used_pct || 0;
      ctx.fillStyle = hexToRgba(c.bgPrimary, 0.8);
      roundRect(ctx, barX, barY, barW, barH, 2);
      ctx.fill();
      const barColor = pct > 90 ? c.red : pct > 70 ? c.yellow : c.green;
      ctx.fillStyle = barColor;
      roundRect(ctx, barX, barY, barW * (pct / 100), barH, 2);
      ctx.fill();
    }

    ctx.restore();
  }

  function drawSingleNodeEffects(n) {
    const c = CONFIG.colors;
    ctx.save();
    ctx.globalAlpha = n.alpha;

    // Pulsing ring
    const pulsePhase = (time * 1.5) % (Math.PI * 2);
    const ringRadius = CONFIG.nodeWidth * 0.75 + Math.sin(pulsePhase) * 10;
    const ringAlpha = 0.1 + Math.sin(pulsePhase) * 0.05;
    ctx.strokeStyle = hexToRgba(c.accent, ringAlpha);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(n.x, n.y, ringRadius, 0, Math.PI * 2);
    ctx.stroke();

    // Second ring offset
    const ring2Radius = CONFIG.nodeWidth * 0.95 + Math.cos(pulsePhase * 0.7) * 15;
    const ring2Alpha = 0.06 + Math.cos(pulsePhase * 0.7) * 0.03;
    ctx.strokeStyle = hexToRgba(c.accent, ring2Alpha);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(n.x, n.y, ring2Radius, 0, Math.PI * 2);
    ctx.stroke();

    // "Waiting for peers..." subtitle
    ctx.font = '12px Inter, -apple-system, sans-serif';
    ctx.fillStyle = hexToRgba(c.textMuted, 0.5 + Math.sin(time * 2) * 0.2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Waiting for peers...', n.x, n.y + CONFIG.nodeHeight / 2 + 20);

    // Scanning dots
    for (let i = 0; i < 3; i++) {
      const dotAngle = time * 0.8 + (i * Math.PI * 2 / 3);
      const dotR = CONFIG.nodeWidth * 0.85;
      const dx = n.x + Math.cos(dotAngle) * dotR;
      const dy = n.y + Math.sin(dotAngle) * dotR;
      const dotAlpha = 0.15 + Math.sin(time * 3 + i) * 0.1;
      ctx.fillStyle = hexToRgba(c.accent, dotAlpha);
      ctx.beginPath();
      ctx.arc(dx, dy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  // --- Utilities ---
  function roundRect(ctx, x, y, w, h, r) {
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

  function truncate(str, max) {
    if (!str) return '';
    return str.length > max ? str.slice(0, max - 1) + '\u2026' : str;
  }

  function formatUptime(seconds) {
    if (seconds < 60) return seconds + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
    return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm';
  }

  // --- Animation Loop ---
  function startLoop() {
    function frame(ts) {
      time = ts * CONFIG.pulseSpeed;
      simulate();
      draw();
      animFrameId = requestAnimationFrame(frame);
    }
    animFrameId = requestAnimationFrame(frame);
  }

  // --- Public API ---
  return {
    init,
    destroy,
    update,
    resize,
    getSelectedNodeId: () => selectedNodeId,
  };
})();
