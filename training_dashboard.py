"""
Nine Men's Morris - Training Dashboard
Reads CSV training logs and displays all important metrics visually.
"""

import os
import csv
import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

LOGS_DIR = Path(__file__).parent / "src" / "logs"

# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def get_log_files():
    """Return sorted list of log files (newest first)."""
    if not LOGS_DIR.exists():
        return []
    files = sorted(LOGS_DIR.glob("*_curriculum.csv"), reverse=True)
    return [{"name": f.name, "path": str(f), "size_kb": round(f.stat().st_size / 1024, 1)} for f in files]


def read_log(path: str):
    """Parse a curriculum CSV log into a list of dicts."""
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                cleaned = {}
                for k, v in row.items():
                    try:
                        cleaned[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                    except (ValueError, TypeError):
                        cleaned[k] = v
                rows.append(cleaned)
    except Exception as e:
        print(f"Error reading log: {e}")
    return rows


def summarise(rows):
    """Build a summary dict from parsed rows."""
    if not rows:
        return {}

    last = rows[-1]
    first = rows[0]

    # Find phase transitions
    phase_transitions = []
    prev_phase = None
    for r in rows:
        p = r.get("phase")
        if p != prev_phase:
            phase_transitions.append({"phase": p, "episode": r.get("episode"), "random_moves": r.get("random_moves")})
            prev_phase = p

    # Find milestones where minimax depth was first beaten
    depth_milestones = {}
    for r in rows:
        d = r.get("minimax_depth_beaten", 0)
        if d not in depth_milestones:
            depth_milestones[d] = r.get("episode")

    # Find clone generation transitions
    clone_gen_transitions = []
    prev_clone_gen = None
    for r in rows:
        cg = r.get("clone_gen", 0)
        if cg != prev_clone_gen:
            clone_gen_transitions.append({"clone_gen": cg, "episode": r.get("episode"), "index": len(clone_gen_transitions)})
            prev_clone_gen = cg

    # Best win rates
    best_wr_mm3 = max((r.get("wr_vs_mm_d3", 0) or 0 for r in rows), default=0)
    best_wr_mm4 = max((r.get("wr_vs_mm_d4", 0) or 0 for r in rows), default=0)
    best_wr_mm5 = max((r.get("wr_vs_mm_d5", 0) or 0 for r in rows), default=0)
    best_wr_mm6 = max((r.get("wr_vs_mm_d6", 0) or 0 for r in rows), default=0)
    best_wr_mm7 = max((r.get("wr_vs_mm_d7", 0) or 0 for r in rows), default=0)

    # Downsample rows for chart (max 400 points to keep response small)
    step = max(1, len(rows) // 400)
    sampled = rows[::step]
    if rows[-1] not in sampled:
        sampled.append(rows[-1])

    return {
        "first_episode": first.get("episode"),
        "last_episode": last.get("episode"),
        "total_steps": last.get("steps"),
        "current_phase": last.get("phase"),
        "current_random_moves": last.get("random_moves"),
        "current_wr": last.get("win_rate"),
        "current_dr": last.get("draw_rate"),
        "current_ema": last.get("ema_return"),
        "current_avg_return": last.get("avg_return"),
        "current_lr": last.get("lr"),
        "minimax_depth_beaten": last.get("minimax_depth_beaten"),
        "clone_gen": last.get("clone_gen"),
        "active_mm_depth": last.get("active_mm_max_depth"),
        "shaping_mult": last.get("shaping_mult"),
        "wr_vs_mm_d1": last.get("wr_vs_mm_d1"),
        "wr_vs_mm_d2": last.get("wr_vs_mm_d2"),
        "wr_vs_mm_d3": last.get("wr_vs_mm_d3"),
        "wr_vs_mm_d4": last.get("wr_vs_mm_d4"),
        "wr_vs_mm_d5": last.get("wr_vs_mm_d5"),
        "wr_vs_mm_d6": last.get("wr_vs_mm_d6"),
        "wr_vs_mm_d7": last.get("wr_vs_mm_d7"),
        "wr_vs_random": last.get("wr_vs_random"),
        "wr_vs_self": last.get("wr_vs_self"),
        "best_wr_mm3": best_wr_mm3,
        "best_wr_mm4": best_wr_mm4,
        "best_wr_mm5": best_wr_mm5,
        "best_wr_mm6": best_wr_mm6,
        "best_wr_mm7": best_wr_mm7,
        "total_rows": len(rows),
        "phase_transitions": phase_transitions,
        "depth_milestones": depth_milestones,
        "clone_gen_transitions": clone_gen_transitions,
        "sampled_rows": sampled,
    }


# ─────────────────────────────────────────────
# HTML Dashboard
# ─────────────────────────────────────────────

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🧠 NMM Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --panel: #1a1d27;
    --border: #2a2d3e;
    --accent: #4f8ef7;
    --green: #4caf50;
    --orange: #ff9800;
    --red: #f44336;
    --yellow: #ffeb3b;
    --text: #e2e8f0;
    --muted: #718096;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; min-height: 100vh; }

  header {
    background: linear-gradient(135deg, #1a1d27, #0f1117);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  header h1 { font-size: 1.4em; color: var(--accent); }
  header .subtitle { color: var(--muted); font-size: 0.9em; }

  .file-selector { display: flex; gap: 8px; align-items: center; margin-left: auto; }
  .file-selector select {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    padding: 8px 12px; border-radius: 8px; font-size: 0.9em; min-width: 280px; cursor: pointer;
  }
  .file-selector select:focus { outline: none; border-color: var(--accent); }
  .btn {
    background: var(--accent); color: #fff; border: none; padding: 8px 16px;
    border-radius: 8px; cursor: pointer; font-size: 0.9em; transition: opacity .2s;
  }
  .btn:hover { opacity: .85; }
  .btn-sm { padding: 5px 10px; font-size: 0.8em; }

  main { padding: 24px; max-width: 1600px; margin: 0 auto; }

  /* ── Summary cards ── */
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 14px;
    display: flex; flex-direction: column; gap: 6px;
  }
  .card .label { font-size: 0.75em; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
  .card .value { font-size: 1.5em; font-weight: 700; }
  .card .sub   { font-size: 0.75em; color: var(--muted); }
  .card.accent  { border-color: var(--accent); }
  .card.green   { border-color: var(--green); }
  .card.orange  { border-color: var(--orange); }
  .card.red     { border-color: var(--red); }
  .card.yellow  { border-color: var(--yellow); }

  /* ── Win‑rate bar section ── */
  .wr-section {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; margin-bottom: 24px;
  }
  .wr-section h2 { font-size: 1em; color: var(--muted); margin-bottom: 16px; text-transform: uppercase; letter-spacing: .5px; }
  .wr-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
  .wr-row { display: flex; flex-direction: column; gap: 4px; }
  .wr-row .wr-label { font-size: 0.85em; display: flex; justify-content: space-between; }
  .wr-bar-bg { height: 10px; background: rgba(255,255,255,.08); border-radius: 99px; overflow: hidden; }
  .wr-bar    { height: 100%; border-radius: 99px; transition: width .6s; }

  /* ── Charts grid ── */
  .charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
    gap: 20px; margin-bottom: 24px;
  }
  .chart-panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px;
  }
  .chart-panel h2 { font-size: 0.95em; color: var(--muted); margin-bottom: 14px; text-transform: uppercase; letter-spacing: .5px; }
  .chart-wrap { position: relative; height: 220px; }
  .zoom-info {
    background: rgba(79, 142, 247, 0.1);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 16px;
    font-size: 0.85em;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .zoom-info .zoom-hint { color: var(--muted); }
  .zoom-info .zoom-level { font-weight: 700; color: var(--accent); }
  .btn-reset {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 4px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.8em;
    transition: all .2s;
  }
  .btn-reset:hover { border-color: var(--accent); color: var(--accent); }

  /* ── Phase timeline ── */
  .timeline-section {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; margin-bottom: 24px;
  }
  .timeline-section h2 { font-size: 0.95em; color: var(--muted); margin-bottom: 16px; text-transform: uppercase; letter-spacing: .5px; }
  .timeline { display: flex; flex-wrap: wrap; gap: 10px; }
  .tl-item {
    background: rgba(255,255,255,.04); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 14px; min-width: 160px;
  }
  .tl-phase { font-weight: 700; font-size: 1.1em; }
  .tl-detail { font-size: 0.78em; color: var(--muted); margin-top: 4px; }

  /* ── Milestones ── */
  .milestone-section {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; margin-bottom: 24px;
  }
  .milestone-section h2 { font-size: 0.95em; color: var(--muted); margin-bottom: 16px; text-transform: uppercase; letter-spacing: .5px; }
  .ms-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
  .ms-item {
    background: rgba(255,255,255,.04); border-radius: 10px; padding: 12px 14px;
  }
  .ms-item .ms-label { font-size: 0.78em; color: var(--muted); }
  .ms-item .ms-val   { font-size: 1.1em; font-weight: 700; margin-top: 4px; }

  /* spinner */
  .spinner-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,.6);
    display: flex; align-items: center; justify-content: center; z-index: 999;
  }
  .spinner {
    width: 48px; height: 48px; border: 5px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%; animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .loading { display: flex; align-items: center; justify-content: center; height: 200px; color: var(--muted); }

  /* phase color helpers */
  .phase-3 { color: #ff7043; }
  .phase-4 { color: #ffa726; }
  .phase-5 { color: var(--yellow); }
  .phase-6 { color: #66bb6a; }
  .phase-7 { color: var(--accent); }

  /* Chart legend toggles */
  .chart-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 12px;
  }
  .chart-legend label {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    font-size: 0.85em;
    color: var(--muted);
    user-select: none;
    padding: 4px 8px;
    border-radius: 6px;
    transition: background-color 0.2s, color 0.2s;
  }
  .chart-legend label:hover {
    background-color: rgba(255, 255, 255, 0.05);
  }
  .chart-legend input[type="checkbox"] {
    accent-color: var(--accent);
    cursor: pointer;
    width: 14px;
    height: 14px;
  }
  .chart-legend .legend-dot {
    font-weight: bold;
  }
</style>
</head>
<body>

<header>
  <div>
    <h1>🧠 Nine Men's Morris — Training Dashboard</h1>
    <div class="subtitle" id="subtitle">Select a log file to begin</div>
  </div>
  <div class="file-selector">
    <select id="logSelect" onchange="loadLog(this.value)">
      <option value="">— choose log file —</option>
    </select>
    <button class="btn" onclick="refreshFiles()">↻ Refresh</button>
  </div>
</header>

<main>
  <div id="content">
    <div class="loading">← Select a log file from the top-right dropdown to start</div>
  </div>
</main>

<!-- Spinner -->
<div class="spinner-overlay" id="spinner" style="display:none"><div class="spinner"></div></div>

<script>
// ─── chart theme ───
Chart.defaults.color = '#718096';
Chart.defaults.borderColor = '#2a2d3e';

const chartInstances = {};
let syncZoomState = { xMin: null, xMax: null, yMin: null, yMax: null };
let isSyncingZoom = false;
let globalLabels = null;
let showCloneLines = true;

function destroyChart(id) {
  if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
}

function applySyncZoom(chart, skipChartId) {
  const xScale = chart.scales.x;

  const newZoomState = {
    xMin: xScale.min,
    xMax: xScale.max
  };

  // Update global sync state
  syncZoomState = newZoomState;
  updateZoomLevelDisplay();

  // Recalc y-axis on source chart to match the new visible x-range
  if (chart._recalcYAxis) {
    chart._recalcYAxis();
    chart.update('none');
  }

  // Apply to all other charts (x-axis), then recompute their y-axis
  isSyncingZoom = true;
  Object.entries(chartInstances).forEach(([id, otherChart]) => {
    if (id !== skipChartId && otherChart) {
      otherChart.zoomScale('x', { min: newZoomState.xMin, max: newZoomState.xMax }, 'none');
      if (otherChart._recalcYAxis) otherChart._recalcYAxis();
      otherChart.update('none');
    }
  });
  isSyncingZoom = false;
}

function makeLineChart(id, labels, datasets, yLabel = '', yMin = null, yMax = null, annotations = []) {
  destroyChart(id);
  const ctx = document.getElementById(id).getContext('2d');
  const chartPanel = document.getElementById(id).closest('.chart-panel');

  // Store original datasets and visibility state
  const originalDatasets = datasets.map(d => ({
    ...d,
    visible: true,
    data: d.data.map((v, i) => ({ x: i, y: v }))
  }));

  // Build annotation config for vertical lines
  const annotationConfig = {};
  annotations.forEach((ann, idx) => {
    annotationConfig[`cloneGenLine${idx}`] = {
      type: 'line',
      xMin: ann.x,
      xMax: ann.x,
      borderColor: ann.color || 'rgba(255, 235, 59, 0.6)',
      borderWidth: 2,
      borderDash: [6, 4],
      label: {
        display: false,
        enabled: false,
      }
    };
  });
  
  // Create legend toggle controls
  const existingLegend = chartPanel.querySelector('.chart-legend');
  if (existingLegend) existingLegend.remove();
  
  const legendDiv = document.createElement('div');
  legendDiv.className = 'chart-legend';
  legendDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;';
  
  datasets.forEach((d, idx) => {
    const toggle = document.createElement('label');
    toggle.style.cssText = 'display:flex;align-items:center;gap:6px;cursor:pointer;font-size:0.85em;color:#718096;user-select:none;';
    toggle.innerHTML = `
      <input type="checkbox" checked data-idx="${idx}" data-chart="${id}" 
        style="accent-color:${d.color};cursor:pointer;width:14px;height:14px;">
      <span style="color:${d.color}">●</span> ${d.label}
    `;
    legendDiv.appendChild(toggle);
  });
  
  chartPanel.querySelector('h2').insertAdjacentElement('afterend', legendDiv);
  
  // Calculate y-axis bounds from visible datasets, optionally filtered by x-range
  function calcYBounds(xRangeMin, xRangeMax) {
    let min = Infinity, max = -Infinity;
    originalDatasets.forEach((d) => {
      if (d.visible) {
        d.data.forEach(pt => {
          if (pt.y == null || !isFinite(pt.y)) return;
          if (xRangeMin != null && pt.x < xRangeMin) return;
          if (xRangeMax != null && pt.x > xRangeMax) return;
          min = Math.min(min, pt.y);
          max = Math.max(max, pt.y);
        });
      }
    });
    if (!isFinite(min) || !isFinite(max)) {
      min = yMin ?? 0;
      max = yMax ?? 1;
    }
    // Add 5% padding (fall back to abs value for a flat line)
    const range = max - min;
    if (range > 0) {
      min -= range * 0.05;
      max += range * 0.05;
    } else {
      const pad = Math.abs(min) * 0.05 || 0.5;
      min -= pad;
      max += pad;
    }
    return { min, max };
  }

  // Recompute y-axis based on the chart's currently visible x-range
  function recalcYAxis() {
    const chart = chartInstances[id];
    if (!chart) return;
    const anyVisible = originalDatasets.some(d => d.visible);
    if (!anyVisible) {
      chart.options.scales.y.min = yMin ?? 0;
      chart.options.scales.y.max = yMax ?? 1;
      return;
    }
    const xScale = chart.scales.x;
    const bounds = calcYBounds(xScale.min, xScale.max);
    chart.options.scales.y.min = bounds.min;
    chart.options.scales.y.max = bounds.max;
  }

  // Update chart visibility and rescale
  function updateChartVisibility() {
    const chart = chartInstances[id];
    if (!chart) return;

    chart.data.datasets.forEach((ds, idx) => {
      ds.hidden = !originalDatasets[idx].visible;
    });

    recalcYAxis();
    chart.update('none');

    // Sync zoom to other charts
    if (!isSyncingZoom) {
      applySyncZoom(chart, id);
    }
  }
  
  // Add toggle event listeners
  legendDiv.addEventListener('change', (e) => {
    if (e.target.tagName === 'INPUT') {
      const idx = parseInt(e.target.dataset.idx);
      originalDatasets[idx].visible = e.target.checked;
      updateChartVisibility();
    }
  });
  
  // Create numeric x-values for proper zoom support
  const numericData = originalDatasets.map(d => ({
    ...d,
    data: d.data
  }));
  
  const initialBounds = calcYBounds();
  const scales = {
    x: {
      type: 'linear',
      ticks: { 
        maxTicksLimit: 8,
        callback: function(val) {
          const idx = Math.round(val);
          return labels[idx] || '';
        }
      },
      grid: { color: '#2a2d3e' },
      min: 0,
      max: labels.length - 1
    },
    y: {
      ticks: { maxTicksLimit: 6 },
      grid: { color: '#2a2d3e' },
      title: { display: !!yLabel, text: yLabel, color: '#718096' },
      min: yMin !== null ? yMin : initialBounds.min,
      max: yMax !== null ? yMax : initialBounds.max,
    }
  };

  chartInstances[id] = new Chart(ctx, {
    type: 'line',
    data: { datasets: numericData.map(d => ({
      data: d.data, label: d.label,
      borderColor: d.color, backgroundColor: d.color + '18',
      borderWidth: 2, pointRadius: 0, tension: 0.3,
      fill: d.fill ?? false,
    }))},
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: function(context) {
              return context.dataset.label + ': ' + context.parsed.y;
            },
            title: function(items) {
              const idx = Math.round(items[0].parsed.x);
              return 'Episode: ' + (labels[idx] || idx);
            }
          }
        },
        annotation: {
          annotations: annotationConfig
        },
        zoom: {
          pan: {
            enabled: true,
            mode: 'x',
            modifierKey: null,
            onPanComplete: (ctx) => {
              if (!isSyncingZoom) {
                applySyncZoom(ctx.chart, id);
              }
            },
          },
          zoom: {
            wheel: {
              enabled: false,
            },
            drag: {
              enabled: true,
              backgroundColor: 'rgba(79, 142, 247, 0.2)',
              borderColor: 'rgba(79, 142, 247, 0.6)',
              borderWidth: 1,
            },
            pinch: {
              enabled: false,
            },
            mode: 'x',
            onZoomComplete: (ctx) => {
              if (!isSyncingZoom) {
                applySyncZoom(ctx.chart, id);
              }
            },
          },
        },
      },
      scales
    }
  });

  chartInstances[id]._recalcYAxis = recalcYAxis;
}

// ─── Load list of log files ───
async function refreshFiles() {
  const res = await fetch('/api/logs');
  const logs = await res.json();
  const sel = document.getElementById('logSelect');
  sel.innerHTML = '<option value="">— choose log file —</option>' +
    logs.map(l => `<option value="${l.path}">${l.name} (${l.size_kb} KB)</option>`).join('');

  // Auto-load newest
  if (logs.length > 0) {
    sel.value = logs[0].path;
    loadLog(logs[0].path);
  }
}

// ─── Load and render a log ───
async function loadLog(path) {
  if (!path) return;
  document.getElementById('spinner').style.display = 'flex';
  document.getElementById('content').innerHTML = '';

  try {
    const res = await fetch('/api/log_data?path=' + encodeURIComponent(path));
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    render(data);
  } catch(e) {
    document.getElementById('content').innerHTML = `<div class="loading" style="color:#f44336">Error loading data: ${e}</div>`;
  } finally {
    document.getElementById('spinner').style.display = 'none';
  }
}

// ─── Render everything ───
function render(d) {
  const s = d.summary;
  const rows = d.summary.sampled_rows;

  // subtitle
  document.getElementById('subtitle').textContent =
    `${Math.round(s.total_steps/1e6)}M steps · ${s.total_rows} checkpoints · Phase ${s.current_phase}`;

  // Reset sync zoom state
  syncZoomState = { xMin: null, xMax: null, yMin: null, yMax: null };

  // Build clone generation annotations (vertical lines)
  const cloneGenAnnotations = [];
  console.log('showCloneLines:', showCloneLines);
  if (showCloneLines) {
    const cloneGenTransitions = s.clone_gen_transitions || [];
    console.log('Building clone gen annotations:', cloneGenTransitions.length);
    cloneGenTransitions.forEach((t, idx) => {
      // Find the index in the sampled rows that corresponds to this transition
      const targetEpisode = t.episode;
      // Find closest index in the sampled rows
      let closestIdx = 0;
      let minDist = Infinity;
      for (let i = 0; i < rows.length; i++) {
        const dist = Math.abs(rows[i].episode - targetEpisode);
        if (dist < minDist) {
          minDist = dist;
          closestIdx = i;
        }
      }
      // Only add annotation if it's within a reasonable distance
      if (minDist < (rows.length > 1 ? Math.abs(rows[1].episode - rows[0].episode) * 2 : 1000000)) {
        cloneGenAnnotations.push({
          x: closestIdx,
          color: 'rgba(255, 235, 59, 0.7)'
        });
      }
    });
  }
  console.log('Clone gen annotations count:', cloneGenAnnotations.length);
  
  // Build HTML
  const html = `
    ${renderCards(s)}
    ${renderWinRates(s)}
    ${renderTimeline(s)}
    ${renderMilestones(s)}
    <div class="zoom-info">
      <span class="zoom-hint">🔍 Zoom: drag to select area on any graph • Pan: right-click drag • <span class="zoom-level" id="zoomLevel">All charts synchronized</span></span>
      <div style="display:flex;gap:8px;align-items:center">
        <label style="display:flex;align-items:center;gap:6px;font-size:0.85em;color:var(--muted);cursor:pointer">
          <input type="checkbox" id="toggleCloneLines" ${showCloneLines ? 'checked' : ''} onchange="toggleCloneLines(this.checked)">
          Clone Lines
        </label>
        <button class="btn-reset" onclick="resetAllZoom()">↺ Reset Zoom</button>
      </div>
    </div>
    <div class="charts-grid">
      <div class="chart-panel">
        <h2>Win Rate vs Opponents</h2>
        <div class="chart-wrap"><canvas id="chartWR"></canvas></div>
      </div>
      <div class="chart-panel">
        <h2>EMA Return &amp; Average Return</h2>
        <div class="chart-wrap"><canvas id="chartReturn"></canvas></div>
      </div>
      <div class="chart-panel">
        <h2>Policy Loss &amp; Value Loss</h2>
        <div class="chart-wrap"><canvas id="chartLoss"></canvas></div>
      </div>
      <div class="chart-panel">
        <h2>Entropy &amp; KL Divergence</h2>
        <div class="chart-wrap"><canvas id="chartEntropy"></canvas></div>
      </div>
      <div class="chart-panel">
        <h2>Win Rate vs Self-Play</h2>
        <div class="chart-wrap"><canvas id="chartSelf"></canvas></div>
      </div>
      <div class="chart-panel">
        <h2>Clip Fraction &amp; Gradient Norm</h2>
        <div class="chart-wrap"><canvas id="chartClip"></canvas></div>
      </div>
      <div class="chart-panel">
        <h2>Learning Rate</h2>
        <div class="chart-wrap"><canvas id="chartLR"></canvas></div>
      </div>
      <div class="chart-panel">
        <h2>Shaping Multiplier</h2>
        <div class="chart-wrap"><canvas id="chartShaping"></canvas></div>
      </div>
      <div class="chart-panel">
        <h2>Win Rate &amp; Draw Rate (Overall)</h2>
        <div class="chart-wrap"><canvas id="chartWRDraw"></canvas></div>
      </div>
      <div class="chart-panel">
        <h2>Clone Gen &amp; MM Depth Beaten</h2>
        <div class="chart-wrap"><canvas id="chartProgress"></canvas></div>
      </div>
    </div>
  `;
  document.getElementById('content').innerHTML = html;

  // Build chart data
  globalLabels = rows.map(r => fmtEp(r.episode));
  const labels = globalLabels;

  // Win rates chart
  makeLineChart('chartWR', labels, [
    { label: 'vs MM-D1', data: rows.map(r => r.wr_vs_mm_d1), color: '#4f8ef7' },
    { label: 'vs MM-D2', data: rows.map(r => r.wr_vs_mm_d2), color: '#66bb6a' },
    { label: 'vs MM-D3', data: rows.map(r => r.wr_vs_mm_d3), color: '#ffb74d' },
    { label: 'vs MM-D4', data: rows.map(r => r.wr_vs_mm_d4), color: '#ef5350' },
    { label: 'vs MM-D5', data: rows.map(r => r.wr_vs_mm_d5), color: '#ec407a' },
    { label: 'vs MM-D6', data: rows.map(r => r.wr_vs_mm_d6), color: '#7e57c2' },
    { label: 'vs MM-D7', data: rows.map(r => r.wr_vs_mm_d7), color: '#26a69a' },
    { label: 'vs Random', data: rows.map(r => r.wr_vs_random), color: '#ab47bc' },
  ], 'Win Rate', 0, 1, cloneGenAnnotations);

  // Return chart
  makeLineChart('chartReturn', labels, [
    { label: 'EMA Return', data: rows.map(r => r.ema_return), color: '#4f8ef7', fill: true },
    { label: 'Avg Return', data: rows.map(r => r.avg_return), color: '#ffb74d' },
    { label: 'Win Rate',   data: rows.map(r => r.win_rate),   color: '#66bb6a' },
  ], 'Return', null, null, cloneGenAnnotations);

  // Loss chart (clip extreme values)
  makeLineChart('chartLoss', labels, [
    { label: 'Policy Loss', data: rows.map(r => clamp(r.policy_loss, -1, 1)), color: '#ef5350' },
    { label: 'Value Loss',  data: rows.map(r => clamp(r.value_loss,  0, 2)),  color: '#ffb74d' },
  ], 'Loss', null, null, cloneGenAnnotations);

  // Entropy + KL
  makeLineChart('chartEntropy', labels, [
    { label: 'Entropy', data: rows.map(r => r.entropy), color: '#4f8ef7' },
    { label: 'KL Div',  data: rows.map(r => clamp(r.kl_div, 0, 0.2)), color: '#ab47bc' },
  ], '', null, null, cloneGenAnnotations);

  // Self-play WR
  makeLineChart('chartSelf', labels, [
    { label: 'vs Self', data: rows.map(r => r.wr_vs_self), color: '#26c6da', fill: true },
  ], 'Win Rate', 0, 1, cloneGenAnnotations);

  // Clip + grad norm
  makeLineChart('chartClip', labels, [
    { label: 'Clip Frac', data: rows.map(r => r.clip_frac),  color: '#ffb74d' },
    { label: 'Grad Norm', data: rows.map(r => clamp(r.grad_norm, 0, 5)), color: '#ef5350' },
  ], '', null, null, cloneGenAnnotations);

  // Learning Rate (log scale via raw values, clamp to avoid nulls)
  makeLineChart('chartLR', labels, [
    { label: 'Learning Rate', data: rows.map(r => clamp(r.lr, 0, 1e-3)), color: '#26c6da', fill: true },
  ], 'LR', null, null, cloneGenAnnotations);

  // Shaping Multiplier
  makeLineChart('chartShaping', labels, [
    { label: 'Shaping Mult', data: rows.map(r => r.shaping_mult), color: '#ab47bc', fill: true },
  ], 'Multiplier', 0, null, cloneGenAnnotations);

  // Overall Win Rate + Draw Rate
  makeLineChart('chartWRDraw', labels, [
    { label: 'Win Rate',  data: rows.map(r => r.win_rate),  color: '#66bb6a', fill: false },
    { label: 'Draw Rate', data: rows.map(r => r.draw_rate), color: '#ffb74d', fill: false },
    { label: 'Loss Rate', data: rows.map(r => clamp(1 - (r.win_rate||0) - (r.draw_rate||0), 0, 1)), color: '#ef5350', fill: false },
  ], 'Rate', 0, 1, cloneGenAnnotations);

  // Clone generation + minimax depth beaten (discrete step lines)
  makeLineChart('chartProgress', labels, [
    { label: 'MM Depth Beaten',  data: rows.map(r => r.minimax_depth_beaten),  color: '#4f8ef7' },
    { label: 'Active MM Depth',  data: rows.map(r => r.active_mm_max_depth),   color: '#ffb74d' },
  ], 'Level', null, null, cloneGenAnnotations);
}

function renderCards(s) {
  const pct = v => v != null ? (v * 100).toFixed(1) + '%' : '—';
  const fmt = v => v != null ? v.toLocaleString() : '—';
  const fmtM = v => v != null ? (v/1e6).toFixed(1) + 'M' : '—';
  return `<div class="cards">
    <div class="card accent">
      <div class="label">Current Phase</div>
      <div class="value phase-${s.current_phase}">${s.current_phase}</div>
      <div class="sub">${s.current_random_moves} random moves</div>
    </div>
    <div class="card green">
      <div class="label">Win Rate</div>
      <div class="value">${pct(s.current_wr)}</div>
      <div class="sub">Draw: ${pct(s.current_dr)}</div>
    </div>
    <div class="card accent">
      <div class="label">EMA Return</div>
      <div class="value">${s.current_ema != null ? s.current_ema.toFixed(3) : '—'}</div>
      <div class="sub">avg: ${s.current_avg_return != null ? s.current_avg_return.toFixed(3) : '—'}</div>
    </div>
    <div class="card orange">
      <div class="label">Total Steps</div>
      <div class="value">${fmtM(s.total_steps)}</div>
      <div class="sub">eps: ${fmtM(s.last_episode)}</div>
    </div>
    <div class="card yellow">
      <div class="label">MM Depth Beaten</div>
      <div class="value">${s.minimax_depth_beaten ?? '—'}</div>
      <div class="sub">active max: ${s.active_mm_depth ?? '—'}</div>
    </div>
    <div class="card">
      <div class="label">Clone Gen</div>
      <div class="value">${s.clone_gen ?? '—'}</div>
      <div class="sub">opponent clones</div>
    </div>
    <div class="card">
      <div class="label">Shaping Mult</div>
      <div class="value">${s.shaping_mult != null ? s.shaping_mult.toFixed(3) : '—'}</div>
      <div class="sub">reward shaping</div>
    </div>
    <div class="card green">
      <div class="label">vs Random</div>
      <div class="value">${pct(s.wr_vs_random)}</div>
      <div class="sub">WR</div>
    </div>
    <div class="card">
      <div class="label">vs Self</div>
      <div class="value">${pct(s.wr_vs_self)}</div>
      <div class="sub">WR self-play</div>
    </div>
    <div class="card">
      <div class="label">Learning Rate</div>
      <div class="value" style="font-size:1.1em">${s.current_lr != null ? s.current_lr.toExponential(1) : '—'}</div>
      <div class="sub">current LR</div>
    </div>
    <div class="card">
      <div class="label">Checkpoints</div>
      <div class="value">${fmt(s.total_rows)}</div>
      <div class="sub">log entries</div>
    </div>
  </div>`;
}

function renderWinRates(s) {
  const pct = v => v != null ? (v * 100).toFixed(1) : '0';
  const bar = (v, color) => {
    const w = Math.min(100, Math.max(0, (v ?? 0) * 100));
    return `<div class="wr-bar-bg"><div class="wr-bar" style="width:${w}%;background:${color}"></div></div>`;
  };
  const entries = [
    { label: 'vs Minimax Depth 1', val: s.wr_vs_mm_d1, color: '#4f8ef7' },
    { label: 'vs Minimax Depth 2', val: s.wr_vs_mm_d2, color: '#66bb6a' },
    { label: 'vs Minimax Depth 3', val: s.wr_vs_mm_d3, color: '#ffb74d' },
    { label: 'vs Minimax Depth 4', val: s.wr_vs_mm_d4, color: '#ef5350' },
    { label: 'vs Minimax Depth 5', val: s.wr_vs_mm_d5, color: '#ec407a' },
    { label: 'vs Minimax Depth 6', val: s.wr_vs_mm_d6, color: '#7e57c2' },
    { label: 'vs Minimax Depth 7', val: s.wr_vs_mm_d7, color: '#26a69a' },
    { label: 'vs Random',          val: s.wr_vs_random, color: '#ab47bc' },
    { label: 'vs Self-Play',       val: s.wr_vs_self,  color: '#26c6da' },
  ];
  const rows = entries.map(e => `
    <div class="wr-row">
      <div class="wr-label"><span>${e.label}</span><span style="color:${e.color};font-weight:700">${pct(e.val)}%</span></div>
      ${bar(e.val, e.color)}
    </div>`).join('');
  return `<div class="wr-section"><h2>Current Win Rates (latest checkpoint)</h2><div class="wr-grid">${rows}</div></div>`;
}

function renderTimeline(s) {
  const phaseNames = { 3: 'Phase 3', 4: 'Phase 4', 5: 'Phase 5', 6: 'Phase 6', 7: 'Phase 7' };
  const items = (s.phase_transitions || []).map(t => `
    <div class="tl-item">
      <div class="tl-phase phase-${t.phase}">${phaseNames[t.phase] || 'Phase ' + t.phase}</div>
      <div class="tl-detail">Started: ep ${fmtEp(t.episode)}</div>
      <div class="tl-detail">${t.random_moves} random moves</div>
    </div>`).join('');
  return `<div class="timeline-section"><h2>Curriculum Phase Transitions</h2><div class="timeline">${items}</div></div>`;
}

function renderMilestones(s) {
  const dm = s.depth_milestones || {};
  const items = Object.entries(dm)
    .filter(([d]) => d > 0)
    .sort((a, b) => a[0] - b[0])
    .map(([d, ep]) => `
      <div class="ms-item">
        <div class="ms-label">Minimax Depth ${d} first beaten</div>
        <div class="ms-val" style="color:#ffb74d">ep ${fmtEp(ep)}</div>
      </div>`).join('');

  const bestItems = `
    <div class="ms-item">
      <div class="ms-label">Best WR vs MM-D3 (all time)</div>
      <div class="ms-val" style="color:#66bb6a">${((s.best_wr_mm3 || 0) * 100).toFixed(1)}%</div>
    </div>
    <div class="ms-item">
      <div class="ms-label">Best WR vs MM-D4 (all time)</div>
      <div class="ms-val" style="color:#ef5350">${((s.best_wr_mm4 || 0) * 100).toFixed(1)}%</div>
    </div>
    <div class="ms-item">
      <div class="ms-label">Best WR vs MM-D5 (all time)</div>
      <div class="ms-val" style="color:#ec407a">${((s.best_wr_mm5 || 0) * 100).toFixed(1)}%</div>
    </div>
    <div class="ms-item">
      <div class="ms-label">Best WR vs MM-D6 (all time)</div>
      <div class="ms-val" style="color:#7e57c2">${((s.best_wr_mm6 || 0) * 100).toFixed(1)}%</div>
    </div>
    <div class="ms-item">
      <div class="ms-label">Best WR vs MM-D7 (all time)</div>
      <div class="ms-val" style="color:#26a69a">${((s.best_wr_mm7 || 0) * 100).toFixed(1)}%</div>
    </div>`;

  return `<div class="milestone-section"><h2>Milestones</h2><div class="ms-grid">${items}${bestItems}</div></div>`;
}

// ─── helpers ───
function fmtEp(v) {
  if (v == null) return '';
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
  return String(v);
}

function clamp(v, lo, hi) {
  if (v == null || !isFinite(v)) return null;
  return Math.max(lo, Math.min(hi, v));
}

function resetAllZoom() {
  isSyncingZoom = true;
  Object.values(chartInstances).forEach(chart => {
    if (chart) {
      chart.resetZoom('none');
      if (chart._recalcYAxis) {
        chart._recalcYAxis();
        chart.update('none');
      }
    }
  });
  isSyncingZoom = false;
  syncZoomState = { xMin: null, xMax: null, yMin: null, yMax: null };
  updateZoomLevelDisplay();
}

function updateZoomLevelDisplay() {
  const zoomEl = document.getElementById('zoomLevel');
  if (zoomEl && syncZoomState.xMin !== null && syncZoomState.xMax !== null && globalLabels) {
    const zoomPercent = ((syncZoomState.xMax - syncZoomState.xMin) / (globalLabels.length || 1) * 100).toFixed(1);
    zoomEl.textContent = `Zoom: ${zoomPercent}% of data visible`;
  } else {
    zoomEl.textContent = 'All charts synchronized';
  }
}

function toggleCloneLines(checked) {
  showCloneLines = checked;
  console.log('Clone lines toggled:', showCloneLines);
  // Re-render current chart by triggering a re-load
  const select = document.getElementById('logSelect');
  if (select.value) {
    loadLog(select.value);
  }
}

// ─── init ───
window.addEventListener('DOMContentLoaded', refreshFiles);
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return DASHBOARD_HTML


@app.route('/api/logs')
def api_logs():
    return jsonify(get_log_files())


@app.route('/api/log_data')
def api_log_data():
    path = request.args.get('path', '')
    if not path or not Path(path).exists():
        return jsonify({'error': f'File not found: {path}'})
    rows = read_log(path)
    if not rows:
        return jsonify({'error': 'Empty or unreadable log file'})
    summary = summarise(rows)
    # Don't send raw rows to avoid huge payload; sampled_rows is already in summary
    return jsonify({'summary': summary})


if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = '0.0.0.0'

    print("=" * 60)
    print("Nine Men's Morris — Training Dashboard")
    print("=" * 60)
    log_files = get_log_files()
    print(f"Found {len(log_files)} log file(s) in {LOGS_DIR}")
    for lf in log_files[:5]:
        print(f"  • {lf['name']} ({lf['size_kb']} KB)")
    print()
    print(f"Open http://{local_ip}:7860 in your browser")
    print("=" * 60)

    app.run(host='0.0.0.0', port=7860, debug=False)


#AXSIWOkDrzSgENojGzLywCqRL14OoHRITzGqBAqyL5MhVVY8#whAa2xh4tpNpLyBwFEYDpifW0skCtrCMUe6A6VWt9XU