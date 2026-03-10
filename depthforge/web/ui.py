"""
depthforge.web.ui
==================
Single-page HTML/JS UI for the DepthForge preview server.
Served at GET / by the Flask app.
"""

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DepthForge Preview</title>
<style>
  :root {
    --bg: #0e1117;
    --surface: #161b27;
    --surface2: #1e2535;
    --border: #2a3347;
    --accent: #4f8ef7;
    --accent2: #7c5cbf;
    --green: #3ecf8e;
    --red: #f56565;
    --yellow: #ecc94b;
    --text: #e2e8f0;
    --muted: #718096;
    --radius: 8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 14px;
    min-height: 100vh;
    display: grid;
    grid-template-rows: 52px 1fr;
    grid-template-columns: 320px 1fr;
    grid-template-areas: "header header" "sidebar main";
  }

  /* ── Header ─────────────────────────────────────────────── */
  header {
    grid-area: header;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 12px;
  }
  header .logo {
    font-weight: 700;
    font-size: 18px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
  }
  header .version { color: var(--muted); font-size: 11px; }
  header .spacer  { flex: 1; }
  header .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
  }
  header .status-dot.offline { background: var(--red); box-shadow: none; }

  /* ── Sidebar ─────────────────────────────────────────────── */
  #sidebar {
    grid-area: sidebar;
    background: var(--surface);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .section {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
  }
  .section-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--muted);
    margin-bottom: 10px;
  }

  label {
    display: block;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 3px;
    margin-top: 8px;
  }
  label:first-child { margin-top: 0; }

  select, input[type=range], input[type=number], input[type=text] {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text);
    padding: 5px 8px;
    font-size: 13px;
    outline: none;
    transition: border-color 0.15s;
  }
  select:focus, input:focus { border-color: var(--accent); }

  .range-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .range-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .range-val {
    min-width: 38px;
    text-align: right;
    font-size: 12px;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
  }

  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
  }
  .checkbox-row input[type=checkbox] { accent-color: var(--accent); }
  .checkbox-row span { font-size: 13px; color: var(--text); }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 8px 16px;
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    transition: opacity 0.15s, transform 0.1s;
    width: 100%;
    margin-top: 6px;
  }
  .btn:active { transform: scale(0.97); }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
  }
  .btn-secondary {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
  }
  .btn-green  { background: var(--green);  color: #0e1117; }
  .btn-danger { background: var(--red);    color: #fff; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* ── Pattern grid ──────────────────────────────────────── */
  #pattern-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5px;
    max-height: 200px;
    overflow-y: auto;
  }
  .pattern-thumb {
    border: 2px solid transparent;
    border-radius: 5px;
    cursor: pointer;
    overflow: hidden;
    aspect-ratio: 1;
    transition: border-color 0.15s;
  }
  .pattern-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .pattern-thumb.active { border-color: var(--accent); }
  .pattern-thumb:hover  { border-color: var(--accent2); }

  /* ── Main ────────────────────────────────────────────────── */
  #main {
    grid-area: main;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Tabs */
  .tab-bar {
    display: flex;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 20px;
    gap: 2px;
  }
  .tab {
    padding: 14px 18px;
    font-size: 13px;
    font-weight: 500;
    color: var(--muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
  }
  .tab:hover   { color: var(--text); }
  .tab.active  { color: var(--accent); border-bottom-color: var(--accent); }

  .tab-content {
    flex: 1;
    overflow: auto;
    padding: 24px;
    display: none;
  }
  .tab-content.active { display: flex; flex-direction: column; gap: 20px; }

  /* Result viewer */
  #result-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    min-height: 400px;
    position: relative;
    overflow: hidden;
  }
  #result-panel .placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: var(--muted);
  }
  #result-panel .placeholder .icon { font-size: 48px; }
  #result-img {
    max-width: 100%;
    max-height: 65vh;
    border-radius: 4px;
    display: none;
  }

  /* Progress bar */
  #progress-bar-wrap {
    width: 100%;
    background: var(--border);
    border-radius: 4px;
    height: 4px;
    display: none;
  }
  #progress-bar {
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    width: 0%;
    transition: width 0.3s;
  }
  #progress-msg { font-size: 12px; color: var(--muted); margin-top: 6px; }

  /* Result meta */
  #result-meta {
    display: none;
    gap: 20px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--muted);
  }
  #result-meta span b { color: var(--text); }

  /* Analyze panel */
  #analyze-result {
    display: none;
    flex-direction: column;
    gap: 12px;
  }
  .comfort-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 14px;
  }
  .comfort-badge.pass { background: rgba(62,207,142,0.15); color: var(--green); border: 1px solid var(--green); }
  .comfort-badge.fail { background: rgba(245,101,101,0.15); color: var(--red);   border: 1px solid var(--red); }

  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 10px;
  }
  .stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px;
  }
  .stat-card .val { font-size: 22px; font-weight: 700; color: var(--accent); }
  .stat-card .lbl { font-size: 11px; color: var(--muted); margin-top: 2px; }

  .advice-list { list-style: none; display: flex; flex-direction: column; gap: 6px; }
  .advice-list li {
    padding: 8px 12px;
    background: var(--surface2);
    border-left: 3px solid var(--yellow);
    border-radius: 0 6px 6px 0;
    font-size: 12px;
    color: var(--text);
  }

  .qc-images {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .qc-images img {
    width: 100%;
    border-radius: var(--radius);
    border: 1px solid var(--border);
  }
  .qc-images .img-label {
    font-size: 11px;
    color: var(--muted);
    text-align: center;
    margin-top: 4px;
  }

  /* Presets panel */
  .preset-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
  }
  .preset-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
    cursor: pointer;
    transition: border-color 0.15s, transform 0.1s;
  }
  .preset-card:hover  { border-color: var(--accent2); transform: translateY(-2px); }
  .preset-card.active { border-color: var(--accent); background: rgba(79,142,247,0.08); }
  .preset-card .name  { font-weight: 700; font-size: 14px; margin-bottom: 4px; }
  .preset-card .desc  { font-size: 12px; color: var(--muted); line-height: 1.5; }
  .preset-card .tags  { margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; }
  .tag {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(79,142,247,0.15);
    color: var(--accent);
  }

  /* Toast */
  #toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    padding: 10px 18px;
    border-radius: var(--radius);
    font-size: 13px;
    font-weight: 500;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s;
    z-index: 9999;
  }
  #toast.show    { opacity: 1; }
  #toast.success { background: var(--green); color: #0e1117; }
  #toast.error   { background: var(--red);   color: #fff; }
  #toast.info    { background: var(--accent); color: #fff; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<!-- ── Header ──────────────────────────────────────────────────────── -->
<header>
  <span class="logo">⬡ DepthForge</span>
  <span class="version">v0.3.0</span>
  <div class="spacer"></div>
  <span id="status-label" style="font-size:12px;color:var(--muted)">Connecting…</span>
  <div id="status-dot" class="status-dot offline"></div>
</header>

<!-- ── Sidebar ─────────────────────────────────────────────────────── -->
<div id="sidebar">

  <!-- Mode -->
  <div class="section">
    <div class="section-title">Mode</div>
    <select id="mode">
      <option value="texture">Texture Stereogram</option>
      <option value="sirds">SIRDS (Random Dot)</option>
      <option value="anaglyph">Anaglyph 3D</option>
      <option value="hidden">Hidden Image</option>
    </select>
    <label>Preset</label>
    <select id="preset">
      <option value="none">Custom</option>
    </select>
  </div>

  <!-- Depth -->
  <div class="section">
    <div class="section-title">Depth Map</div>
    <input type="file" id="depth-file" accept="image/*"
      style="font-size:12px;color:var(--muted);width:100%;padding:5px 0;background:none;border:none;cursor:pointer;">
    <div id="depth-thumb" style="display:none;margin-top:8px;">
      <img id="depth-thumb-img" style="width:100%;border-radius:5px;border:1px solid var(--border);">
    </div>
    <label>Depth Factor</label>
    <div class="range-row">
      <input type="range" id="depth-factor" min="0.0" max="1.0" step="0.01" value="0.35">
      <span class="range-val" id="depth-factor-val">0.35</span>
    </div>
    <label>Max Parallax</label>
    <div class="range-row">
      <input type="range" id="max-parallax" min="0.005" max="0.1" step="0.001" value="0.033">
      <span class="range-val" id="max-parallax-val">0.033</span>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="safe-mode">
      <span>Safe Mode (hard clamp)</span>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="invert-depth">
      <span>Invert Depth</span>
    </div>
  </div>

  <!-- Pattern -->
  <div class="section" id="pattern-section">
    <div class="section-title">Pattern</div>
    <div id="pattern-grid"></div>
    <label style="margin-top:10px;">Tile Size</label>
    <div class="range-row">
      <input type="range" id="tile-size" min="32" max="256" step="16" value="128">
      <span class="range-val" id="tile-size-val">128</span>
    </div>
    <label>Pattern Scale</label>
    <div class="range-row">
      <input type="range" id="pattern-scale" min="0.5" max="4.0" step="0.1" value="1.0">
      <span class="range-val" id="pattern-scale-val">1.0</span>
    </div>
    <label>Seed</label>
    <input type="number" id="seed" value="42" min="0" max="99999">
  </div>

  <!-- Hidden image options -->
  <div class="section" id="hidden-section" style="display:none;">
    <div class="section-title">Hidden Image</div>
    <label>Shape</label>
    <select id="hidden-shape">
      <option value="star">Star</option>
      <option value="circle">Circle</option>
      <option value="square">Square</option>
      <option value="triangle">Triangle</option>
      <option value="diamond">Diamond</option>
      <option value="arrow">Arrow</option>
    </select>
    <label>Text (overrides shape)</label>
    <input type="text" id="hidden-text" placeholder="e.g. HI">
    <label>Foreground Depth</label>
    <div class="range-row">
      <input type="range" id="fg-depth" min="0.1" max="1.0" step="0.05" value="0.75">
      <span class="range-val" id="fg-depth-val">0.75</span>
    </div>
  </div>

  <!-- Generate button -->
  <button class="btn btn-primary" id="btn-generate">
    <span>⬡</span> Generate Stereogram
  </button>
  <button class="btn btn-secondary" id="btn-download" disabled>
    ↓ Download PNG
  </button>

</div>

<!-- ── Main ────────────────────────────────────────────────────────── -->
<div id="main">
  <div class="tab-bar">
    <div class="tab active" data-tab="generate">Generate</div>
    <div class="tab" data-tab="analyze">Analyze</div>
    <div class="tab" data-tab="patterns">Patterns</div>
    <div class="tab" data-tab="presets">Presets</div>
  </div>

  <!-- GENERATE TAB -->
  <div class="tab-content active" id="tab-generate">
    <div id="result-panel">
      <div class="placeholder" id="result-placeholder">
        <span class="icon">⬡</span>
        <span>Upload a depth map and click Generate</span>
        <span style="font-size:12px;color:var(--muted)">Supports PNG, JPG, TIFF, EXR</span>
      </div>
      <div id="progress-bar-wrap">
        <div id="progress-bar"></div>
      </div>
      <p id="progress-msg"></p>
      <img id="result-img" alt="Stereogram output">
    </div>
    <div id="result-meta">
      <span><b id="meta-size">–</b> size</span>
      <span><b id="meta-mode">–</b> mode</span>
      <span><b id="meta-depth">–</b> depth factor</span>
      <span><b id="meta-parallax">–</b> max parallax</span>
      <span><b id="meta-preset">–</b> preset</span>
      <span id="meta-time"></span>
    </div>
  </div>

  <!-- ANALYZE TAB -->
  <div class="tab-content" id="tab-analyze">
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <input type="file" id="analyze-file" accept="image/*"
        style="font-size:12px;color:var(--muted);flex:1;padding:5px;background:var(--surface2);
               border:1px solid var(--border);border-radius:var(--radius);cursor:pointer;">
      <button class="btn btn-primary" id="btn-analyze" style="width:auto;min-width:140px;">
        Run Analysis
      </button>
    </div>
    <div id="analyze-result">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <div class="comfort-badge" id="comfort-badge">–</div>
        <div style="font-size:13px;color:var(--muted)">Comfort score: <b id="comfort-score" style="color:var(--text)">–</b></div>
        <div style="font-size:13px;color:var(--muted)">Strain: <b id="strain-rating" style="color:var(--text)">–</b></div>
      </div>
      <div class="stat-grid">
        <div class="stat-card"><div class="val" id="stat-parallax">–</div><div class="lbl">Max parallax (px)</div></div>
        <div class="stat-card"><div class="val" id="stat-vergence">–</div><div class="lbl">Max vergence (°)</div></div>
        <div class="stat-card"><div class="val" id="stat-pct">–</div><div class="lbl">% uncomfortable</div></div>
        <div class="stat-card"><div class="val" id="stat-score">–</div><div class="lbl">Overall score</div></div>
      </div>
      <div>
        <p style="font-size:12px;font-weight:600;color:var(--muted);margin-bottom:8px;">ADVICE</p>
        <ul class="advice-list" id="advice-list"></ul>
      </div>
      <div class="qc-images">
        <div>
          <img id="heatmap-img" alt="Parallax heatmap">
          <div class="img-label">Parallax Heatmap</div>
        </div>
        <div>
          <img id="overlay-img" alt="Violation overlay">
          <div class="img-label">Violation Overlay</div>
        </div>
      </div>
    </div>
  </div>

  <!-- PATTERNS TAB -->
  <div class="tab-content" id="tab-patterns">
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
      <select id="pattern-category-filter" style="width:180px;">
        <option value="">All Categories</option>
      </select>
      <input type="range" id="pattern-preview-size" min="64" max="256" step="32" value="96"
        style="width:120px;accent-color:var(--accent);">
      <span style="font-size:12px;color:var(--muted)" id="preview-size-lbl">96px</span>
    </div>
    <div id="all-patterns-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:12px;"></div>
  </div>

  <!-- PRESETS TAB -->
  <div class="tab-content" id="tab-presets">
    <div class="preset-cards" id="preset-cards"></div>
  </div>
</div>

<!-- Toast -->
<div id="toast"></div>

<script>
/* ─────────────────────────────────────────────────────────────
   DepthForge Preview UI — vanilla JS, no dependencies
   ───────────────────────────────────────────────────────────── */

const API = '';   // same-origin

// ── State ────────────────────────────────────────────────────
let state = {
  depthB64:        null,
  resultB64:       null,
  selectedPattern: null,
  patterns:        [],
  presets:         [],
};

// ── Utils ────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className   = `show ${type}`;
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.classList.remove('show'), 3000);
}

function post(url, body) {
  return fetch(url, {
    method:  'POST',
    headers: {'Content-Type': 'application/json'},
    body:    JSON.stringify(body),
  }).then(r => r.json());
}

function get(url) { return fetch(url).then(r => r.json()); }

function fileToB64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload  = () => res(r.result.split(',')[1]);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

// ── Range inputs ─────────────────────────────────────────────
document.querySelectorAll('input[type=range]').forEach(el => {
  const valEl = document.getElementById(el.id + '-val');
  if (valEl) {
    el.addEventListener('input', () => { valEl.textContent = parseFloat(el.value).toFixed(el.step.includes('.') ? el.step.split('.')[1].length : 0); });
  }
});

// ── Tabs ─────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab, .tab-content').forEach(el => el.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});

// ── Mode toggle ──────────────────────────────────────────────
document.getElementById('mode').addEventListener('change', function() {
  const isHidden = this.value === 'hidden';
  document.getElementById('hidden-section').style.display = isHidden ? 'block' : 'none';
});

// ── Depth file upload ────────────────────────────────────────
document.getElementById('depth-file').addEventListener('change', async function() {
  const file = this.files[0];
  if (!file) return;
  state.depthB64 = await fileToB64(file);
  const url = URL.createObjectURL(file);
  const thumb = document.getElementById('depth-thumb');
  document.getElementById('depth-thumb-img').src = url;
  thumb.style.display = 'block';
  toast('Depth map loaded', 'success');
});

// ── Preset select ────────────────────────────────────────────
document.getElementById('preset').addEventListener('change', async function() {
  if (this.value === 'none') return;
  const data = await get(`${API}/api/preset/${this.value}`);
  if (data.error) return;
  document.getElementById('depth-factor').value = data.stereo.depth_factor;
  document.getElementById('depth-factor-val').textContent = data.stereo.depth_factor.toFixed(2);
  document.getElementById('max-parallax').value = data.stereo.max_parallax_fraction;
  document.getElementById('max-parallax-val').textContent = data.stereo.max_parallax_fraction.toFixed(3);
  document.getElementById('safe-mode').checked = data.stereo.safe_mode;
  toast(`Preset "${data.display_name}" applied`, 'success');
});

// ── Generate ─────────────────────────────────────────────────
document.getElementById('btn-generate').addEventListener('click', async () => {
  if (!state.depthB64) { toast('Please upload a depth map first', 'error'); return; }

  const btn = document.getElementById('btn-generate');
  btn.disabled = true;
  btn.textContent = '⏳ Generating…';

  const progressWrap = document.getElementById('progress-bar-wrap');
  const progressBar  = document.getElementById('progress-bar');
  const progressMsg  = document.getElementById('progress-msg');
  const placeholder  = document.getElementById('result-placeholder');
  const resultImg    = document.getElementById('result-img');
  const resultMeta   = document.getElementById('result-meta');

  progressWrap.style.display = 'block';
  progressBar.style.width    = '5%';
  progressMsg.textContent    = 'Preparing…';
  placeholder.style.display  = 'none';
  resultImg.style.display    = 'none';
  resultMeta.style.display   = 'none';

  const body = {
    depth_b64:     state.depthB64,
    mode:          document.getElementById('mode').value,
    preset:        document.getElementById('preset').value,
    depth_factor:  parseFloat(document.getElementById('depth-factor').value),
    max_parallax:  parseFloat(document.getElementById('max-parallax').value),
    safe_mode:     document.getElementById('safe-mode').checked,
    invert_depth:  document.getElementById('invert-depth').checked,
    tile_size:     parseInt(document.getElementById('tile-size').value),
    pattern_scale: parseFloat(document.getElementById('pattern-scale').value),
    seed:          parseInt(document.getElementById('seed').value),
    hidden_shape:  document.getElementById('hidden-shape').value,
    hidden_text:   document.getElementById('hidden-text').value,
    fg_depth:      parseFloat(document.getElementById('fg-depth').value),
    async_job:     true,
  };

  if (state.selectedPattern) body.pattern_name = state.selectedPattern;

  const t0 = performance.now();

  try {
    const { job_id } = await post(`${API}/api/synthesize`, body);

    // Listen to SSE progress
    await new Promise((resolve, reject) => {
      const es = new EventSource(`${API}/api/progress/${job_id}`);
      es.onmessage = e => {
        const evt = JSON.parse(e.data);
        if (evt.heartbeat) return;
        progressBar.style.width = evt.progress + '%';
        if (evt.message) progressMsg.textContent = evt.message;
        if (evt.done) {
          es.close();
          if (evt.error) reject(new Error(evt.message));
          else resolve();
        }
      };
      es.onerror = () => { es.close(); reject(new Error('SSE connection lost')); };
    });

    // Fetch result
    const result = await get(`${API}/api/result/${job_id}`);
    if (result.error) throw new Error(result.error);

    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    state.resultB64 = result.image_b64;
    resultImg.src   = 'data:image/png;base64,' + result.image_b64;
    resultImg.style.display = 'block';
    progressWrap.style.display = 'none';
    progressMsg.textContent = '';

    document.getElementById('meta-size').textContent     = `${result.width}×${result.height}`;
    document.getElementById('meta-mode').textContent     = result.mode;
    document.getElementById('meta-depth').textContent    = result.depth_factor;
    document.getElementById('meta-parallax').textContent = result.max_parallax;
    document.getElementById('meta-preset').textContent   = result.preset || 'custom';
    document.getElementById('meta-time').textContent     = `${elapsed}s`;
    resultMeta.style.display = 'flex';

    document.getElementById('btn-download').disabled = false;
    toast('Stereogram generated!', 'success');

  } catch (err) {
    progressWrap.style.display = 'none';
    placeholder.style.display  = 'flex';
    toast('Error: ' + err.message, 'error');
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>⬡</span> Generate Stereogram';
  }
});

// ── Download ─────────────────────────────────────────────────
document.getElementById('btn-download').addEventListener('click', () => {
  if (!state.resultB64) return;
  const a = document.createElement('a');
  a.href     = 'data:image/png;base64,' + state.resultB64;
  a.download = `depthforge_${document.getElementById('mode').value}_${Date.now()}.png`;
  a.click();
});

// ── Analyze ──────────────────────────────────────────────────
document.getElementById('btn-analyze').addEventListener('click', async () => {
  const file = document.getElementById('analyze-file').files[0];
  if (!file) {
    // Fall back to the depth map already loaded
    if (!state.depthB64) { toast('Upload a depth map in the Generate tab or here', 'error'); return; }
  }
  const btn = document.getElementById('btn-analyze');
  btn.disabled = true; btn.textContent = 'Analyzing…';

  try {
    const b64 = file ? await fileToB64(file) : state.depthB64;
    const result = await post(`${API}/api/analyze`, {
      depth_b64:    b64,
      depth_factor: parseFloat(document.getElementById('depth-factor').value),
      max_parallax: parseFloat(document.getElementById('max-parallax').value),
    });
    if (result.error) throw new Error(result.error);

    const badge = document.getElementById('comfort-badge');
    badge.textContent = result.passed ? '✓ PASS' : '✗ FAIL';
    badge.className   = 'comfort-badge ' + (result.passed ? 'pass' : 'fail');

    document.getElementById('comfort-score').textContent  = result.overall_score.toFixed(2) + ' / 1.0';
    document.getElementById('strain-rating').textContent  = result.strain_rating;
    document.getElementById('stat-parallax').textContent  = result.max_parallax_px;
    document.getElementById('stat-vergence').textContent  = result.max_vergence_deg + '°';
    document.getElementById('stat-pct').textContent       = (result.pct_uncomfortable * 100).toFixed(1) + '%';
    document.getElementById('stat-score').textContent     = result.overall_score.toFixed(2);

    const adviceList = document.getElementById('advice-list');
    adviceList.innerHTML = result.advice.map(a => `<li>${a}</li>`).join('');

    document.getElementById('heatmap-img').src = 'data:image/png;base64,' + result.heatmap_b64;
    document.getElementById('overlay-img').src = 'data:image/png;base64,' + result.overlay_b64;
    document.getElementById('analyze-result').style.display = 'flex';
    toast('Analysis complete', 'success');
  } catch (err) {
    toast('Analysis error: ' + err.message, 'error');
  } finally {
    btn.disabled = false; btn.textContent = 'Run Analysis';
  }
});

// ── Load presets ─────────────────────────────────────────────
async function loadPresets() {
  const data = await get(`${API}/api/presets`);
  state.presets = data;

  // Populate sidebar select
  const sel = document.getElementById('preset');
  sel.innerHTML = '<option value="none">Custom</option>';
  data.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.name;
    opt.textContent = p.display_name || p.name;
    sel.appendChild(opt);
  });

  // Populate presets tab
  const cards = document.getElementById('preset-cards');
  cards.innerHTML = data.map(p => `
    <div class="preset-card" data-name="${p.name}" onclick="applyPreset('${p.name}')">
      <div class="name">${p.display_name || p.name}</div>
      <div class="desc">${p.description || ''}</div>
      <div class="tags">
        <span class="tag">df=${p.depth_factor}</span>
        <span class="tag">${p.output_dpi}dpi</span>
        ${p.safe_mode ? '<span class="tag">safe</span>' : ''}
        <span class="tag">${p.color_space}</span>
      </div>
    </div>
  `).join('');
}

function applyPreset(name) {
  document.getElementById('preset').value = name;
  document.getElementById('preset').dispatchEvent(new Event('change'));
  document.querySelectorAll('.preset-card').forEach(c => {
    c.classList.toggle('active', c.dataset.name === name);
  });
}

// ── Load patterns ─────────────────────────────────────────────
async function loadPatterns() {
  const data = await get(`${API}/api/patterns`);
  state.patterns = data.patterns;

  // Category filter
  const filter = document.getElementById('pattern-category-filter');
  data.categories.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c.charAt(0).toUpperCase() + c.slice(1);
    filter.appendChild(opt);
  });
  filter.addEventListener('change', () => renderAllPatterns(filter.value));

  // Sidebar mini-grid (first 16, small)
  const grid = document.getElementById('pattern-grid');
  const first16 = data.patterns.slice(0, 16);
  grid.innerHTML = first16.map(p => `
    <div class="pattern-thumb" data-name="${p.name}" title="${p.name}">
      <img src="" data-pattern="${p.name}" alt="${p.name}">
    </div>
  `).join('');

  // Lazy-load sidebar thumbnails
  grid.querySelectorAll('.pattern-thumb').forEach(thumb => {
    thumb.addEventListener('click', () => {
      document.querySelectorAll('.pattern-thumb').forEach(t => t.classList.remove('active'));
      thumb.classList.add('active');
      state.selectedPattern = thumb.dataset.name;
    });
    const img = thumb.querySelector('img');
    get(`${API}/api/pattern/${img.dataset.pattern}?size=56`)
      .then(d => { if (d.image) img.src = 'data:image/png;base64,' + d.image; });
  });

  // Select first by default
  if (first16.length) {
    state.selectedPattern = first16[0].name;
    grid.querySelector('.pattern-thumb').classList.add('active');
  }

  // Full patterns tab
  renderAllPatterns('');

  // Preview size slider
  document.getElementById('pattern-preview-size').addEventListener('input', function() {
    document.getElementById('preview-size-lbl').textContent = this.value + 'px';
    renderAllPatterns(document.getElementById('pattern-category-filter').value);
  });
}

function renderAllPatterns(category) {
  const size  = parseInt(document.getElementById('pattern-preview-size').value) || 96;
  const items = category ? state.patterns.filter(p => p.category === category) : state.patterns;
  const grid  = document.getElementById('all-patterns-grid');

  grid.innerHTML = items.map(p => `
    <div style="cursor:pointer;" onclick="selectPattern('${p.name}')">
      <div style="
        border-radius:var(--radius);
        overflow:hidden;
        border:2px solid ${state.selectedPattern === p.name ? 'var(--accent)' : 'var(--border)'};
        aspect-ratio:1;
        background:var(--surface2);
      " id="pgrid-${p.name}">
        <img data-pattern="${p.name}" data-size="${size}" alt="${p.name}"
          style="width:100%;height:100%;object-fit:cover;display:block;"
          src="">
      </div>
      <div style="font-size:11px;color:var(--muted);margin-top:4px;text-align:center;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
        title="${p.name}">${p.name}</div>
      <div style="font-size:10px;color:var(--accent2);text-align:center;">${p.category}</div>
    </div>
  `).join('');

  // Load images
  grid.querySelectorAll('img[data-pattern]').forEach(img => {
    get(`${API}/api/pattern/${img.dataset.pattern}?size=${img.dataset.size}`)
      .then(d => { if (d.image) img.src = 'data:image/png;base64,' + d.image; });
  });
}

function selectPattern(name) {
  state.selectedPattern = name;
  // Update sidebar
  document.querySelectorAll('.pattern-thumb').forEach(t => {
    t.classList.toggle('active', t.dataset.name === name);
  });
  // Update all-patterns grid borders
  document.querySelectorAll('[id^="pgrid-"]').forEach(el => {
    el.style.borderColor = el.id === `pgrid-${name}` ? 'var(--accent)' : 'var(--border)';
  });
  toast(`Pattern: ${name}`, 'info');
}

// ── Status ────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const data = await get(`${API}/api/status`);
    document.getElementById('status-dot').classList.remove('offline');
    document.getElementById('status-label').textContent = 'Online';
  } catch {
    document.getElementById('status-dot').classList.add('offline');
    document.getElementById('status-label').textContent = 'Offline';
  }
}

// ── Init ──────────────────────────────────────────────────────
(async () => {
  await checkStatus();
  await Promise.all([loadPresets(), loadPatterns()]);
})();
</script>
</body>
</html>
"""
