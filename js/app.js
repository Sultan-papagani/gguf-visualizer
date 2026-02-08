/**
 * Main Application
 * Wires file upload, GGUF parsing, point cloud generation, and rendering.
 */

import { parseGGUFHeader, extractArchInfo, computeTotalParams, GGMLTypeName } from './gguf-parser.js';
import { generatePointCloud, generateConnections, computeLayerBounds } from './point-cloud.js';
import { ModelRenderer } from './renderer.js';

// ─── DOM Elements ───────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const progressContainer = document.getElementById('progress-container');
const progressBarInner = document.getElementById('progress-bar-inner');
const progressText = document.getElementById('progress-text');
const metadataPanel = document.getElementById('metadata-panel');
const metaArch = document.getElementById('meta-arch');
const modelNameEl = document.getElementById('model-name');
const quickStatsEl = document.getElementById('quick-stats');
const tensorListEl = document.getElementById('tensor-list');
const tensorCountLabel = document.getElementById('tensor-count-label');
const pointSlider = document.getElementById('point-slider');
const pointCountDisplay = document.getElementById('point-count-display');
const sizeSlider = document.getElementById('size-slider');
const sizeDisplay = document.getElementById('size-display');
const colorMode = document.getElementById('color-mode');
const welcomeOverlay = document.getElementById('welcome-overlay');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const canvasContainer = document.getElementById('canvas-container');
const tooltip = document.getElementById('tooltip');
const controlsHud = document.getElementById('controls-hud');
const flyPrompt = document.getElementById('fly-prompt');
const connectionsToggle = document.getElementById('connections-toggle');
const connDensityRow = document.getElementById('conn-density-row');
const connDensitySlider = document.getElementById('conn-density-slider');
const connDensityDisplay = document.getElementById('conn-density-display');
const layerBoxMode = document.getElementById('layer-box-mode');
const legend = document.getElementById('legend');
const legendWeight = document.getElementById('legend-weight');
const legendTensor = document.getElementById('legend-tensor');
const legendLayer = document.getElementById('legend-layer');

// ─── State ──────────────────────────────────────────────────────────
let renderer = null;
let currentFile = null;
let parsedData = null; // { metadata, tensors, tensorDataOffset, archInfo }
let isGenerating = false;
let lastPointCloudData = null; // { positions, tensorRegions } for connections

// ─── Initialize Renderer ───────────────────────────────────────────
function initRenderer() {
  if (!renderer) {
    renderer = new ModelRenderer(canvasContainer);
  }
}

// ─── File Upload Handling ───────────────────────────────────────────

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

// Also support drag-drop on the entire canvas area
canvasContainer.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

canvasContainer.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length > 0) handleFile(files[0]);
});

// ─── Worker-based header parsing ────────────────────────────────────

function parseHeaderInWorker(file, onProgress) {
  return new Promise((resolve, reject) => {
    let worker;
    try {
      worker = new Worker(new URL('./parse-worker.js', import.meta.url), { type: 'module' });
    } catch (_) {
      // Worker creation failed (e.g. file:// protocol) — reject to trigger fallback
      return reject(new Error('Worker unavailable'));
    }

    const timeout = setTimeout(() => {
      worker.terminate();
      reject(new Error('Worker timeout'));
    }, 180_000); // 3 min timeout

    worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === 'progress') {
        if (onProgress) onProgress(msg.phase, msg.current, msg.total);
      } else if (msg.type === 'result') {
        clearTimeout(timeout);
        worker.terminate();
        resolve(msg);
      } else if (msg.type === 'error') {
        clearTimeout(timeout);
        worker.terminate();
        reject(new Error(msg.message));
      }
    };

    worker.onerror = (e) => {
      clearTimeout(timeout);
      worker.terminate();
      reject(new Error(e.message || 'Worker error'));
    };

    worker.postMessage({ file });
  });
}

// ─── Main Pipeline ──────────────────────────────────────────────────

async function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.gguf')) {
    alert('Please select a .gguf file');
    return;
  }

  currentFile = file;
  initRenderer();

  // Show progress
  showProgress(true);
  setProgress(0, 'Reading GGUF header...');
  welcomeOverlay.classList.add('hidden');

  try {
    // Phase 1: Parse header (prefer worker, fallback to main thread)
    console.log(`[GGUF] Loading file: ${file.name} (${(file.size / 1e9).toFixed(2)} GB)`);

    const progressCb = (phase, current, total) => {
      if (phase === 'metadata') {
        setProgress(10 + (current / total) * 20, `Parsing metadata ${current}/${total}...`);
      } else if (phase === 'tensors') {
        setProgress(30 + (current / total) * 20, `Reading tensor info ${current}/${total}...`);
      }
    };

    let metadata, tensors, tensorDataOffset, version, archInfo, totalParams;

    try {
      // Try worker (keeps UI responsive during heavy parsing)
      const result = await parseHeaderInWorker(file, progressCb);
      ({ metadata, tensors, tensorDataOffset, version, archInfo, totalParams } = result);
      console.log('[GGUF] Parsed in background worker');
    } catch (workerErr) {
      // Fallback to main thread
      console.warn('[GGUF] Worker unavailable, parsing on main thread:', workerErr.message);
      const result = await parseGGUFHeader(file, progressCb);
      ({ metadata, tensors, tensorDataOffset, version } = result);
      archInfo = extractArchInfo(metadata);
      totalParams = computeTotalParams(tensors);
    }

    console.log(`[GGUF] Parsed: v${version}, ${tensors.length} tensors, data offset=${tensorDataOffset}`);

    parsedData = { metadata, tensors, tensorDataOffset, archInfo, totalParams };

    setProgress(50, 'Populating metadata...');

    // Update UI with metadata
    updateMetadataPanel(archInfo, totalParams, tensors, version, file);

    // Phase 2: Generate point cloud
    await regeneratePointCloud();

  } catch (err) {
    console.error('[GGUF] Error processing file:', err);
    alert(`Error processing file: ${err.message}`);
    showProgress(false);
    showLoading(false);
  }
}

async function regeneratePointCloud() {
  if (!parsedData || !currentFile || isGenerating) return;
  isGenerating = true;

  const targetPoints = parseInt(pointSlider.value);
  const currentColorMode = colorMode.value;

  showLoading(true);
  loadingText.textContent = 'Generating point cloud...';

  try {
    const { positions, colors, tensorRegions, actualPointCount } = await generatePointCloud(
      currentFile,
      parsedData.archInfo,
      parsedData.tensors,
      parsedData.tensorDataOffset,
      targetPoints,
      currentColorMode,
      (phase, current, total, name) => {
        const pct = Math.floor((current / total) * 100);
        loadingText.textContent = `Sampling tensors... ${pct}% ${name ? `(${name})` : ''}`;
      }
    );

    // Trim buffers to actual point count
    const pos = positions.subarray(0, actualPointCount * 3);
    const col = colors.subarray(0, actualPointCount * 3);

    renderer.setPointCloud(pos, col, tensorRegions);

    // Store for connection regeneration
    lastPointCloudData = { positions: pos, tensorRegions };

    // Generate connections if enabled
    if (connectionsToggle.checked) {
      rebuildConnections();
    }

    // Generate layer bounding boxes
    rebuildLayerBoxes();

    // Update quick stats with actual count
    updateQuickStats(parsedData.archInfo, parsedData.totalParams, actualPointCount);

    showLoading(false);
    showProgress(false);
    showFlyPrompt();

  } catch (err) {
    console.error('Error generating point cloud:', err);
    loadingText.textContent = `Error: ${err.message}`;
    setTimeout(() => showLoading(false), 2000);
  }

  isGenerating = false;
}

// ─── UI Updates ─────────────────────────────────────────────────────

function showProgress(visible) {
  progressContainer.classList.toggle('active', visible);
}

function setProgress(pct, text) {
  progressBarInner.style.width = `${Math.min(100, pct)}%`;
  if (text) progressText.textContent = text;
}

function showLoading(visible) {
  loadingOverlay.classList.toggle('active', visible);
}

function formatNumber(n) {
  if (n >= 1e12) return (n / 1e12).toFixed(1) + 'T';
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function formatBytes(bytes) {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB';
  return bytes + ' B';
}

function updateMetadataPanel(archInfo, totalParams, tensors, version, file) {
  metadataPanel.classList.add('active');
  legend.classList.add('active');

  // Model name in top bar
  modelNameEl.textContent = archInfo.name || file.name;

  // Architecture metadata
  const fields = [
    ['Architecture', archInfo.architecture.toUpperCase()],
    ['Quantization', archInfo.fileTypeName],
    ['Parameters', formatNumber(totalParams)],
    ['File Size', formatBytes(file.size)],
    ['Layers', archInfo.blockCount || '?'],
    ['Context', formatNumber(archInfo.contextLength) || '?'],
    ['Embedding', formatNumber(archInfo.embeddingLength) || '?'],
    ['FFN Size', formatNumber(archInfo.feedForwardLength) || '?'],
    ['Attn Heads', archInfo.headCount || '?'],
    ['KV Heads', archInfo.headCountKV || '?'],
    ['Head Dim', archInfo.headDim || '?'],
    ['Vocab Size', formatNumber(archInfo.vocabSize) || '?'],
  ];

  if (archInfo.isMoE) {
    fields.push(['Experts', archInfo.expertCount]);
    fields.push(['Active Experts', archInfo.expertUsedCount]);
  }

  if (archInfo.ropeFreqBase) {
    fields.push(['RoPE Base', formatNumber(archInfo.ropeFreqBase)]);
  }

  fields.push(['GGUF Version', `v${version}`]);
  fields.push(['Model Type', archInfo.isMoE ? 'MoE' : (archInfo.isGQA ? 'Dense (GQA)' : 'Dense')]);

  metaArch.innerHTML = fields.map(([label, value]) =>
    `<div class="meta-item"><span class="meta-label">${label}</span><span class="meta-value">${value}</span></div>`
  ).join('');

  // Tensor list
  tensorCountLabel.textContent = `(${tensors.length})`;
  tensorListEl.innerHTML = tensors.slice(0, 200).map(t => {
    const dims = t.dims.join(' x ');
    const typeName = GGMLTypeName[t.type] || `?${t.type}`;
    return `<div class="tensor-entry"><span class="t-name" title="${t.name}">${t.name}</span><span class="t-shape">${dims} [${typeName}]</span></div>`;
  }).join('');

  if (tensors.length > 200) {
    tensorListEl.innerHTML += `<div class="tensor-entry" style="color:var(--text-muted); text-align:center;">...and ${tensors.length - 200} more</div>`;
  }
}

function updateQuickStats(archInfo, totalParams, pointCount) {
  quickStatsEl.innerHTML = [
    `<span><span class="stat-val">${formatNumber(totalParams)}</span> params</span>`,
    `<span><span class="stat-val">${archInfo.blockCount}</span> layers</span>`,
    `<span><span class="stat-val">${archInfo.headCount}</span> heads</span>`,
    archInfo.isMoE ? `<span><span class="stat-val">${archInfo.expertCount}</span> experts</span>` : '',
    `<span><span class="stat-val">${formatNumber(pointCount)}</span> points</span>`,
    `<span><span class="stat-val">${archInfo.fileTypeName}</span></span>`,
  ].filter(Boolean).join('');
}

// ─── Controls ───────────────────────────────────────────────────────

// Point count slider
pointSlider.addEventListener('input', () => {
  const val = parseInt(pointSlider.value);
  pointCountDisplay.textContent = (val / 1e6).toFixed(1) + 'M';
});

let regenerateTimeout = null;
pointSlider.addEventListener('change', () => {
  clearTimeout(regenerateTimeout);
  regenerateTimeout = setTimeout(() => regeneratePointCloud(), 100);
});

// Point size slider
sizeSlider.addEventListener('input', () => {
  const val = parseFloat(sizeSlider.value);
  sizeDisplay.textContent = val.toFixed(1);
  if (renderer) renderer.setPointSize(val);
});

// Color mode selector
colorMode.addEventListener('change', () => {
  // Show correct legend
  legendWeight.style.display = colorMode.value === 'weight' ? 'block' : 'none';
  legendTensor.style.display = colorMode.value === 'tensor' ? 'block' : 'none';
  legendLayer.style.display = colorMode.value === 'layer' ? 'block' : 'none';

  regeneratePointCloud();
});

// ─── Connection toggle ──────────────────────────────────────────────

function rebuildConnections() {
  if (!lastPointCloudData || !renderer) return;
  const density = parseFloat(connDensitySlider.value);
  const { positions: linePos, colors: lineCol, lineCount } =
    generateConnections(lastPointCloudData.tensorRegions, lastPointCloudData.positions, density);
  renderer.setConnections(linePos, lineCol);
  console.log(`[Connections] ${lineCount} neural links generated (density=${density})`);
}

connectionsToggle.addEventListener('change', () => {
  const enabled = connectionsToggle.checked;
  connDensityRow.style.display = enabled ? 'flex' : 'none';
  if (enabled) {
    rebuildConnections();
  } else {
    if (renderer) renderer.toggleConnections(false);
  }
});

connDensitySlider.addEventListener('input', () => {
  connDensityDisplay.textContent = parseFloat(connDensitySlider.value).toFixed(1);
});

let connDensityTimeout = null;
connDensitySlider.addEventListener('change', () => {
  clearTimeout(connDensityTimeout);
  connDensityTimeout = setTimeout(() => {
    if (connectionsToggle.checked) rebuildConnections();
  }, 100);
});

// ─── Layer bounding boxes ────────────────────────────────────────────

function rebuildLayerBoxes() {
  if (!lastPointCloudData || !renderer || !parsedData) return;
  const bounds = computeLayerBounds(lastPointCloudData.tensorRegions);
  renderer.setLayerBoxes(bounds, parsedData.archInfo.blockCount || 1);
  renderer.setLayerBoxMode(layerBoxMode.value);
}

layerBoxMode.addEventListener('change', () => {
  if (renderer) renderer.setLayerBoxMode(layerBoxMode.value);
});

// Sidebar toggle
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
  // Trigger resize for renderer
  setTimeout(() => {
    if (renderer) renderer._onResize();
  }, 350);
});

// ─── Tooltip on hover ───────────────────────────────────────────────

let tooltipTimeout = null;

canvasContainer.addEventListener('mousemove', (e) => {
  clearTimeout(tooltipTimeout);
  tooltipTimeout = setTimeout(() => {
    if (!renderer) return;
    const region = renderer.getTensorAtScreen(e.clientX, e.clientY);
    if (region) {
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top = (e.clientY + 14) + 'px';
      tooltip.querySelector('.tt-known').textContent = region.knownName || region.category;
      tooltip.querySelector('.tt-name').textContent = region.name;
      const dims = region.dims.join(' × ');
      const typeName = GGMLTypeName[region.type] || '?';
      tooltip.querySelector('.tt-detail').textContent = `${dims} | ${typeName}`;
      // Highlight layer box on hover
      if (renderer) renderer.highlightLayer(region.layerIdx);
    } else {
      tooltip.style.display = 'none';
      if (renderer) renderer.highlightLayer(-1);
    }
  }, 50);
});

canvasContainer.addEventListener('mouseleave', () => {
  tooltip.style.display = 'none';
});

// ─── Pointer lock HUD ───────────────────────────────────────────────

document.addEventListener('pointerlockchange', () => {
  const locked = !!document.pointerLockElement;
  controlsHud.classList.toggle('visible', locked);
  flyPrompt.classList.toggle('visible', !locked && !!parsedData);
  tooltip.style.display = 'none';
});

// Show fly prompt once a model is loaded
function showFlyPrompt() {
  flyPrompt.classList.add('visible');
  setTimeout(() => flyPrompt.classList.remove('visible'), 6000);
}

// ─── Keyboard shortcuts ─────────────────────────────────────────────

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'Tab') {
    e.preventDefault();
    sidebar.classList.toggle('collapsed');
    setTimeout(() => {
      if (renderer) renderer._onResize();
    }, 350);
  }
});

// ─── Init ───────────────────────────────────────────────────────────
initRenderer();

