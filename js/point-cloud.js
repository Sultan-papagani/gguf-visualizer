/**
 * Point Cloud Generator
 * Computes 3D positions, colors, and metadata for each point based on
 * model architecture. The spatial layout encodes the model structure so
 * different models produce visually distinct shapes.
 */

import { classifyTensor, sampleTensorWeights } from './gguf-parser.js';

// ─── Color palettes ─────────────────────────────────────────────────

const TENSOR_COLORS = {
  attn_q:     [0.27, 0.53, 1.00],  // #4488ff
  attn_k:     [0.27, 0.67, 1.00],  // #44aaff
  attn_v:     [0.27, 0.80, 1.00],  // #44ccff
  attn_out:   [0.40, 0.53, 0.87],  // #6688dd
  attn_norm:  [0.87, 0.87, 0.27],  // #dddd44
  attn_other: [0.33, 0.60, 0.87],  // #5599dd

  ffn_gate:   [1.00, 0.53, 0.27],  // #ff8844
  ffn_up:     [1.00, 0.67, 0.27],  // #ffaa44
  ffn_down:   [1.00, 0.40, 0.27],  // #ff6644
  ffn_norm:   [0.87, 0.87, 0.27],  // #dddd44
  ffn_other:  [1.00, 0.60, 0.33],  // #ff9955

  moe_gate:   [0.00, 0.90, 0.85],  // #00e5d9 — teal (router)
  moe_up:     [0.85, 0.25, 0.95],  // #d940f2 — magenta
  moe_down:   [0.95, 0.75, 0.10],  // #f2bf1a — golden

  embedding:  [0.27, 0.87, 0.53],  // #44dd88
  output:     [0.67, 0.40, 1.00],  // #aa66ff
  output_norm:[0.87, 0.87, 0.27],  // #dddd44
  norm:       [0.87, 0.87, 0.27],  // #dddd44
  other:      [0.53, 0.53, 0.60],  // #888899
};

// ─── Human-readable tensor names ─────────────────────────────────────

const KNOWN_NAMES = {
  embedding:   'Token Embedding',
  output:      'Output Projection (LM Head)',
  output_norm: 'Final Layer Norm',
  attn_q:      'Attention Query',
  attn_k:      'Attention Key',
  attn_v:      'Attention Value',
  attn_out:    'Attention Output',
  attn_norm:   'Pre-Attention Norm',
  attn_other:  'Attention (misc)',
  ffn_gate:    'FFN Gate Projection',
  ffn_up:      'FFN Up Projection',
  ffn_down:    'FFN Down Projection',
  ffn_norm:    'Pre-FFN Norm',
  ffn_other:   'FFN (misc)',
  moe_gate:    'MoE Router / Gate',
  moe_up:      'MoE Expert Up',
  moe_down:    'MoE Expert Down',
  norm:        'Layer Norm',
  other:       'Tensor',
};

/**
 * Return a human-readable "known name" for a tensor based on its role.
 */
export function getKnownName(category, layerIdx, expertIdx) {
  let base = KNOWN_NAMES[category] || category;
  const parts = [];
  if (layerIdx >= 0) parts.push(`Layer ${layerIdx}`);
  if (expertIdx >= 0) parts.push(`Expert ${expertIdx}`);
  if (parts.length > 0) return `${base} — ${parts.join(', ')}`;
  return base;
}

/**
 * Diverging colormap: blue -> white -> red
 */
function weightToColor(value, maxAbsVal) {
  const t = maxAbsVal > 0 ? value / maxAbsVal : 0; // [-1, 1]
  const clamped = Math.max(-1, Math.min(1, t));

  if (clamped < 0) {
    // Blue to white
    const s = 1 + clamped; // 0 to 1 (0 = pure blue, 1 = white)
    return [0.2 + 0.8 * s, 0.33 + 0.67 * s, 1.0];
  } else {
    // White to red
    const s = clamped; // 0 to 1 (0 = white, 1 = pure red)
    return [1.0, 1.0 - 0.67 * s, 1.0 - 0.8 * s];
  }
}

/**
 * Layer depth colormap: green -> blue -> purple
 */
function layerToColor(layerIdx, totalLayers) {
  const t = totalLayers > 1 ? layerIdx / (totalLayers - 1) : 0;
  if (t < 0.5) {
    const s = t * 2;
    return [0.13 * (1 - s) + 0.27 * s, 0.8 * (1 - s) + 0.53 * s, 0.53 * (1 - s) + 1.0 * s];
  } else {
    const s = (t - 0.5) * 2;
    return [0.27 * (1 - s) + 0.67 * s, 0.53 * (1 - s) + 0.27 * s, 1.0];
  }
}

/**
 * Generate the full point cloud from parsed GGUF data.
 *
 * @param {File} file - The .gguf file
 * @param {Object} archInfo - Architecture info from extractArchInfo()
 * @param {Array} tensors - Array of tensor info objects
 * @param {number} tensorDataOffset - Byte offset where tensor data starts
 * @param {number} targetPointCount - Desired number of points (1M-6M)
 * @param {string} colorMode - 'weight', 'tensor', or 'layer'
 * @param {Function} onProgress - Progress callback
 * @returns {{ positions: Float32Array, colors: Float32Array, tensorRegions: Array }}
 */
// Minimum number of points any single tensor will be rendered with,
// even if proportional allocation would give it fewer. Prevents tiny
// tensors (norms, biases, etc.) from being invisible single dots.
const MIN_POINTS_PER_TENSOR = 400;

export async function generatePointCloud(file, archInfo, tensors, tensorDataOffset, targetPointCount, colorMode, onProgress) {
  // Compute total params and allocate points proportionally
  const totalParams = tensors.reduce((s, t) => s + t.numElements, 0);
  const decimationRatio = totalParams / targetPointCount;

  // Assign point counts to each tensor proportionally, with a floor
  const tensorAllocs = tensors.map(t => {
    const cls = classifyTensor(t.name);
    const rawCount = Math.max(MIN_POINTS_PER_TENSOR, Math.floor(t.numElements / decimationRatio));
    return { tensor: t, cls, pointCount: rawCount };
  });

  // Adjust total to match target exactly
  let allocTotal = tensorAllocs.reduce((s, a) => s + a.pointCount, 0);
  // Scale if needed
  if (allocTotal > 0) {
    const scale = targetPointCount / allocTotal;
    tensorAllocs.forEach(a => {
      a.pointCount = Math.max(MIN_POINTS_PER_TENSOR, Math.round(a.pointCount * scale));
    });
  }

  const actualTotal = tensorAllocs.reduce((s, a) => s + a.pointCount, 0);

  // Prepare output buffers
  const positions = new Float32Array(actualTotal * 3);
  const colors = new Float32Array(actualTotal * 3);
  const tensorRegions = [];

  // ─── Layout computation ───────────────────────────────────────────
  const layout = computeLayout(archInfo, tensorAllocs);

  const totalTensors = tensorAllocs.length;

  // ─── Phase 1: Parallel batched weight sampling (I/O heavy) ──────
  const allWeights = new Array(totalTensors).fill(null);

  if (colorMode === 'weight') {
    const BATCH = 8; // Read 8 tensors in parallel
    for (let bi = 0; bi < totalTensors; bi += BATCH) {
      const end = Math.min(bi + BATCH, totalTensors);
      const promises = [];
      for (let ti = bi; ti < end; ti++) {
        const { tensor, pointCount } = tensorAllocs[ti];
        promises.push(
          sampleTensorWeights(file, tensorDataOffset, tensor, pointCount)
            .catch(() => null)
        );
      }
      const batchResults = await Promise.all(promises);
      for (let j = 0; j < batchResults.length; j++) {
        allWeights[bi + j] = batchResults[j];
      }
      if (onProgress) {
        onProgress('sampling', Math.min(end, totalTensors), totalTensors,
          tensorAllocs[bi].tensor.name);
      }
    }
  }

  // ─── Phase 2: Compute positions & colors (CPU only, fast) ──────
  let globalIdx = 0;

  for (let ti = 0; ti < totalTensors; ti++) {
    const { tensor, cls, pointCount } = tensorAllocs[ti];
    const region = layout.getRegion(cls.category, cls.layerIdx, cls.expertIdx, tensor);

    if (onProgress && colorMode !== 'weight') {
      onProgress('sampling', ti, totalTensors, tensor.name);
    }

    const weightValues = allWeights[ti];

    // Find max abs value for normalization
    let maxAbs = 0;
    if (weightValues) {
      for (let i = 0; i < weightValues.length; i++) {
        const a = Math.abs(weightValues[i]);
        if (a > maxAbs && isFinite(a)) maxAbs = a;
      }
      if (maxAbs === 0) maxAbs = 1;
    }

    // Determine tensor dimensions for point placement
    const rows = tensor.dims.length >= 2 ? tensor.dims[1] : 1;
    const cols = tensor.dims[0] || 1;

    const startIdx = globalIdx;

    for (let pi = 0; pi < pointCount; pi++) {
      // Map point index to row/col within the tensor
      const paramIdx = Math.floor(pi * (tensor.numElements / pointCount));
      const row = rows > 1 ? Math.floor(paramIdx / cols) : 0;
      const col = paramIdx % cols;

      const rowT = rows > 1 ? row / (rows - 1) : 0.5;
      const colT = cols > 1 ? col / (cols - 1) : 0.5;

      // Add jitter for visual density (avoid perfect grid)
      const jx = (Math.random() - 0.5) * 0.3;
      const jy = (Math.random() - 0.5) * 0.3;
      const jz = (Math.random() - 0.5) * 0.15;

      // Compute 3D position
      const x = region.x + (colT + jx * (1 / Math.max(cols, 1))) * region.width;
      const y = region.y + (rowT + jy * (1 / Math.max(rows, 1))) * region.height;
      const z = region.z + jz * region.depth;

      positions[globalIdx * 3]     = x;
      positions[globalIdx * 3 + 1] = y;
      positions[globalIdx * 3 + 2] = z;

      // Color
      let r, g, b;
      if (colorMode === 'weight' && weightValues && pi < weightValues.length) {
        [r, g, b] = weightToColor(weightValues[pi], maxAbs);
      } else if (colorMode === 'layer') {
        if (cls.layerIdx >= 0) {
          [r, g, b] = layerToColor(cls.layerIdx, archInfo.blockCount || 1);
        } else {
          [r, g, b] = [0.5, 0.5, 0.6];
        }
      } else {
        // Tensor type coloring
        const tc = TENSOR_COLORS[cls.category] || TENSOR_COLORS.other;
        r = tc[0]; g = tc[1]; b = tc[2];
        // Add slight brightness variation
        const variation = 0.85 + Math.random() * 0.3;
        r *= variation; g *= variation; b *= variation;
      }

      colors[globalIdx * 3]     = Math.min(1, r);
      colors[globalIdx * 3 + 1] = Math.min(1, g);
      colors[globalIdx * 3 + 2] = Math.min(1, b);

      globalIdx++;
    }

    tensorRegions.push({
      name: tensor.name,
      knownName: getKnownName(cls.category, cls.layerIdx, cls.expertIdx),
      category: cls.category,
      layerIdx: cls.layerIdx,
      expertIdx: cls.expertIdx,
      dims: tensor.dims,
      type: tensor.type,
      region: region,
      startIdx: startIdx,
      endIdx: globalIdx,
    });
  }

  return { positions, colors, tensorRegions, actualPointCount: globalIdx };
}

// ─── Neural Connection Generator ────────────────────────────────────

/**
 * Generate line segments that connect tensor regions like neural pathways.
 * Lines flow from source tensors to target tensors following the model's
 * data-flow architecture: embedding → attention → FFN → next layer → output.
 *
 * @param {Array} tensorRegions - Region metadata from generatePointCloud
 * @param {Float32Array} positions - Point positions (xyz interleaved)
 * @param {number} density - Multiplier for number of lines (0.1 – 3.0, default 1.0)
 * @returns {{ positions: Float32Array, colors: Float32Array, lineCount: number }}
 */
export function generateConnections(tensorRegions, positions, density = 1.0) {
  // ── Full transformer data-flow rules ──
  // matchExpert: only connect regions sharing the same expertIdx
  const INTRA_LAYER_RULES = [
    // Pre-attention norm → Q, K, V
    { from: ['attn_norm'], to: ['attn_q', 'attn_k', 'attn_v'] },
    // Q, K, V → Attention Output
    { from: ['attn_q'],    to: ['attn_out'] },
    { from: ['attn_k'],    to: ['attn_out'] },
    { from: ['attn_v'],    to: ['attn_out'] },
    // Attention Output → Pre-FFN Norm
    { from: ['attn_out'],  to: ['ffn_norm'] },
    // Pre-FFN Norm → gate/up (dense) or MoE router
    { from: ['ffn_norm'],  to: ['ffn_gate', 'ffn_up', 'moe_gate'] },
    // MoE: router → individual expert gates/ups and packed experts
    { from: ['moe_gate'],  to: ['ffn_gate', 'ffn_up', 'moe_up'] },
    // FFN gate/up → down (expert-aware for MoE individual experts)
    { from: ['ffn_gate'],  to: ['ffn_down'], matchExpert: true },
    { from: ['ffn_up'],    to: ['ffn_down'], matchExpert: true },
    // Packed MoE: up → down
    { from: ['moe_up'],    to: ['moe_down'] },
  ];

  // Group regions by layer
  const layerMap = new Map();
  const globals = [];

  for (const region of tensorRegions) {
    if (region.layerIdx >= 0) {
      if (!layerMap.has(region.layerIdx)) layerMap.set(region.layerIdx, []);
      layerMap.get(region.layerIdx).push(region);
    } else {
      globals.push(region);
    }
  }

  const linePositions = [];
  const lineColors = [];

  const layers = Array.from(layerMap.keys()).sort((a, b) => a - b);

  // ── Intra-layer connections ──
  for (const layerIdx of layers) {
    const regions = layerMap.get(layerIdx);
    for (const rule of INTRA_LAYER_RULES) {
      const sources = regions.filter(r => rule.from.includes(r.category));
      const targets = regions.filter(r => rule.to.includes(r.category));

      for (const src of sources) {
        // Expert-aware: only connect same expert (or both non-expert)
        let validTargets = targets;
        if (rule.matchExpert) {
          validTargets = targets.filter(tgt =>
            src.expertIdx === tgt.expertIdx ||
            (src.expertIdx < 0 && tgt.expertIdx < 0)
          );
        }

        for (const tgt of validTargets) {
          _sampleLines(src, tgt, positions, linePositions, lineColors, density);
        }
      }
    }
  }

  // ── Cross-layer: FFN down → next layer's attn_norm (or Q,K,V) ──
  for (let i = 0; i < layers.length - 1; i++) {
    const currentRegions = layerMap.get(layers[i]);
    const nextRegions = layerMap.get(layers[i + 1]);

    const downs = currentRegions.filter(r =>
      r.category === 'ffn_down' || r.category === 'moe_down'
    );
    const nextNorms = nextRegions.filter(r => r.category === 'attn_norm');
    const nextTargets = nextNorms.length > 0
      ? nextNorms
      : nextRegions.filter(r => ['attn_q', 'attn_k', 'attn_v'].includes(r.category));

    for (const src of downs) {
      for (const tgt of nextTargets) {
        _sampleLines(src, tgt, positions, linePositions, lineColors, density * 0.4, 0.35);
      }
    }
  }

  // ── Embedding → first layer's attn_norm (or Q,K,V) ──
  if (layers.length > 0) {
    const embeddings = globals.filter(r => r.category === 'embedding');
    const firstRegions = layerMap.get(layers[0]) || [];
    const firstNorms = firstRegions.filter(r => r.category === 'attn_norm');
    const firstTargets = firstNorms.length > 0
      ? firstNorms
      : firstRegions.filter(r => ['attn_q', 'attn_k', 'attn_v'].includes(r.category));

    for (const src of embeddings) {
      for (const tgt of firstTargets) {
        _sampleLines(src, tgt, positions, linePositions, lineColors, density * 0.5, 0.4);
      }
    }
  }

  // ── Last layer → output_norm → output ──
  if (layers.length > 0) {
    const lastRegions = layerMap.get(layers[layers.length - 1]) || [];
    const lastDowns = lastRegions.filter(r =>
      r.category === 'ffn_down' || r.category === 'moe_down'
    );
    const outputNorms = globals.filter(r => r.category === 'output_norm');
    const outputs = globals.filter(r => r.category === 'output');

    // down → output_norm (or directly to output)
    const endTargets = outputNorms.length > 0 ? outputNorms : outputs;
    for (const src of lastDowns) {
      for (const tgt of endTargets) {
        _sampleLines(src, tgt, positions, linePositions, lineColors, density * 0.5, 0.4);
      }
    }

    // output_norm → output
    if (outputNorms.length > 0) {
      for (const src of outputNorms) {
        for (const tgt of outputs) {
          _sampleLines(src, tgt, positions, linePositions, lineColors, density * 0.5, 0.4);
        }
      }
    }
  }

  const totalLines = linePositions.length / 6;

  return {
    positions: new Float32Array(linePositions),
    colors: new Float32Array(lineColors),
    lineCount: totalLines,
  };
}

/**
 * Sample random line segments between two tensor regions.
 */
function _sampleLines(srcRegion, tgtRegion, positions, outPos, outCol, density = 1.0, dimFactor = 0.5) {
  const srcCount = srcRegion.endIdx - srcRegion.startIdx;
  const tgtCount = tgtRegion.endIdx - tgtRegion.startIdx;
  if (srcCount === 0 || tgtCount === 0) return;

  // Number of lines scales with region size, clamped and modulated by density
  const baseLine = Math.min(100, Math.max(8, Math.floor(Math.sqrt(Math.min(srcCount, tgtCount)))));
  const lineCount = Math.max(2, Math.round(baseLine * density));

  const srcColor = TENSOR_COLORS[srcRegion.category] || TENSOR_COLORS.other;
  const tgtColor = TENSOR_COLORS[tgtRegion.category] || TENSOR_COLORS.other;

  for (let i = 0; i < lineCount; i++) {
    const sIdx = srcRegion.startIdx + Math.floor(Math.random() * srcCount);
    const tIdx = tgtRegion.startIdx + Math.floor(Math.random() * tgtCount);

    // Source vertex
    outPos.push(
      positions[sIdx * 3],
      positions[sIdx * 3 + 1],
      positions[sIdx * 3 + 2]
    );
    outCol.push(
      srcColor[0] * dimFactor,
      srcColor[1] * dimFactor,
      srcColor[2] * dimFactor
    );

    // Target vertex
    outPos.push(
      positions[tIdx * 3],
      positions[tIdx * 3 + 1],
      positions[tIdx * 3 + 2]
    );
    outCol.push(
      tgtColor[0] * dimFactor,
      tgtColor[1] * dimFactor,
      tgtColor[2] * dimFactor
    );
  }
}

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  LAYOUT SETTINGS — Edit these values to adjust spacing             ║
// ╚══════════════════════════════════════════════════════════════════════╝
//
// LAYER_SPACING: Z distance between the start of consecutive layers.
//                Increase this to push layers further apart.
//
// STAGE_*: Z offset within each layer for each transformer sub-block.
//          Data flows along Z in this order:
//          ATTN_NORM → QKV → ATTN_OUT → FFN_NORM → FFN_GATE_UP → FFN_DOWN
//
// All values are in world-space units. The camera auto-fits to the model.

const LAYER_SPACING     = 18.0;   // ← Main knob: distance between layers

const STAGE_ATTN_NORM   =  0.0;   // Pre-attention norm
const STAGE_QKV         =  2.5;   // Query / Key / Value projections
const STAGE_ATTN_OUT    =  7.0;   // Attention output projection
const STAGE_FFN_NORM    = 10.0;   // Pre-FFN norm
const STAGE_FFN_GATE_UP = 12.5;   // FFN gate + up projections
const STAGE_FFN_DOWN    = 16.0;   // FFN down projection

// ─── Layout Engine ──────────────────────────────────────────────────
//
// The layout follows the transformer data-flow order along the Z axis:
//
//   [Embedding]
//       ↓
//   ┌─ Layer 0 ──────────────────────┐
//   │  [Attn Norm]  (thin bar)       │
//   │  [Q] [K] [V]  (side by side)   │
//   │  [Attn Out]   (centered)       │
//   │  [FFN Norm]   (thin bar)       │
//   │  [Gate] [Up]  (side by side)   │
//   │  [Down]       (centered)       │
//   └────────────────────────────────┘
//       ↓
//   ┌─ Layer 1 ─ ... ─┐
//       ↓
//   [Output Norm]
//   [Output]

function computeLayout(archInfo, tensorAllocs) {
  const {
    blockCount,
    headCount,
    headCountKV,
    embeddingLength,
    feedForwardLength,
    expertCount,
    isMoE,
  } = archInfo;

  const layers = blockCount || 1;
  const heads = headCount || 1;
  const headsKV = headCountKV || heads;
  const ffnMult = feedForwardLength ? feedForwardLength / Math.max(embeddingLength, 1) : 4;
  const experts = isMoE ? (expertCount || 1) : 1;

  // ── Per-block constants ──
  const HEAD_WIDTH = 1.0;
  const COMPONENT_GAP = 0.6;
  const EXPERT_GAP = 0.3;
  const TENSOR_HEIGHT_BASE = 2.0;
  const BLOCK_DEPTH = 1.2;        // Z thickness of each sub-block

  // ── Width calculations ──
  const attnQWidth = heads * HEAD_WIDTH;
  const attnKWidth = headsKV * HEAD_WIDTH;
  const attnVWidth = headsKV * HEAD_WIDTH;
  const attnOutWidth = heads * HEAD_WIDTH * 0.5;
  const qkvTotalWidth = attnQWidth + attnKWidth + attnVWidth + COMPONENT_GAP * 2;

  const ffnBlockWidth = Math.max(2, ffnMult) * HEAD_WIDTH;
  const ffnTotalWidth = isMoE
    ? experts * (ffnBlockWidth + EXPERT_GAP) * 0.4
    : ffnBlockWidth * 1.5;

  const centerX = 0;
  const embHeight = TENSOR_HEIGHT_BASE * 1.5;
  const layerHeight = TENSOR_HEIGHT_BASE;

  // Max width for embedding/output blocks
  const globalBlockWidth = Math.max(qkvTotalWidth, ffnTotalWidth) * 0.8;

  return {
    getRegion(category, layerIdx, expertIdx, tensor) {
      // ── Global: Token Embedding ──
      if (category === 'embedding') {
        return {
          x: centerX - globalBlockWidth / 2,
          y: 0,
          z: -LAYER_SPACING * 1.5,
          width: globalBlockWidth,
          height: embHeight,
          depth: BLOCK_DEPTH * 2,
        };
      }

      // ── Global: Output Norm + Output ──
      if (category === 'output_norm') {
        return {
          x: centerX - globalBlockWidth / 2,
          y: 0,
          z: layers * LAYER_SPACING + 0.5,
          width: globalBlockWidth,
          height: 0.3,
          depth: BLOCK_DEPTH,
        };
      }
      if (category === 'output') {
        return {
          x: centerX - globalBlockWidth / 2,
          y: 0,
          z: layers * LAYER_SPACING + 2.0,
          width: globalBlockWidth,
          height: embHeight,
          depth: BLOCK_DEPTH,
        };
      }

      // ── Global: Misc norms / other ──
      if (layerIdx < 0 && (category === 'norm' || category === 'other')) {
        return {
          x: centerX - 2,
          y: 0,
          z: layers * LAYER_SPACING + LAYER_SPACING,
          width: 4,
          height: 0.3,
          depth: BLOCK_DEPTH,
        };
      }

      // ── Block-level tensors (per layer) ──
      const layerZ = Math.max(0, layerIdx) * LAYER_SPACING;

      // ── Stage 0: Pre-Attention Norm (thin bar) ──
      if (category === 'attn_norm') {
        return {
          x: centerX - qkvTotalWidth / 2,
          y: layerHeight + 0.2,
          z: layerZ + STAGE_ATTN_NORM,
          width: qkvTotalWidth,
          height: 0.15,
          depth: BLOCK_DEPTH * 0.5,
        };
      }

      // ── Stage 1: Q, K, V (side by side, centered) ──
      const qkvBaseX = centerX - qkvTotalWidth / 2;

      if (category === 'attn_q') {
        return {
          x: qkvBaseX,
          y: 0,
          z: layerZ + STAGE_QKV,
          width: attnQWidth,
          height: layerHeight,
          depth: BLOCK_DEPTH,
        };
      }

      if (category === 'attn_k') {
        return {
          x: qkvBaseX + attnQWidth + COMPONENT_GAP,
          y: 0,
          z: layerZ + STAGE_QKV,
          width: attnKWidth,
          height: layerHeight * 0.7,  // shorter for GQA
          depth: BLOCK_DEPTH,
        };
      }

      if (category === 'attn_v') {
        return {
          x: qkvBaseX + attnQWidth + attnKWidth + COMPONENT_GAP * 2,
          y: 0,
          z: layerZ + STAGE_QKV,
          width: attnVWidth,
          height: layerHeight * 0.7,  // shorter for GQA
          depth: BLOCK_DEPTH,
        };
      }

      // ── Stage 2: Attention Output (centered) ──
      if (category === 'attn_out' || category === 'attn_other') {
        return {
          x: centerX - attnOutWidth / 2,
          y: 0,
          z: layerZ + STAGE_ATTN_OUT,
          width: attnOutWidth,
          height: layerHeight,
          depth: BLOCK_DEPTH,
        };
      }

      // ── Stage 3: Pre-FFN Norm (thin bar) ──
      if (category === 'ffn_norm') {
        return {
          x: centerX - ffnTotalWidth / 2,
          y: layerHeight + 0.2,
          z: layerZ + STAGE_FFN_NORM,
          width: ffnTotalWidth,
          height: 0.15,
          depth: BLOCK_DEPTH * 0.5,
        };
      }

      // ── Stage 4 & 5: FFN / MoE ──

      // MoE layout
      if (isMoE && experts > 1) {
        // MoE router — thin bar spanning all experts
        if (category === 'moe_gate') {
          return {
            x: centerX - ffnTotalWidth / 2,
            y: layerHeight + 0.5,
            z: layerZ + STAGE_FFN_GATE_UP - 1.0,
            width: ffnTotalWidth,
            height: 0.3,
            depth: BLOCK_DEPTH * 0.5,
          };
        }

        // Packed expert tensors — span all experts
        if (category === 'moe_up') {
          return {
            x: centerX - ffnTotalWidth / 2,
            y: 0,
            z: layerZ + STAGE_FFN_GATE_UP,
            width: ffnTotalWidth,
            height: layerHeight,
            depth: BLOCK_DEPTH,
          };
        }
        if (category === 'moe_down') {
          return {
            x: centerX - ffnTotalWidth / 2,
            y: 0,
            z: layerZ + STAGE_FFN_DOWN,
            width: ffnTotalWidth,
            height: layerHeight,
            depth: BLOCK_DEPTH,
          };
        }

        // Individual expert tensors — positioned at expert column
        const eIdx = Math.max(0, expertIdx);
        const expertWidth = ffnTotalWidth / experts - EXPERT_GAP;
        const moeBaseX = centerX - ffnTotalWidth / 2;
        const expertX = moeBaseX + eIdx * (expertWidth + EXPERT_GAP);

        const subWidth = expertWidth / 2;
        if (category === 'ffn_gate') {
          return {
            x: expertX,
            y: 0,
            z: layerZ + STAGE_FFN_GATE_UP,
            width: subWidth,
            height: layerHeight,
            depth: BLOCK_DEPTH,
          };
        }
        if (category === 'ffn_up') {
          return {
            x: expertX + subWidth,
            y: 0,
            z: layerZ + STAGE_FFN_GATE_UP,
            width: subWidth,
            height: layerHeight,
            depth: BLOCK_DEPTH,
          };
        }
        if (category === 'ffn_down') {
          return {
            x: expertX,
            y: 0,
            z: layerZ + STAGE_FFN_DOWN,
            width: expertWidth,
            height: layerHeight,
            depth: BLOCK_DEPTH,
          };
        }
      }

      // Dense FFN layout (centered)
      const ffnSubWidth = ffnTotalWidth / 3 - COMPONENT_GAP * 0.3;
      const ffnH = layerHeight * Math.min(ffnMult / 4, 1.5);

      // Gate + Up side by side at FFN_GATE_UP stage (centered)
      if (category === 'ffn_gate') {
        const pairWidth = ffnSubWidth * 2 + COMPONENT_GAP * 0.3;
        return {
          x: centerX - pairWidth / 2,
          y: 0,
          z: layerZ + STAGE_FFN_GATE_UP,
          width: ffnSubWidth,
          height: ffnH,
          depth: BLOCK_DEPTH,
        };
      }
      if (category === 'ffn_up') {
        const pairWidth = ffnSubWidth * 2 + COMPONENT_GAP * 0.3;
        return {
          x: centerX - pairWidth / 2 + ffnSubWidth + COMPONENT_GAP * 0.3,
          y: 0,
          z: layerZ + STAGE_FFN_GATE_UP,
          width: ffnSubWidth,
          height: ffnH,
          depth: BLOCK_DEPTH,
        };
      }

      // Down centered at FFN_DOWN stage
      if (category === 'ffn_down') {
        return {
          x: centerX - ffnSubWidth / 2,
          y: 0,
          z: layerZ + STAGE_FFN_DOWN,
          width: ffnSubWidth,
          height: ffnH,
          depth: BLOCK_DEPTH,
        };
      }

      if (category === 'ffn_other') {
        return {
          x: centerX - ffnTotalWidth / 2,
          y: 0,
          z: layerZ + STAGE_FFN_GATE_UP,
          width: ffnTotalWidth,
          height: layerHeight,
          depth: BLOCK_DEPTH,
        };
      }

      // Fallback: unrecognized tensors
      return {
        x: centerX - 3,
        y: -3,
        z: layerIdx >= 0 ? layerZ + STAGE_QKV : layers * LAYER_SPACING + 3,
        width: 6,
        height: 1,
        depth: BLOCK_DEPTH,
      };
    }
  };
}

// ─── Layer Bounding Boxes ───────────────────────────────────────────

/**
 * Compute per-component bounding boxes so each visible cluster of dots
 * (Q, K, V, FFN gate, embedding, output, …) gets its own tight box.
 *
 * Boxes are grouped by (layerIdx, category) to deduplicate regions that
 * share the same spatial area. Each entry also carries an RGB color
 * matching the tensor type palette.
 *
 * @param {Array} tensorRegions - Region metadata from generatePointCloud
 * @returns {Array<{layerIdx: number, category: string, color: number[], min: number[], max: number[]}>}
 */
export function computeLayerBounds(tensorRegions) {
  const groups = new Map();

  for (const region of tensorRegions) {
    const key = `${region.layerIdx}:${region.category}`;
    if (!groups.has(key)) {
      groups.set(key, {
        layerIdx: region.layerIdx,
        category: region.category,
        minX: Infinity, minY: Infinity, minZ: Infinity,
        maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity,
      });
    }
    const b = groups.get(key);
    const r = region.region;
    b.minX = Math.min(b.minX, r.x);
    b.minY = Math.min(b.minY, r.y);
    b.minZ = Math.min(b.minZ, r.z);
    b.maxX = Math.max(b.maxX, r.x + r.width);
    b.maxY = Math.max(b.maxY, r.y + r.height);
    b.maxZ = Math.max(b.maxZ, r.z + r.depth);
  }

  const bounds = [];
  for (const [, b] of groups) {
    const pad = 0.3;
    const color = TENSOR_COLORS[b.category] || TENSOR_COLORS.other;
    bounds.push({
      layerIdx: b.layerIdx,
      category: b.category,
      color,
      min: [b.minX - pad, b.minY - pad, b.minZ - pad],
      max: [b.maxX + pad, b.maxY + pad, b.maxZ + pad],
    });
  }

  bounds.sort((a, b) => a.layerIdx - b.layerIdx);
  return bounds;
}

