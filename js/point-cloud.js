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

  moe_gate:   [1.00, 0.40, 0.67],  // #ff66aa
  moe_up:     [1.00, 0.53, 0.73],  // #ff88bb
  moe_down:   [1.00, 0.27, 0.53],  // #ff4488

  embedding:  [0.27, 0.87, 0.53],  // #44dd88
  output:     [0.67, 0.40, 1.00],  // #aa66ff
  output_norm:[0.87, 0.87, 0.27],  // #dddd44
  norm:       [0.87, 0.87, 0.27],  // #dddd44
  other:      [0.53, 0.53, 0.60],  // #888899
};

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
export async function generatePointCloud(file, archInfo, tensors, tensorDataOffset, targetPointCount, colorMode, onProgress) {
  // Compute total params and allocate points proportionally
  const totalParams = tensors.reduce((s, t) => s + t.numElements, 0);
  const decimationRatio = totalParams / targetPointCount;

  // Assign point counts to each tensor proportionally
  const tensorAllocs = tensors.map(t => {
    const cls = classifyTensor(t.name);
    const rawCount = Math.max(1, Math.floor(t.numElements / decimationRatio));
    return { tensor: t, cls, pointCount: rawCount };
  });

  // Adjust total to match target exactly
  let allocTotal = tensorAllocs.reduce((s, a) => s + a.pointCount, 0);
  // Scale if needed
  if (allocTotal > 0) {
    const scale = targetPointCount / allocTotal;
    tensorAllocs.forEach(a => {
      a.pointCount = Math.max(1, Math.round(a.pointCount * scale));
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
      category: cls.category,
      layerIdx: cls.layerIdx,
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
  // Connection rules: [sourceCategories, targetCategories]
  const INTRA_LAYER_RULES = [
    // Attention: Q, K, V feed into Out
    { from: ['attn_q'],   to: ['attn_out'] },
    { from: ['attn_k'],   to: ['attn_out'] },
    { from: ['attn_v'],   to: ['attn_out'] },
    // Attention output feeds into FFN
    { from: ['attn_out'],  to: ['ffn_gate', 'ffn_up'] },
    // FFN internal flow
    { from: ['ffn_gate'],  to: ['ffn_down'] },
    { from: ['ffn_up'],    to: ['ffn_down'] },
    // MoE routing
    { from: ['moe_gate'],  to: ['moe_up'] },
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
        for (const tgt of targets) {
          _sampleLines(src, tgt, positions, linePositions, lineColors, density);
        }
      }
    }
  }

  // ── Cross-layer connections: FFN down → next layer's attention ──
  for (let i = 0; i < layers.length - 1; i++) {
    const currentRegions = layerMap.get(layers[i]);
    const nextRegions = layerMap.get(layers[i + 1]);

    const downs = currentRegions.filter(r => r.category === 'ffn_down' || r.category === 'moe_down');
    const nextAttns = nextRegions.filter(r => ['attn_q', 'attn_k', 'attn_v'].includes(r.category));

    for (const src of downs) {
      // Connect to one random target attention type (not all, to keep lines tidy)
      if (nextAttns.length > 0) {
        const tgt = nextAttns[Math.floor(Math.random() * nextAttns.length)];
        _sampleLines(src, tgt, positions, linePositions, lineColors, density * 0.3, 0.35);
      }
    }
  }

  // ── Embedding → first layer ──
  if (layers.length > 0) {
    const embeddings = globals.filter(r => r.category === 'embedding');
    const firstAttns = (layerMap.get(layers[0]) || []).filter(r =>
      ['attn_q', 'attn_k', 'attn_v'].includes(r.category)
    );
    for (const src of embeddings) {
      for (const tgt of firstAttns) {
        _sampleLines(src, tgt, positions, linePositions, lineColors, density * 0.5, 0.4);
      }
    }
  }

  // ── Last layer → output ──
  if (layers.length > 0) {
    const lastRegions = layerMap.get(layers[layers.length - 1]) || [];
    const lastDowns = lastRegions.filter(r => r.category === 'ffn_down' || r.category === 'moe_down');
    const outputs = globals.filter(r => r.category === 'output');

    for (const src of lastDowns) {
      for (const tgt of outputs) {
        _sampleLines(src, tgt, positions, linePositions, lineColors, density * 0.5, 0.4);
      }
    }
  }

  const totalLines = linePositions.length / 6; // 6 floats per line (2 vertices × 3)

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

// ─── Layout Engine ──────────────────────────────────────────────────

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

  // Spacing constants (these define the overall shape)
  const LAYER_SPACING = 3.0;
  const HEAD_WIDTH = 1.0;
  const COMPONENT_GAP = 0.6;
  const EXPERT_GAP = 0.3;
  const TENSOR_HEIGHT_BASE = 2.0;
  const TENSOR_DEPTH = 1.5;

  // Attention section width: Q(heads) + K(headsKV) + V(headsKV) + Out(1) + gaps
  const attnQWidth = heads * HEAD_WIDTH;
  const attnKWidth = headsKV * HEAD_WIDTH;
  const attnVWidth = headsKV * HEAD_WIDTH;
  const attnOutWidth = heads * HEAD_WIDTH * 0.5;
  const attnTotalWidth = attnQWidth + attnKWidth + attnVWidth + attnOutWidth + COMPONENT_GAP * 3;

  // FFN section width
  const ffnBlockWidth = Math.max(2, ffnMult) * HEAD_WIDTH;
  const ffnTotalWidth = isMoE
    ? experts * (ffnBlockWidth + EXPERT_GAP) * 0.4
    : ffnBlockWidth * 1.5;

  const totalLayerWidth = attnTotalWidth + COMPONENT_GAP * 2 + ffnTotalWidth;

  // Center everything
  const centerX = 0;
  const baseX = centerX - totalLayerWidth / 2;

  // Embedding / output heights
  const embHeight = TENSOR_HEIGHT_BASE * 1.5;
  const layerHeight = TENSOR_HEIGHT_BASE;

  return {
    getRegion(category, layerIdx, expertIdx, tensor) {
      // Special tensors (not in blocks)
      if (category === 'embedding') {
        return {
          x: centerX - attnTotalWidth * 0.4,
          y: 0,
          z: -LAYER_SPACING * 2,
          width: attnTotalWidth * 0.8,
          height: embHeight,
          depth: TENSOR_DEPTH * 2,
        };
      }

      if (category === 'output' || category === 'output_norm') {
        const zOff = category === 'output_norm' ? 0.5 : 0;
        return {
          x: centerX - attnTotalWidth * 0.4,
          y: 0,
          z: layers * LAYER_SPACING + LAYER_SPACING + zOff,
          width: attnTotalWidth * 0.8,
          height: embHeight,
          depth: TENSOR_DEPTH,
        };
      }

      // Norm layers not in blocks
      if (layerIdx < 0 && (category === 'norm' || category === 'other')) {
        return {
          x: centerX - 2,
          y: 0,
          z: layers * LAYER_SPACING + LAYER_SPACING * 2,
          width: 4,
          height: 0.3,
          depth: TENSOR_DEPTH,
        };
      }

      // Block-level tensors
      const layerZ = Math.max(0, layerIdx) * LAYER_SPACING;
      let x = baseX;

      // Attention Q
      if (category === 'attn_q') {
        return {
          x: x,
          y: 0,
          z: layerZ,
          width: attnQWidth,
          height: layerHeight,
          depth: TENSOR_DEPTH,
        };
      }

      x += attnQWidth + COMPONENT_GAP;

      // Attention K
      if (category === 'attn_k') {
        return {
          x: x,
          y: 0,
          z: layerZ,
          width: attnKWidth,
          height: layerHeight * 0.7, // Visually shorter for GQA
          depth: TENSOR_DEPTH,
        };
      }

      x += attnKWidth + COMPONENT_GAP;

      // Attention V
      if (category === 'attn_v') {
        return {
          x: x,
          y: 0,
          z: layerZ,
          width: attnVWidth,
          height: layerHeight * 0.7,
          depth: TENSOR_DEPTH,
        };
      }

      x += attnVWidth + COMPONENT_GAP;

      // Attention Output
      if (category === 'attn_out' || category === 'attn_other') {
        return {
          x: x,
          y: 0,
          z: layerZ,
          width: attnOutWidth,
          height: layerHeight,
          depth: TENSOR_DEPTH,
        };
      }

      // Attention norm (thin slice)
      if (category === 'attn_norm') {
        return {
          x: baseX,
          y: layerHeight + 0.2,
          z: layerZ,
          width: attnTotalWidth,
          height: 0.15,
          depth: TENSOR_DEPTH,
        };
      }

      x += attnOutWidth + COMPONENT_GAP * 2;
      const ffnBaseX = x;

      // FFN norm (thin slice)
      if (category === 'ffn_norm') {
        return {
          x: ffnBaseX,
          y: layerHeight + 0.2,
          z: layerZ,
          width: ffnTotalWidth,
          height: 0.15,
          depth: TENSOR_DEPTH,
        };
      }

      // FFN layers
      if (isMoE && experts > 1) {
        // MoE layout: each expert gets its own column
        const eIdx = Math.max(0, expertIdx);
        const expertWidth = ffnTotalWidth / experts - EXPERT_GAP;
        const expertX = ffnBaseX + eIdx * (expertWidth + EXPERT_GAP);

        // MoE gate (router) spans all experts
        if (category === 'moe_gate') {
          return {
            x: ffnBaseX,
            y: layerHeight + 0.5,
            z: layerZ,
            width: ffnTotalWidth,
            height: 0.3,
            depth: TENSOR_DEPTH,
          };
        }

        const subWidth = expertWidth / 3;
        if (category === 'ffn_gate' || category === 'moe_gate') {
          return {
            x: expertX,
            y: 0,
            z: layerZ,
            width: subWidth,
            height: layerHeight,
            depth: TENSOR_DEPTH,
          };
        }
        if (category === 'ffn_up' || category === 'moe_up') {
          return {
            x: expertX + subWidth,
            y: 0,
            z: layerZ,
            width: subWidth,
            height: layerHeight,
            depth: TENSOR_DEPTH,
          };
        }
        if (category === 'ffn_down' || category === 'moe_down') {
          return {
            x: expertX + subWidth * 2,
            y: 0,
            z: layerZ,
            width: subWidth,
            height: layerHeight,
            depth: TENSOR_DEPTH,
          };
        }
      }

      // Dense FFN layout
      const ffnSubWidth = ffnTotalWidth / 3 - COMPONENT_GAP * 0.3;

      if (category === 'ffn_gate') {
        return {
          x: ffnBaseX,
          y: 0,
          z: layerZ,
          width: ffnSubWidth,
          height: layerHeight * Math.min(ffnMult / 4, 1.5),
          depth: TENSOR_DEPTH,
        };
      }
      if (category === 'ffn_up') {
        return {
          x: ffnBaseX + ffnSubWidth + COMPONENT_GAP * 0.3,
          y: 0,
          z: layerZ,
          width: ffnSubWidth,
          height: layerHeight * Math.min(ffnMult / 4, 1.5),
          depth: TENSOR_DEPTH,
        };
      }
      if (category === 'ffn_down') {
        return {
          x: ffnBaseX + (ffnSubWidth + COMPONENT_GAP * 0.3) * 2,
          y: 0,
          z: layerZ,
          width: ffnSubWidth,
          height: layerHeight * Math.min(ffnMult / 4, 1.5),
          depth: TENSOR_DEPTH,
        };
      }

      if (category === 'ffn_other') {
        return {
          x: ffnBaseX,
          y: 0,
          z: layerZ,
          width: ffnTotalWidth,
          height: layerHeight,
          depth: TENSOR_DEPTH,
        };
      }

      // Fallback: place unrecognized tensors at the edge
      return {
        x: centerX - 3,
        y: -3,
        z: layerIdx >= 0 ? layerIdx * LAYER_SPACING : layers * LAYER_SPACING + 3,
        width: 6,
        height: 1,
        depth: TENSOR_DEPTH,
      };
    }
  };
}

