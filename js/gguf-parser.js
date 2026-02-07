/**
 * GGUF Binary Format Parser
 * Parses .gguf files (v2/v3) in the browser using File API + DataView.
 * Only loads the header into memory; weight data is streamed via file.slice().
 */

// GGUF metadata value types
const GGUFValueType = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3,
  UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7,
  STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12
};

// ggml tensor data types
const GGMLType = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3,
  Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
  Q2_K: 10, Q3_K_S: 11, Q3_K_M: 12, Q3_K_L: 13,
  Q4_K_S: 14, Q4_K_M: 15, Q5_K_S: 16, Q5_K_M: 17,
  Q6_K: 18, Q8_K: 19, IQ2_XXS: 20, IQ2_XS: 21,
  IQ3_XXS: 22, IQ1_S: 23, IQ4_NL: 24, IQ3_S: 25,
  IQ2_S: 26, IQ4_XS: 27, I8: 28, I16: 29,
  I32: 30, I64: 31, F64: 32, IQ1_M: 33, BF16: 34
};

const GGMLTypeName = {};
for (const [k, v] of Object.entries(GGMLType)) GGMLTypeName[v] = k;

// Block sizes and byte sizes for quantized types
const QUANT_INFO = {
  [GGMLType.F32]:    { blockSize: 1,   bytesPerBlock: 4 },
  [GGMLType.F16]:    { blockSize: 1,   bytesPerBlock: 2 },
  [GGMLType.BF16]:   { blockSize: 1,   bytesPerBlock: 2 },
  [GGMLType.Q4_0]:   { blockSize: 32,  bytesPerBlock: 18 },
  [GGMLType.Q4_1]:   { blockSize: 32,  bytesPerBlock: 20 },
  [GGMLType.Q5_0]:   { blockSize: 32,  bytesPerBlock: 22 },
  [GGMLType.Q5_1]:   { blockSize: 32,  bytesPerBlock: 24 },
  [GGMLType.Q8_0]:   { blockSize: 32,  bytesPerBlock: 34 },
  [GGMLType.Q8_1]:   { blockSize: 32,  bytesPerBlock: 40 },
  [GGMLType.Q2_K]:   { blockSize: 256, bytesPerBlock: 84 },
  [GGMLType.Q3_K_S]: { blockSize: 256, bytesPerBlock: 110 },
  [GGMLType.Q3_K_M]: { blockSize: 256, bytesPerBlock: 110 },
  [GGMLType.Q3_K_L]: { blockSize: 256, bytesPerBlock: 110 },
  [GGMLType.Q4_K_S]: { blockSize: 256, bytesPerBlock: 144 },
  [GGMLType.Q4_K_M]: { blockSize: 256, bytesPerBlock: 144 },
  [GGMLType.Q5_K_S]: { blockSize: 256, bytesPerBlock: 176 },
  [GGMLType.Q5_K_M]: { blockSize: 256, bytesPerBlock: 176 },
  [GGMLType.Q6_K]:   { blockSize: 256, bytesPerBlock: 210 },
  [GGMLType.Q8_K]:   { blockSize: 256, bytesPerBlock: 292 },
  [GGMLType.I8]:     { blockSize: 1,   bytesPerBlock: 1 },
  [GGMLType.I16]:    { blockSize: 1,   bytesPerBlock: 2 },
  [GGMLType.I32]:    { blockSize: 1,   bytesPerBlock: 4 },
  [GGMLType.I64]:    { blockSize: 1,   bytesPerBlock: 8 },
  [GGMLType.F64]:    { blockSize: 1,   bytesPerBlock: 8 },
};

// Known file type IDs -> human-readable quant names
const FILE_TYPE_NAMES = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0',
  8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M',
  13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S',
  17: 'Q5_K_M', 18: 'Q6_K', 19: 'Q8_K',
};

/**
 * Cursor-style reader over an ArrayBuffer using DataView.
 * Throws BUFFER_TOO_SMALL when it runs out of data so the
 * caller can retry with a larger slice of the file.
 */
class BufferReader {
  constructor(buffer, offset = 0) {
    this.view = new DataView(buffer);
    this.buf = buffer;
    this.offset = offset;
  }

  _need(n) {
    if (this.offset + n > this.buf.byteLength) {
      throw new Error('BUFFER_TOO_SMALL');
    }
  }

  readUint8() {
    this._need(1);
    const v = this.view.getUint8(this.offset);
    this.offset += 1;
    return v;
  }

  readInt8() {
    this._need(1);
    const v = this.view.getInt8(this.offset);
    this.offset += 1;
    return v;
  }

  readUint16() {
    this._need(2);
    const v = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return v;
  }

  readInt16() {
    this._need(2);
    const v = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return v;
  }

  readUint32() {
    this._need(4);
    const v = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return v;
  }

  readInt32() {
    this._need(4);
    const v = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return v;
  }

  readUint64() {
    this._need(8);
    const lo = this.view.getUint32(this.offset, true);
    const hi = this.view.getUint32(this.offset + 4, true);
    this.offset += 8;
    if (hi === 0) return lo;
    return Number(BigInt(hi) << 32n | BigInt(lo));
  }

  readInt64() {
    this._need(8);
    const lo = this.view.getUint32(this.offset, true);
    const hi = this.view.getInt32(this.offset + 4, true);
    this.offset += 8;
    if (hi === 0 && lo < 0x80000000) return lo;
    if (hi === -1 && lo >= 0x80000000) return lo | 0;
    return Number(BigInt(hi) << 32n | BigInt(lo >>> 0));
  }

  readFloat32() {
    this._need(4);
    const v = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return v;
  }

  readFloat64() {
    this._need(8);
    const v = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return v;
  }

  readBool() {
    return this.readUint8() !== 0;
  }

  readString() {
    const len = this.readUint64();
    this._need(len);
    const bytes = new Uint8Array(this.buf, this.offset, len);
    this.offset += len;
    return new TextDecoder().decode(bytes);
  }

  readValue(type) {
    switch (type) {
      case GGUFValueType.UINT8:   return this.readUint8();
      case GGUFValueType.INT8:    return this.readInt8();
      case GGUFValueType.UINT16:  return this.readUint16();
      case GGUFValueType.INT16:   return this.readInt16();
      case GGUFValueType.UINT32:  return this.readUint32();
      case GGUFValueType.INT32:   return this.readInt32();
      case GGUFValueType.FLOAT32: return this.readFloat32();
      case GGUFValueType.BOOL:    return this.readBool();
      case GGUFValueType.STRING:  return this.readString();
      case GGUFValueType.UINT64:  return this.readUint64();
      case GGUFValueType.INT64:   return this.readInt64();
      case GGUFValueType.FLOAT64: return this.readFloat64();
      case GGUFValueType.ARRAY: {
        const arrType = this.readUint32();
        const arrLen = this.readUint64();
        const arr = [];
        for (let i = 0; i < arrLen; i++) {
          arr.push(this.readValue(arrType));
        }
        return arr;
      }
      default:
        throw new Error(`Unknown GGUF value type: ${type}`);
    }
  }
}

/**
 * Parse the GGUF header (metadata + tensor info) from a File object.
 * Automatically retries with a larger buffer when it runs out of data
 * (handles huge tokenizer arrays that can be 10-50 MB in the header).
 * Returns { metadata, tensors, tensorDataOffset, version }
 */
export async function parseGGUFHeader(file, onProgress) {
  // Start with 10 MB, double on each retry up to the full file size
  let readSize = Math.min(file.size, 10 * 1024 * 1024);

  while (true) {
    try {
      const result = await _parseHeader(file, readSize, onProgress);
      return result;
    } catch (e) {
      if (e.message === 'BUFFER_TOO_SMALL' && readSize < file.size) {
        readSize = Math.min(file.size, readSize * 4);
        console.log(`[GGUF] Header exceeded buffer, expanding to ${(readSize / 1e6).toFixed(0)} MB...`);
        continue;
      }
      throw e;
    }
  }
}

async function _parseHeader(file, readSize, onProgress) {
  const ALIGNMENT = 32;

  const buffer = await file.slice(0, readSize).arrayBuffer();
  const reader = new BufferReader(buffer);

  // Magic: first 4 bytes must be ASCII "GGUF"
  const magic = reader.readUint32();
  const expected = 0x46554747;
  if (magic !== expected) {
    throw new Error(`Not a valid GGUF file (got 0x${magic.toString(16)}, expected 0x${expected.toString(16)})`);
  }

  // Version
  const version = reader.readUint32();
  if (version < 2 || version > 3) {
    throw new Error(`Unsupported GGUF version: ${version} (expected 2 or 3)`);
  }

  // Counts
  const tensorCount = reader.readUint64();
  const metadataKVCount = reader.readUint64();

  if (onProgress) onProgress('header', 0, metadataKVCount);

  // Parse metadata KV pairs (this is where huge arrays live)
  const metadata = {};
  for (let i = 0; i < metadataKVCount; i++) {
    const key = reader.readString();
    const valueType = reader.readUint32();
    const value = reader.readValue(valueType);
    metadata[key] = value;
    if (onProgress && i % 50 === 0) onProgress('metadata', i, metadataKVCount);
  }

  if (onProgress) onProgress('tensors', 0, tensorCount);

  // Parse tensor info entries
  const tensors = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = reader.readString();
    const nDims = reader.readUint32();
    const dims = [];
    for (let d = 0; d < nDims; d++) {
      dims.push(reader.readUint64());
    }
    const type = reader.readUint32();
    const offset = reader.readUint64();

    const numElements = dims.reduce((a, b) => a * b, 1);
    const info = QUANT_INFO[type];
    const dataSize = info
      ? Math.ceil(numElements / info.blockSize) * info.bytesPerBlock
      : numElements * 4;

    tensors.push({ name, dims, type, offset, numElements, dataSize });
    if (onProgress && i % 100 === 0) onProgress('tensors', i, tensorCount);
  }

  // Tensor data starts after alignment padding
  const headerEnd = reader.offset;
  const tensorDataOffset = Math.ceil(headerEnd / ALIGNMENT) * ALIGNMENT;

  return { metadata, tensors, tensorDataOffset, version, tensorCount };
}

/**
 * Extract architecture-related metadata into a friendly object.
 */
export function extractArchInfo(metadata) {
  const arch = metadata['general.architecture'] || 'unknown';
  const get = (key) => metadata[key] ?? metadata[`${arch}.${key}`];

  const info = {
    architecture: arch,
    name: metadata['general.name'] || 'Unknown Model',
    fileType: metadata['general.file_type'],
    fileTypeName: FILE_TYPE_NAMES[metadata['general.file_type']] || `Type ${metadata['general.file_type']}`,
    quantVersionOverride: metadata['general.quantization_version'],

    blockCount: get('block_count') || 0,
    contextLength: get('context_length') || 0,
    embeddingLength: get('embedding_length') || 0,
    feedForwardLength: get('feed_forward_length') || 0,

    headCount: get('attention.head_count') || 0,
    headCountKV: get('attention.head_count_kv') || get('attention.head_count') || 0,

    expertCount: get('expert_count') || 0,
    expertUsedCount: get('expert_used_count') || 0,

    vocabSize: get('vocab_size') || metadata['tokenizer.ggml.tokens']?.length || 0,
    ropeFreqBase: get('rope.freq_base') || 0,
    ropeDimensionCount: get('rope.dimension_count') || 0,
  };

  info.isMoE = info.expertCount > 1;
  info.isGQA = info.headCountKV > 0 && info.headCountKV < info.headCount;
  info.headDim = info.embeddingLength && info.headCount
    ? Math.floor(info.embeddingLength / info.headCount) : 0;

  return info;
}

/**
 * Sample weight values from a specific tensor in the file.
 * Returns Float32Array of dequantized values.
 */
export async function sampleTensorWeights(file, tensorDataOffset, tensor, sampleCount) {
  const { type, offset, numElements, dataSize } = tensor;
  const absOffset = tensorDataOffset + offset;

  // Cap sample count
  const actualSamples = Math.min(sampleCount, numElements);

  // Strategy depends on type
  if (type === GGMLType.F32) {
    return sampleF32(file, absOffset, numElements, actualSamples);
  } else if (type === GGMLType.F16) {
    return sampleF16(file, absOffset, numElements, actualSamples);
  } else if (type === GGMLType.BF16) {
    return sampleBF16(file, absOffset, numElements, actualSamples);
  } else if (type === GGMLType.Q8_0) {
    return sampleQ8_0(file, absOffset, numElements, actualSamples);
  } else if (type === GGMLType.Q4_0) {
    return sampleQ4_0(file, absOffset, numElements, actualSamples);
  } else {
    // Generic: read raw bytes and normalize to [-1, 1]
    return sampleGeneric(file, absOffset, dataSize, numElements, actualSamples);
  }
}

// ─── Sampling helpers ───────────────────────────────────────────────

async function sampleF32(file, absOffset, numElements, sampleCount) {
  const stride = numElements / sampleCount;
  const result = new Float32Array(sampleCount);

  // Read in chunks to avoid huge memory allocation
  const CHUNK = 65536;
  const bytesPerElem = 4;
  let sampleIdx = 0;
  let nextSampleAt = 0;

  for (let start = 0; start < numElements && sampleIdx < sampleCount; start += CHUNK) {
    const end = Math.min(start + CHUNK, numElements);
    if (nextSampleAt >= end) continue;

    const buf = await file.slice(
      absOffset + start * bytesPerElem,
      absOffset + end * bytesPerElem
    ).arrayBuffer();
    const view = new Float32Array(buf);

    while (nextSampleAt < end && sampleIdx < sampleCount) {
      const localIdx = Math.floor(nextSampleAt) - start;
      if (localIdx >= 0 && localIdx < view.length) {
        result[sampleIdx++] = view[localIdx];
      }
      nextSampleAt += stride;
    }
  }

  return result;
}

async function sampleF16(file, absOffset, numElements, sampleCount) {
  const stride = numElements / sampleCount;
  const result = new Float32Array(sampleCount);
  const CHUNK = 65536;
  let sampleIdx = 0;
  let nextSampleAt = 0;

  for (let start = 0; start < numElements && sampleIdx < sampleCount; start += CHUNK) {
    const end = Math.min(start + CHUNK, numElements);
    if (nextSampleAt >= end) continue;

    const buf = await file.slice(
      absOffset + start * 2,
      absOffset + end * 2
    ).arrayBuffer();
    const u16 = new Uint16Array(buf);

    while (nextSampleAt < end && sampleIdx < sampleCount) {
      const localIdx = Math.floor(nextSampleAt) - start;
      if (localIdx >= 0 && localIdx < u16.length) {
        result[sampleIdx++] = f16ToF32(u16[localIdx]);
      }
      nextSampleAt += stride;
    }
  }
  return result;
}

async function sampleBF16(file, absOffset, numElements, sampleCount) {
  const stride = numElements / sampleCount;
  const result = new Float32Array(sampleCount);
  const CHUNK = 65536;
  let sampleIdx = 0;
  let nextSampleAt = 0;

  for (let start = 0; start < numElements && sampleIdx < sampleCount; start += CHUNK) {
    const end = Math.min(start + CHUNK, numElements);
    if (nextSampleAt >= end) continue;

    const buf = await file.slice(
      absOffset + start * 2,
      absOffset + end * 2
    ).arrayBuffer();
    const u16 = new Uint16Array(buf);

    while (nextSampleAt < end && sampleIdx < sampleCount) {
      const localIdx = Math.floor(nextSampleAt) - start;
      if (localIdx >= 0 && localIdx < u16.length) {
        // BF16: just shift to upper 16 bits of float32
        const tmpBuf = new ArrayBuffer(4);
        const tmpView = new DataView(tmpBuf);
        tmpView.setUint16(2, u16[localIdx], true);
        tmpView.setUint16(0, 0, true);
        result[sampleIdx++] = tmpView.getFloat32(0, true);
      }
      nextSampleAt += stride;
    }
  }
  return result;
}

async function sampleQ8_0(file, absOffset, numElements, sampleCount) {
  // Q8_0: blocks of 32 elements, each block = 2 bytes (f16 scale) + 32 bytes (int8 quants)
  const BLOCK_SIZE = 32;
  const BYTES_PER_BLOCK = 34;
  const stride = numElements / sampleCount;
  const result = new Float32Array(sampleCount);

  // Build list of blocks we need
  const blockNeeds = new Map(); // blockIdx -> [{ sampleIdx, inBlock }]
  for (let i = 0; i < sampleCount; i++) {
    const elemIdx = Math.floor(i * stride);
    const blockIdx = Math.floor(elemIdx / BLOCK_SIZE);
    const inBlock = elemIdx % BLOCK_SIZE;
    if (!blockNeeds.has(blockIdx)) blockNeeds.set(blockIdx, []);
    blockNeeds.get(blockIdx).push({ sampleIdx: i, inBlock });
  }

  // Read blocks in batched ranges (batch up to 4096 consecutive blocks at once)
  const sortedBlocks = [...blockNeeds.keys()].sort((a, b) => a - b);
  const MAX_BATCH = 4096;
  let bi = 0;

  while (bi < sortedBlocks.length) {
    const firstBlock = sortedBlocks[bi];
    let lastBlock = firstBlock;
    let endBi = bi;

    while (endBi + 1 < sortedBlocks.length &&
           sortedBlocks[endBi + 1] - firstBlock < MAX_BATCH) {
      endBi++;
      lastBlock = sortedBlocks[endBi];
    }

    const readStart = absOffset + firstBlock * BYTES_PER_BLOCK;
    const readEnd = absOffset + (lastBlock + 1) * BYTES_PER_BLOCK;
    const buf = await file.slice(readStart, readEnd).arrayBuffer();
    const data = new DataView(buf);
    const bytes = new Uint8Array(buf);

    for (let j = bi; j <= endBi; j++) {
      const blockIdx = sortedBlocks[j];
      const localOff = (blockIdx - firstBlock) * BYTES_PER_BLOCK;
      const scaleU16 = data.getUint16(localOff, true);
      const scale = f16ToF32(scaleU16);

      for (const { sampleIdx, inBlock } of blockNeeds.get(blockIdx)) {
        const qval = bytes[localOff + 2 + inBlock];
        const ival = qval > 127 ? qval - 256 : qval;
        result[sampleIdx] = scale * ival;
      }
    }
    bi = endBi + 1;
  }

  return result;
}

async function sampleQ4_0(file, absOffset, numElements, sampleCount) {
  // Q4_0: blocks of 32 elements, each block = 2 bytes (f16 scale) + 16 bytes (4-bit pairs)
  const BLOCK_SIZE = 32;
  const BYTES_PER_BLOCK = 18;
  const stride = numElements / sampleCount;
  const result = new Float32Array(sampleCount);

  // Build list of blocks we need (sparse reads for large tensors)
  const blockNeeds = new Map();
  for (let i = 0; i < sampleCount; i++) {
    const elemIdx = Math.floor(i * stride);
    const blockIdx = Math.floor(elemIdx / BLOCK_SIZE);
    const inBlock = elemIdx % BLOCK_SIZE;
    if (!blockNeeds.has(blockIdx)) blockNeeds.set(blockIdx, []);
    blockNeeds.get(blockIdx).push({ sampleIdx: i, inBlock });
  }

  const sortedBlocks = [...blockNeeds.keys()].sort((a, b) => a - b);
  const MAX_BATCH = 8192;
  let bi = 0;

  while (bi < sortedBlocks.length) {
    const firstBlock = sortedBlocks[bi];
    let lastBlock = firstBlock;
    let endBi = bi;

    while (endBi + 1 < sortedBlocks.length &&
           sortedBlocks[endBi + 1] - firstBlock < MAX_BATCH) {
      endBi++;
      lastBlock = sortedBlocks[endBi];
    }

    const readStart = absOffset + firstBlock * BYTES_PER_BLOCK;
    const readEnd = absOffset + (lastBlock + 1) * BYTES_PER_BLOCK;
    const buf = await file.slice(readStart, readEnd).arrayBuffer();
    const data = new DataView(buf);
    const bytes = new Uint8Array(buf);

    for (let j = bi; j <= endBi; j++) {
      const blockIdx = sortedBlocks[j];
      const localOff = (blockIdx - firstBlock) * BYTES_PER_BLOCK;
      const scaleU16 = data.getUint16(localOff, true);
      const scale = f16ToF32(scaleU16);

      for (const { sampleIdx, inBlock } of blockNeeds.get(blockIdx)) {
        const byteIdx = Math.floor(inBlock / 2);
        const nibble = (inBlock % 2 === 0)
          ? (bytes[localOff + 2 + byteIdx] & 0x0F)
          : (bytes[localOff + 2 + byteIdx] >> 4);
        const ival = nibble - 8;
        result[sampleIdx] = scale * ival;
      }
    }
    bi = endBi + 1;
  }

  return result;
}

async function sampleGeneric(file, absOffset, dataSize, numElements, sampleCount) {
  // Fallback: read raw bytes from sparse positions, normalize to [-1, 1]
  const stride = Math.max(1, Math.floor(dataSize / sampleCount));
  const result = new Float32Array(sampleCount);

  // Read in chunks, skipping over unneeded regions
  const CHUNK = 256 * 1024; // 256KB per read
  let sampleIdx = 0;
  let nextByteAt = 0;

  for (let start = 0; start < dataSize && sampleIdx < sampleCount; start += CHUNK) {
    const end = Math.min(start + CHUNK, dataSize);
    if (nextByteAt >= end) continue;

    const buf = await file.slice(absOffset + start, absOffset + end).arrayBuffer();
    const bytes = new Uint8Array(buf);

    while (nextByteAt < end && sampleIdx < sampleCount) {
      const localIdx = Math.floor(nextByteAt) - start;
      if (localIdx >= 0 && localIdx < bytes.length) {
        const raw = bytes[localIdx];
        result[sampleIdx] = (raw > 127 ? raw - 256 : raw) / 128.0;
        sampleIdx++;
      }
      nextByteAt += stride;
    }
  }

  return result;
}

// ─── Float16 conversion ─────────────────────────────────────────────

function f16ToF32(h) {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7C00) >> 10;
  const frac = h & 0x03FF;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Subnormal
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 0x1F) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/**
 * Compute total parameter count from tensor list.
 */
export function computeTotalParams(tensors) {
  return tensors.reduce((sum, t) => sum + t.numElements, 0);
}

/**
 * Classify a tensor by its name into a category.
 */
export function classifyTensor(name) {
  const n = name.toLowerCase();
  // Extract block/layer index
  const blockMatch = n.match(/blk\.(\d+)\./);
  const layerIdx = blockMatch ? parseInt(blockMatch[1]) : -1;

  // Expert index
  const expertMatch = n.match(/ffn_gate_exps|ffn_up_exps|ffn_down_exps/) ? -1 :
    n.match(/\.(\d+)\.weight/) ? null : null;
  // More robust expert detection
  let expertIdx = -1;
  const expMatch = n.match(/ffn_(?:gate|up|down)\.(\d+)/);
  if (expMatch) expertIdx = parseInt(expMatch[1]);

  let category = 'other';
  if (n.includes('token_embd') || n.includes('tok_embd')) category = 'embedding';
  else if (n.includes('output_norm') || n.includes('result_norm')) category = 'output_norm';
  else if (n.includes('output.weight') || n.includes('lm_head')) category = 'output';
  else if (n.includes('attn_q') || n.includes('attn.q')) category = 'attn_q';
  else if (n.includes('attn_k') || n.includes('attn.k')) category = 'attn_k';
  else if (n.includes('attn_v') || n.includes('attn.v')) category = 'attn_v';
  else if (n.includes('attn_output') || n.includes('attn.output') || n.includes('attn_o')) category = 'attn_out';
  else if (n.includes('attn_norm') || n.includes('attn_ln')) category = 'attn_norm';
  else if (n.includes('ffn_gate_exps') || n.includes('ffn_gate_inp')) category = 'moe_gate';
  else if (n.includes('ffn_gate')) category = 'ffn_gate';
  else if (n.includes('ffn_up_exps')) category = 'moe_up';
  else if (n.includes('ffn_up')) category = 'ffn_up';
  else if (n.includes('ffn_down_exps')) category = 'moe_down';
  else if (n.includes('ffn_down')) category = 'ffn_down';
  else if (n.includes('ffn_norm')) category = 'ffn_norm';
  else if (n.includes('attn')) category = 'attn_other';
  else if (n.includes('ffn')) category = 'ffn_other';
  else if (n.includes('norm')) category = 'norm';

  return { category, layerIdx, expertIdx };
}

export { GGMLType, GGMLTypeName, QUANT_INFO };

