/**
 * Web Worker for GGUF header parsing.
 * Offloads the CPU-heavy metadata parsing (especially large tokenizer arrays)
 * to a background thread so the UI stays responsive.
 */

import { parseGGUFHeader, extractArchInfo, computeTotalParams } from './gguf-parser.js';

self.onmessage = async function (e) {
  const { file } = e.data;

  try {
    const result = await parseGGUFHeader(file, (phase, current, total) => {
      self.postMessage({ type: 'progress', phase, current, total });
    });

    const archInfo = extractArchInfo(result.metadata);
    const totalParams = computeTotalParams(result.tensors);

    self.postMessage({
      type: 'result',
      metadata: result.metadata,
      tensors: result.tensors,
      tensorDataOffset: result.tensorDataOffset,
      version: result.version,
      archInfo,
      totalParams,
    });
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
};

