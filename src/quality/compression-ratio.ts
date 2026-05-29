/**
 * Compression Ratio Gate — detects repetitive/hallucinated output.
 *
 * Algorithm (matches faster-whisper/whisper.cpp):
 *   textBytes = text.encode('utf-8')
 *   ratio = len(textBytes) / len(deflate(textBytes))
 *   reject if ratio > threshold (default 2.4)
 *
 * Highly repetitive text compresses well → high ratio → reject.
 * Model-agnostic. Pure function. No ONNX dependency.
 */

import { deflate } from 'pako';
import type { QualityGate, QualityGateResult } from './types.js';

export function compressionRatioGate(threshold: number = 2.4): QualityGate {
  return (text: string): QualityGateResult => {
    const bytes = new TextEncoder().encode(text);
    const compressed = deflate(bytes, { level: 6 });
    const compressionRatio = bytes.length / Math.max(compressed.length, 1);

    if (compressionRatio > threshold) {
      return {
        verdict: 'reject',
        compressionRatio,
        reason: `compression_ratio_too_high (${compressionRatio.toFixed(2)} > ${threshold})`,
      };
    }
    return { verdict: 'accept', compressionRatio };
  };
}
