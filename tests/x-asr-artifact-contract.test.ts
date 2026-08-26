import { describe, expect, it } from 'vitest';
import { inspectXAsrArtifactContract } from '../tools/model-debugging/scripts/x-asr-artifact-contract.mjs';

describe('X-ASR artifact contract', () => {
  it('reports graph boundary metadata and accepts the four-graph shape contract', () => {
    const modelDir = 'C:\\x-asr';
    const files = [
      `${modelDir}\\chunk-160ms-model\\encoder-160ms.onnx`,
      `${modelDir}\\chunk-160ms-model\\decoder-160ms.onnx`,
      `${modelDir}\\chunk-160ms-model\\joiner-160ms.onnx`,
      `${modelDir}\\chunk-160ms-model\\tokens.txt`,
    ];
    const graphs = [
      { path: 'chunk-160ms-model/encoder-160ms.onnx', loaded: true, input_names: ['x', 'state'], output_names: ['y', 'next_state'], input_metadata: [{ name: 'x', type: 'float32', dimensions: [1, 16, 80] }] },
      { path: 'chunk-160ms-model/decoder-160ms.onnx', loaded: true, input_names: ['y'], output_names: ['decoder_out'] },
      { path: 'chunk-160ms-model/joiner-160ms.onnx', loaded: true, input_names: ['enc', 'dec'], output_names: ['logits'] },
    ];
    const report = inspectXAsrArtifactContract({ modelDir, files, graphs });
    expect(report.variants[0]?.ok).toBe(true);
    expect(report.variants[0]?.boundary_checks.every((check) => check.status === 'pass')).toBe(true);
    expect(report.variants[0]?.graphs.encoder.input_metadata[0]?.dimensions).toEqual([1, 16, 80]);
    expect(report.ok).toBe(false);
  });
});
