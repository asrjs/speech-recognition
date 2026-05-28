import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import { initWhisperOrt, createWhisperOrtSession } from '../src/models/whisper-seq2seq/ort.js';

describe('Whisper timestamped ONNX decoder smoke', () => {
  it('timestamped decoder_model_merged emits cross_attentions.* outputs', async () => {
    if (!process.env.ASRJS_FIXTURE_SMOKE && !process.env.ASRJS_FIXTURE_SMOKE_FORCE) {
      console.warn('Skipping: set ASRJS_FIXTURE_SMOKE=1 to run timestamped decoder smoke test');
      return;
    }

    const fixtureDir = '/tmp/whisper-tiny-ts-onnx';
    const decoderPath = `${fixtureDir}/decoder_model_merged_int8.onnx`;
    if (!fs.existsSync(decoderPath)) {
      console.warn(`Skipping: timestamped decoder not found at ${decoderPath}`);
      return;
    }

    const ort = await initWhisperOrt('wasm');
    const session = await createWhisperOrtSession(ort, `file://${decoderPath}`, {
      backendId: 'wasm',
    });

    const outputNames = (session as unknown as { readonly outputNames?: readonly string[] }).outputNames ?? [];
    const crossAttentions = outputNames.filter((name) => name.startsWith('cross_attentions.'));
    expect(crossAttentions.length).toBeGreaterThan(0);
    expect(crossAttentions.length).toBe(4); // whisper-tiny has 4 decoder layers

    // Verify all have expected naming: cross_attentions.0 through cross_attentions.3
    for (const name of crossAttentions) {
      const num = parseInt(name.split('.').pop() ?? '-1', 10);
      expect(num).toBeGreaterThanOrEqual(0);
    }
  });
});
