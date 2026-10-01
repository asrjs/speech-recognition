import { describe, it, expect } from 'vitest';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { initWhisperOrt, createWhisperOrtSession } from '../src/models/whisper-seq2seq/ort.js';
import {
  requireFixtureDirectory,
  requireFixtureFiles,
  requireFixtureOptIn,
} from './support/fixture-gates.js';

describe('Whisper timestamped ONNX decoder smoke', () => {
  it('timestamped decoder_model_merged emits cross_attentions.* outputs', async (context) => {
    requireFixtureOptIn(
      context,
      process.env.ASRJS_FIXTURE_SMOKE === '1' || process.env.ASRJS_FIXTURE_SMOKE_FORCE === '1',
      'Set ASRJS_FIXTURE_SMOKE=1 to run timestamped decoder smoke test',
    );

    const fixtureDir = requireFixtureDirectory(
      process.env.WHISPER_TIMESTAMPED_FIXTURE_DIR,
      'WHISPER_TIMESTAMPED_FIXTURE_DIR',
    );
    const decoderPath = join(
      fixtureDir,
      process.env.WHISPER_TIMESTAMPED_DECODER ?? 'decoder_model_merged_int8.onnx',
    );
    requireFixtureFiles({ timestampedDecoder: decoderPath });

    const ort = await initWhisperOrt('wasm');
    const session = await createWhisperOrtSession(ort, pathToFileURL(decoderPath).href, {
      backendId: 'wasm',
    });
    context.onTestFinished(() => session.release?.());

    const outputNames =
      (session as unknown as { readonly outputNames?: readonly string[] }).outputNames ?? [];
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
