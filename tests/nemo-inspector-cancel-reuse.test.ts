import { describe, expect, it, vi } from 'vitest';
import { PipelineAbortedError } from '@asrjs/speech-recognition';
import {
  parseArgs,
  runInspectorCancelReuse,
  validateInspectorOptions,
} from '../tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs';

const reference = {
  tokens: [1],
  token_frames: [0],
  token_duration: [1],
  transcript: 'probe',
};
const output = {
  canonical: { text: 'probe' },
  native: { debug: { tokenIds: [1], frameIndices: [0], tdtSteps: [1] } },
};
const progress = { stage: 'decode', completedUnits: 1, totalUnits: 10 };

describe('NeMo inspector cancellation/reuse controls (simulated)', () => {
  it('requires an explicitly requested oracle and leaves default behavior unchanged', () => {
    expect(parseArgs([]).cancelReuse).toBe(false);
    const options = parseArgs([
      '--cancel-reuse',
      '--model-dir',
      '.',
      '--family',
      'nemo-tdt',
      '--config',
      'config.json',
    ]);
    expect(options.cancelReuse).toBe(true);
    expect(() => validateInspectorOptions(options)).toThrow('requires an official --oracle');
  });

  it('aborts inside decode, then uses fresh signals for two exact oracle repeats', async () => {
    const signals: AbortSignal[] = [];
    const input = Float32Array.of(0.1);
    const transcribe = vi.fn(async (received, options) => {
      expect(received).toBe(input);
      signals.push(options.signal);
      if (signals.length === 1) {
        options.onProgress(progress);
        expect(options.signal.aborted).toBe(true);
        throw new PipelineAbortedError('decode');
      }
      expect(options.signal.aborted).toBe(false);
      return output;
    });
    const loaded = { model: {}, session: {}, transcribe };
    const result = await runInspectorCancelReuse(loaded, input, {}, reference, 'fixture');
    expect(transcribe).toHaveBeenCalledTimes(3);
    expect(new Set(signals).size).toBe(3);
    expect(result.result).toBe(output);
    expect(result.lifecycle.abortedAt).toEqual(progress);
    expect(result.lifecycle.comparisons.every((row) => row.pass)).toBe(true);
  });

  it.each([
    ['no progress', null],
    ['ignored cancellation', progress],
    ['completed decode', { ...progress, completedUnits: 10 }],
    ['noninteger progress', { ...progress, completedUnits: 0.5 }],
  ])('rejects %s rather than certifying cancellation', async (_name, event) => {
    const transcribe = vi.fn(async (_input, options) => {
      if (event) options.onProgress(event);
      return output;
    });
    await expect(
      runInspectorCancelReuse(
        { model: {}, session: {}, transcribe },
        Float32Array.of(0),
        {},
        reference,
        'fixture',
      ),
    ).rejects.toThrow('not observed and rejected');
    expect(transcribe).toHaveBeenCalledOnce();
  });

  it('does not treat another inference error as successful cancellation', async () => {
    const failure = new Error('native session failure');
    const transcribe = vi.fn(async (_input, options) => {
      options.onProgress(progress);
      throw failure;
    });
    await expect(
      runInspectorCancelReuse(
        { model: {}, session: {}, transcribe },
        Float32Array.of(0),
        {},
        reference,
        'fixture',
      ),
    ).rejects.toBe(failure);
    expect(transcribe).toHaveBeenCalledOnce();
  });

  it('fails on post-cancellation token divergence', async () => {
    let calls = 0;
    const transcribe = vi.fn(async (_input, options) => {
      if (++calls === 1) {
        options.onProgress(progress);
        throw new PipelineAbortedError('decode');
      }
      return { ...output, native: { debug: { ...output.native.debug, tokenIds: [2] } } };
    });
    await expect(
      runInspectorCancelReuse(
        { model: {}, session: {}, transcribe },
        Float32Array.of(0),
        {},
        reference,
        'fixture',
      ),
    ).rejects.toThrow('differs from the oracle');
    expect(transcribe).toHaveBeenCalledTimes(2);
  });
});
