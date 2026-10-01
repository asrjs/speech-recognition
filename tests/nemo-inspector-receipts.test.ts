import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import {
  compareNemoTdtOutput,
  describeLocalArtifact,
  loadNemoTdtOracle,
  runInspectorWithCleanup,
  validateOracleAudioPreparation,
} from '../tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs';

function reference() {
  return {
    tokens: [1, 2],
    token_frames: [0, 4],
    token_duration: [4, 3],
    transcript: 'Exact text.',
  };
}

function candidate() {
  return {
    output: { text: 'Exact text.' },
    native: { debug: { tokenIds: [1, 2], frameIndices: [0, 4], tdtSteps: [4, 3] } },
  };
}

describe('NeMo inspector receipts (not model parity)', () => {
  it('hashes exact file bytes without accepting a directory or empty artifact', async (context) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'asr-nemo-receipt-'));
    context.onTestFinished(() => fs.rmSync(directory, { recursive: true, force: true }));
    const file = path.join(directory, 'encoder.onnx.data');
    fs.writeFileSync(file, 'abc');
    expect(await describeLocalArtifact(file)).toEqual({
      path: file,
      bytes: 3,
      sha256: 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    });
    fs.writeFileSync(file, 'abcd');
    expect((await describeLocalArtifact(file)).sha256).not.toBe(
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    );
    await expect(describeLocalArtifact(directory)).rejects.toThrow('non-empty regular file');
    fs.writeFileSync(file, '');
    await expect(describeLocalArtifact(file)).rejects.toThrow('non-empty regular file');
  });

  it('returns a receipt only after both disposals have completed', async () => {
    const events: string[] = [];
    let finishModel!: () => void;
    const modelGate = new Promise<void>((resolve) => {
      finishModel = resolve;
    });
    const receipt = runInspectorWithCleanup(
      async () => {
        events.push('inference');
        return { output: 'complete' };
      },
      async () => {
        events.push('model-start');
        await modelGate;
        events.push('model-end');
      },
      async () => {
        events.push('runtime');
      },
    );
    const emit = vi.fn();
    const emitted = receipt.then(emit);
    await vi.waitFor(() => expect(events).toEqual(['inference', 'model-start']));
    expect(emit).not.toHaveBeenCalled();
    finishModel();
    await emitted;
    expect(events).toEqual(['inference', 'model-start', 'model-end', 'runtime']);
    expect(emit.mock.calls[0]?.[0].cleanup.status).toBe('PASS');
  });

  it('still disposes the runtime and does not emit success when model cleanup fails', async () => {
    const error = new Error('model release failed');
    const runtimeDispose = vi.fn(async () => {});
    const emit = vi.fn();
    await expect(
      runInspectorWithCleanup(
        async () => candidate(),
        async () => {
          throw error;
        },
        runtimeDispose,
      ).then(emit),
    ).rejects.toBe(error);
    expect(runtimeDispose).toHaveBeenCalledOnce();
    expect(emit).not.toHaveBeenCalled();
  });

  it('retains inference and both cleanup failures rather than hiding one', async () => {
    const errors = [new Error('inference'), new Error('model'), new Error('runtime')];
    const result = runInspectorWithCleanup(
      async () => {
        throw errors[0];
      },
      async () => {
        throw errors[1];
      },
      async () => {
        throw errors[2];
      },
    );
    await expect(result).rejects.toMatchObject({ errors });
  });
});

describe('NeMo inspector full output oracle contract', () => {
  it('requires the same mono sample-rate contract before oracle inference', () => {
    expect(() =>
      validateOracleAudioPreparation({ sampleRate: 16000, numberOfChannels: 1 }, 16000),
    ).not.toThrow();
    expect(() =>
      validateOracleAudioPreparation({ sampleRate: 22050, numberOfChannels: 1 }, 16000),
    ).toThrow('same-rate mono');
    expect(() =>
      validateOracleAudioPreparation({ sampleRate: 16000, numberOfChannels: 2 }, 16000),
    ).toThrow('same-rate mono');
  });

  it('binds an official capture to audio, model and original-weight identity', (context) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'asr-nemo-oracle-'));
    context.onTestFinished(() => fs.rmSync(directory, { recursive: true, force: true }));
    const file = path.join(directory, 'reference.json');
    const expected = {
      sampleId: 'fixture',
      modelId: 'test/model',
      audioSha256: 'a'.repeat(64),
      weightSha256: 'b'.repeat(64),
    };
    const capture = {
      schema_version: 1,
      meta: {
        model: expected.modelId,
        model_revision: 'pinned-revision',
        weight_sha256: expected.weightSha256,
      },
      instrumentation_control: {
        baseline_tokens_match: true,
        baseline_text_match: true,
        baseline_frames_match: true,
        after_tokens_match: true,
        after_text_match: true,
        after_durations_match: true,
      },
      samples: [
        { ...reference(), sample_id: expected.sampleId, audio: { sha256: expected.audioSha256 } },
      ],
    };
    fs.writeFileSync(file, JSON.stringify(capture));
    expect(loadNemoTdtOracle(file, expected).sample).toEqual(capture.samples[0]);
    expect(() => loadNemoTdtOracle(file, { ...expected, weightSha256: 'c'.repeat(64) })).toThrow(
      'weight identity',
    );
    expect(() => loadNemoTdtOracle(file, { ...expected, audioSha256: 'c'.repeat(64) })).toThrow(
      'audio identity',
    );
    expect(() => loadNemoTdtOracle(file, { ...expected, modelId: 'another/model' })).toThrow(
      'model identity',
    );
    capture.samples.push(capture.samples[0]!);
    fs.writeFileSync(file, JSON.stringify(capture));
    expect(() => loadNemoTdtOracle(file, expected)).toThrow('Exactly one');
    capture.samples.pop();
    capture.instrumentation_control.after_tokens_match = false;
    fs.writeFileSync(file, JSON.stringify(capture));
    expect(() => loadNemoTdtOracle(file, expected)).toThrow('instrumentation control');
  });

  it('compares all tokens, frames, durations and exact text', () => {
    expect(compareNemoTdtOutput(reference(), candidate(), 'fixture').pass).toBe(true);
  });

  it.each(['tokens', 'token_frames', 'token_duration', 'transcript'] as const)(
    'reports a %s mismatch without hiding it in a short token preview',
    (field) => {
      const expected = reference();
      if (field === 'transcript') expected.transcript = 'Different text.';
      else expected[field][1] = 99;
      const result = compareNemoTdtOutput(expected, candidate(), 'fixture');
      expect(result.pass).toBe(false);
      expect(result.firstDivergentField).toBe(field);
    },
  );

  it('detects a token mismatch after the first 30 tokens', () => {
    const expected = reference();
    const actual = candidate();
    expected.tokens = actual.native.debug.tokenIds = Array.from(
      { length: 40 },
      (_, index) => index,
    );
    expected.token_frames = actual.native.debug.frameIndices = [...expected.tokens];
    expected.token_duration = actual.native.debug.tdtSteps = new Array(40).fill(1);
    actual.native.debug.tokenIds = [...expected.tokens];
    actual.native.debug.tokenIds[39] = 99;
    const result = compareNemoTdtOutput(expected, actual, 'fixture');
    expect(result.pass).toBe(false);
    expect(result.checks.tokens.firstMismatch).toEqual({ index: 39, reference: 39, candidate: 99 });
  });

  it('rejects absent metadata even when it is missing on both sides', () => {
    expect(() => compareNemoTdtOutput({ tokens: [] }, { output: { text: '' } }, 'silence')).toThrow(
      'requires explicit token_frames',
    );
  });

  it('accepts empty sequences only when all required silence output fields are explicit', () => {
    expect(
      compareNemoTdtOutput(
        { tokens: [], token_frames: [], token_duration: [], transcript: '' },
        {
          output: { text: '' },
          native: { debug: { tokenIds: [], frameIndices: [], tdtSteps: [] } },
        },
        'silence',
      ).pass,
    ).toBe(true);
  });
});
