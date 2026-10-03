import { describe, expect, it, vi } from 'vitest';
import { PcmAudioBuffer } from '../src/audio/index.js';
import { createSpeechRuntime } from '@asrjs/speech-recognition';
import type { ExecutionBackend } from '../src/types/index.js';
import {
  NemoTdtSpeechModel,
  createNemoTdtModelFamily,
  type NemoTdtExecutor,
  type NemoTdtModelConfig,
  type NemoTdtNativeTranscript,
} from '../src/models/nemo-tdt/index.js';

function createBackend(id = 'wasm'): ExecutionBackend {
  return {
    id,
    displayName: id.toUpperCase(),
    async probeCapabilities() {
      return {
        id,
        displayName: id.toUpperCase(),
        available: true,
        priority: 1,
        environments: ['browser', 'node'],
        acceleration: ['cpu'],
        supportedPrecisions: ['fp32', 'int8'],
        supportsFp16: false,
        supportsInt8: true,
        supportsSharedArrayBuffer: true,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        experimental: false,
        notes: [],
      };
    },
    async createExecutionContext() {
      return {
        backendId: id,
        capabilities: await this.probeCapabilities(),
        dispose() {
          return undefined;
        },
      };
    },
  };
}

function createConfig(): NemoTdtModelConfig {
  return {
    ecosystem: 'nemo',
    architecture: 'nemo-tdt',
    decoderArchitecture: 'tdt',
    encoderArchitecture: 'fastconformer',
    sampleRate: 16000,
    frameLengthMs: 25,
    frameStepMs: 10,
    melBins: 80,
    subsamplingFactor: 8,
    blankTokenId: 1024,
    vocabularySize: 1025,
    maxSymbolsPerStep: 4,
    languages: ['en'],
    tokenizer: {
      kind: 'sentencepiece',
    },
  };
}

function createTranscript(): NemoTdtNativeTranscript {
  return {
    utteranceText: 'test',
    isFinal: true,
    words: [],
    tokens: [],
    metrics: { encoderBackend: 'webgpu', decoderBackend: 'wasm', jointBackend: 'wasm' },
  };
}

describe('NemoTdtSpeechModel session lifecycle', () => {
  it('drains every call after inference failure and retains cleanup failure', async () => {
    const finishes: Array<() => void> = [];
    const inferenceError = new Error('First inference failed');
    const cleanupError = new Error('Executor disposal failed');
    const dispose = vi.fn(() => {
      throw cleanupError;
    });
    const executor: NemoTdtExecutor = {
      async transcribe() {
        const index = finishes.length;
        await new Promise<void>((resolve) => {
          finishes.push(resolve);
        });
        if (index === 0) throw inferenceError;
        return createTranscript();
      },
      dispose,
    };
    const model = new NemoTdtSpeechModel(
      createBackend(),
      'nemo-tdt',
      'parakeet-tdt-0.6b-v3',
      { ecosystem: 'nemo', encoder: 'fastconformer', decoder: 'tdt', task: 'asr' },
      createConfig(),
      'parakeet',
      undefined,
      { executor },
      () => 'lifecycle fixture',
    );
    const session = await model.createSession();
    const audio = PcmAudioBuffer.fromMono(new Float32Array(1600), 16000);
    const first = session.transcribe(audio).catch((error: unknown) => error);
    const second = session.transcribe(audio);
    const disposal = session.dispose().catch((error: unknown) => error);
    try {
      expect(finishes).toHaveLength(2);
      finishes[0]!();
      expect(await first).toBe(inferenceError);
      expect(dispose).not.toHaveBeenCalled();
      finishes[1]!();
      expect((await second).text).toBe('test');
      expect(await disposal).toBe(cleanupError);
      await expect(session.dispose()).rejects.toBe(cleanupError);
      await model.dispose();
      expect(dispose).toHaveBeenCalledOnce();
    } finally {
      finishes.forEach((finish) => finish());
      await Promise.allSettled([first, second, disposal, model.dispose()]);
    }
  });

  it.each([
    ['session', false],
    ['session', true],
    ['model', false],
    ['model', true],
    ['runtime', false],
    ['runtime', true],
  ] as const)(
    'drains active public transcription before %s disposal (inference fails: %s)',
    async (owner, fails) => {
      let finish = () => {};
      const gate = new Promise<void>((resolve) => {
        finish = resolve;
      });
      const dispose = vi.fn();
      const error = new Error('Inference failed');
      const transcribe = vi.fn(async () => {
        await gate;
        if (fails) throw error;
        return createTranscript();
      });
      const runtime = createSpeechRuntime({
        backends: [createBackend()],
        modelFamilies: [
          createNemoTdtModelFamily({ dependencies: { executor: { transcribe, dispose } } }),
        ],
      });
      const model = await runtime.loadModel({
        family: 'nemo-tdt',
        modelId: 'parakeet-tdt-0.6b-v3',
      });
      const session = await model.createSession();
      let startedDisposal = () => {};
      const enteredDisposal = new Promise<void>((resolve) => {
        startedDisposal = resolve;
      });
      const originalDispose = session.dispose.bind(session);
      vi.spyOn(session, 'dispose').mockImplementation(() => {
        const pending = originalDispose();
        startedDisposal();
        return pending;
      });
      const audio = PcmAudioBuffer.fromMono(new Float32Array(1600), 16000);
      const processing = session.transcribe(audio);
      const observed = processing.then(
        (value) => ({ value }),
        (failure) => ({ failure }),
      );
      const disposal =
        owner === 'session'
          ? session.dispose()
          : owner === 'model'
            ? model.dispose()
            : runtime.dispose();
      try {
        // Runtime/model ownership closes sessions asynchronously: observe the
        // actual disposal entry instead of assuming a fixed microtask count.
        await enteredDisposal;
        expect(dispose).not.toHaveBeenCalled();
        await expect(session.transcribe(audio)).rejects.toThrow(/disposed/);
        expect(transcribe).toHaveBeenCalledOnce();
        finish();
        const result = await observed;
        if (fails) expect(result).toEqual({ failure: error });
        else expect(result).toMatchObject({ value: { text: 'test' } });
        await disposal;
        expect(dispose).toHaveBeenCalledOnce();
        await session.dispose();
        await model.dispose();
        await runtime.dispose();
        expect(dispose).toHaveBeenCalledOnce();
      } finally {
        finish();
        await Promise.allSettled([observed, disposal]);
        await runtime.dispose();
      }
    },
  );

  it('waits for executor readiness before resolving createSession()', async () => {
    let releaseReady: (() => void) | undefined;
    const readyGate = new Promise<void>((resolve) => {
      releaseReady = resolve;
    });

    const executor: NemoTdtExecutor = {
      async ready() {
        await readyGate;
      },
      async transcribe() {
        return createTranscript();
      },
      dispose() {
        return undefined;
      },
    };

    const model = new NemoTdtSpeechModel(
      createBackend(),
      'nemo-tdt',
      'parakeet-tdt-0.6b-v2',
      {
        ecosystem: 'nemo',
        encoder: 'fastconformer',
        decoder: 'tdt',
        task: 'asr',
      },
      createConfig(),
      'parakeet',
      undefined,
      { executor },
      () => 'test model',
    );

    let resolved = false;
    const sessionPromise = model.createSession().then((session) => {
      resolved = true;
      return session;
    });

    await Promise.resolve();
    await Promise.resolve();
    expect(resolved).toBe(false);

    releaseReady?.();
    const session = await sessionPromise;
    expect(resolved).toBe(true);

    const transcript = await session.transcribe(
      PcmAudioBuffer.fromMono(new Float32Array(1600), 16000),
    );
    expect(transcript.meta.metrics).toMatchObject({
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
      jointBackend: 'wasm',
    });

    await session.dispose();
    await model.dispose();
  });
});
