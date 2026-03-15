import { describe, expect, it } from 'vitest';
import type { ExecutionBackend } from '../src/types/index.js';
import {
  NemoTdtSpeechModel,
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
  };
}

describe('NemoTdtSpeechModel session lifecycle', () => {
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

    await session.dispose();
    await model.dispose();
  });
});
