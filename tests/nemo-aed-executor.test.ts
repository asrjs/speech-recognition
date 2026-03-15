import { describe, expect, it } from 'vitest';

import {
  DEFAULT_NEMO_AED_CLASSIFICATION,
  parseNemoAedConfig,
} from '../src/models/nemo-aed/config.js';
import { OrtNemoAedExecutor } from '../src/models/nemo-aed/executor.js';
import { CanaryTokenizer, type CanaryTokenizerPayload } from '../src/models/nemo-aed/tokenizer.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/nemo-aed/ort.js';
import type { NemoAedArtifactSource } from '../src/models/nemo-aed/types.js';
import type { AudioBufferLike, TranscriptionProgressEvent } from '../src/types/index.js';

class MockTensor<TData extends ArrayBufferView = ArrayBufferView> implements OrtTensorLike<TData> {
  disposed = false;

  constructor(
    readonly data: TData,
    readonly dims: readonly number[],
  ) {}

  dispose(): void {
    this.disposed = true;
  }
}

class MockEncoderSession implements OrtSessionLike {
  async run(): Promise<Record<string, OrtTensorLike>> {
    return {
      encoder_states: new MockTensor(new Float32Array(8), [1, 2, 4]),
      encoded_length: new MockTensor(BigInt64Array.from([2n]), [1]),
      encoder_mask: new MockTensor(new Float32Array([1, 1]), [1, 2]),
    };
  }
}

class MockDecoderSession implements OrtSessionLike {
  readonly inputHistory: number[][] = [];
  private callIndex = 0;

  constructor(private readonly stepTokenIds: readonly number[]) {}

  async run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>> {
    const inputIds = feeds.input_ids as OrtTensorLike<BigInt64Array>;
    this.inputHistory.push(Array.from(inputIds.data, (value) => Number(value)));

    const nextId = this.stepTokenIds[this.callIndex];
    this.callIndex += 1;
    if (nextId === undefined) {
      throw new Error(`Unexpected decoder invocation #${this.callIndex}.`);
    }

    const logits = new Float32Array(84);
    logits[nextId] = 12;
    return {
      next_logits: new MockTensor(logits, [1, 84]),
    };
  }
}

function createMockOrt(): OrtModuleLike {
  class RuntimeTensor<TData extends ArrayBufferView> extends MockTensor<TData> {
    readonly type: 'float32' | 'int32' | 'int64';

    constructor(type: 'float32' | 'int32' | 'int64', data: TData, dims: readonly number[]) {
      super(data, dims);
      this.type = type;
    }
  }

  return {
    env: {
      wasm: {},
    },
    Tensor: RuntimeTensor,
    InferenceSession: {
      async create(): Promise<OrtSessionLike> {
        throw new Error('MockOrt.InferenceSession.create should not be called in executor tests.');
      },
    },
  };
}

function createAudio(sampleRate = 16000, frames = 1600): AudioBufferLike {
  const mono = new Float32Array(frames);
  return {
    sampleRate,
    numberOfChannels: 1,
    numberOfFrames: frames,
    durationSeconds: frames / sampleRate,
    channels: [mono],
  };
}

function createTokenizerPayload(): CanaryTokenizerPayload {
  const specialPieces = Array.from({ length: 80 }, (_, index) => `<special:${index}>`);
  return {
    kind: 'canary-aggregate-tokenizer',
    version: 1,
    prompt_format: 'canary2',
    vocab_size: 84,
    langs: ['spl_tokens', 'en'],
    language_codes: ['en'],
    bos_id: 4,
    eos_id: 3,
    pad_id: 2,
    special_tokens: {
      '<pad>': 2,
      '<|endoftext|>': 3,
      '<|startoftranscript|>': 4,
      '<|pnc|>': 5,
      '<|nopnc|>': 6,
      '<|startofcontext|>': 7,
      '<|itn|>': 8,
      '<|noitn|>': 9,
      '<|timestamp|>': 10,
      '<|notimestamp|>': 11,
      '<|diarize|>': 12,
      '<|nodiarize|>': 13,
      '<|emo:undefined|>': 16,
      '<|en|>': 62,
    },
    subtokenizers: {
      spl_tokens: {
        offset: 0,
        size: 80,
        pieces: specialPieces,
      },
      en: {
        offset: 80,
        size: 4,
        pieces: ['\u2581Hello', ',', '\u2581world', '!'],
      },
    },
  };
}

function createExecutorHarness(source?: NemoAedArtifactSource) {
  const config = parseNemoAedConfig('test-canary', {
    vocabularySize: 84,
    languages: ['en'],
    maxTargetPositions: 16,
  });
  const tokenizer = CanaryTokenizer.fromPayload(createTokenizerPayload());
  const decoderSession = new MockDecoderSession([80, 81, 82, 83, 3]);
  const executor = new OrtNemoAedExecutor(
    'test-canary',
    DEFAULT_NEMO_AED_CLASSIFICATION,
    config,
    'wasm',
    source ? { source } : undefined,
  ) as OrtNemoAedExecutor & { loadStatePromise?: Promise<unknown> };

  (executor as typeof executor & { sourceOptions?: unknown }).sourceOptions =
    source ??
    ({
      kind: 'direct',
      artifacts: {
        encoderUrl: 'encoder',
        decoderUrl: 'decoder',
        tokenizerUrl: 'tokenizer',
        preprocessorUrl: 'nemo128',
      },
    } satisfies NemoAedArtifactSource);
  executor.loadStatePromise = Promise.resolve({
    ort: createMockOrt(),
    tokenizer,
    encoderSession: new MockEncoderSession(),
    decoderSession,
    preprocessorBackend: source?.preprocessorBackend ?? 'onnx',
    preprocessor: {
      async process() {
        return {
          features: new Float32Array(config.melBins * 2),
          frameCount: 2,
          validLength: 2,
        };
      },
    },
    warnings: [],
  });

  return {
    executor,
    decoderSession,
  };
}

describe('nemo-aed executor decode loop', () => {
  it('runs a greedy autoregressive decode with canary prompt ids', async () => {
    const harness = createExecutorHarness();

    const result = await harness.executor.transcribe(
      createAudio(),
      {
        targetLanguage: 'en',
        returnTokenIds: true,
        returnPromptIds: true,
        returnLogProbs: true,
      },
      {} as never,
    );

    expect(result.utteranceText).toBe('Hello, world!');
    expect(result.prompt?.ids).toEqual([7, 4, 16, 62, 62, 5, 9, 11, 13]);
    expect(result.tokens).toEqual([
      expect.objectContaining({ id: 80, text: 'Hello' }),
      expect.objectContaining({ id: 81, text: ',' }),
      expect.objectContaining({ id: 82, text: 'world' }),
      expect.objectContaining({ id: 83, text: '!' }),
    ]);
    expect(result.debug?.tokenIds).toEqual([80, 81, 82, 83, 3]);
    expect(result.debug?.promptIds).toEqual([7, 4, 16, 62, 62, 5, 9, 11, 13]);
    expect(harness.decoderSession.inputHistory[0]).toEqual([7, 4, 16, 62, 62, 5, 9, 11, 13]);
    expect(harness.decoderSession.inputHistory[1]).toEqual([7, 4, 16, 62, 62, 5, 9, 11, 13, 80]);
    expect(result.metrics?.decodeIterations).toBe(5);
    expect(result.metrics?.emittedTokenCount).toBe(4);
  });

  it('emits progress stages and metrics during transcription', async () => {
    const harness = createExecutorHarness();
    const events: TranscriptionProgressEvent[] = [];

    const result = await harness.executor.transcribe(
      createAudio(),
      {
        onProgress(event) {
          events.push(event);
        },
      },
      {} as never,
    );

    expect(result.utteranceText).toBe('Hello, world!');
    expect(events[0]).toEqual(
      expect.objectContaining({
        stage: 'start',
        progress: 0,
        modelId: 'test-canary',
      }),
    );
    expect(events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ stage: 'preprocess', progress: 0.2 }),
        expect.objectContaining({ stage: 'encode', progress: 0.4 }),
        expect.objectContaining({ stage: 'postprocess', progress: 0.95 }),
        expect.objectContaining({ stage: 'complete', progress: 1 }),
      ]),
    );
    expect(events.some((event) => event.stage === 'decode')).toBe(true);
  });

  it('reports requested and effective preprocessor backends when js preprocessing is enabled', async () => {
    const harness = createExecutorHarness({
      kind: 'huggingface',
      repoId: 'nvidia/canary-180m-flash',
      preprocessorBackend: 'js',
    });

    const result = await harness.executor.transcribe(
      createAudio(),
      {
        targetLanguage: 'en',
      },
      {} as never,
    );

    expect(result.metrics?.requestedPreprocessorBackend).toBe('js');
    expect(result.metrics?.preprocessorBackend).toBe('js');
  });
});
