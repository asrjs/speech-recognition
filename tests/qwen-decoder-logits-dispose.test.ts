import { describe, expect, it } from 'vitest';

import { PcmAudioBuffer } from '../src/audio/index.js';
import {
  DEFAULT_QWEN3_ASR_CLASSIFICATION,
  OrtQwen3AsrExecutor,
  Qwen3AsrFeatureProcessor,
  Qwen3AsrTokenizer,
  copyQwenLogits,
  parseOfficialQwen3AsrConfig,
  parseQwen3AsrConfig,
  type QwenOrtModuleLike,
  type QwenOrtSessionLike,
  type QwenOrtTensorLike,
} from '../src/models/qwen-asr/index.js';
import { DEFAULT_QWEN3_ASR_CONFIG } from '../src/models/qwen-asr/config.js';

class TrackingTensor implements QwenOrtTensorLike {
  disposed = 0;
  readonly location = 'cpu';

  constructor(
    readonly type: string,
    readonly data: ArrayBufferView,
    readonly dims: readonly number[],
  ) {}

  dispose(): void {
    this.disposed += 1;
    if (this.data instanceof Float32Array) this.data.fill(0);
    if (this.data instanceof Uint16Array) this.data.fill(0);
  }
}

class Tensor extends TrackingTensor {
  constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
    super(type, data, dims);
  }
}

const ort = {
  env: { wasm: {} },
  Tensor,
  InferenceSession: {
    async create(): Promise<QwenOrtSessionLike> {
      throw new Error('The test must inject sessions.');
    },
  },
} as unknown as QwenOrtModuleLike;

function tokenizer(): Qwen3AsrTokenizer {
  return Qwen3AsrTokenizer.fromJson(
    JSON.stringify({
      model: { vocab: { unused: 0, h: 1, i: 2, '!': 3 }, merges: [] },
      added_tokens: [
        { id: 10, content: '<|audio_start|>', special: true },
        { id: 11, content: '<|audio_end|>', special: true },
        { id: 12, content: '<|audio_pad|>', special: true },
        { id: 13, content: '<|im_start|>', special: true },
        { id: 14, content: '<|im_end|>', special: true },
      ],
    }),
  );
}

function perLayerConfig() {
  return parseQwen3AsrConfig('qwen-dispose', {
    languages: ['Turkish'],
    tokenizer: { kind: 'bpe', vocabSize: 5, eosTokenId: 4, padTokenId: 14 },
    graph: {
      ...DEFAULT_QWEN3_ASR_CONFIG.graph,
      numLayers: 1,
      numKvHeads: 1,
      headDim: 2,
      hiddenSize: 2,
      vocabularySize: 5,
      eosTokenIds: [4],
      audioPadTokenId: 12,
      audioStartTokenId: 10,
      audioEndTokenId: 11,
      imStartTokenId: 13,
      imEndTokenId: 14,
    },
  });
}

function stackedConfig() {
  return parseOfficialQwen3AsrConfig('qwen-stacked-dispose', {
    languages: ['Turkish'],
    tokenizer: { kind: 'bpe', vocabSize: 5, eosTokenId: 4, padTokenId: 14 },
    graph: {
      numLayers: 1,
      numKvHeads: 1,
      headDim: 2,
      hiddenSize: 2,
      vocabularySize: 5,
      eosTokenIds: [4],
      audioPadTokenId: 12,
      audioStartTokenId: 10,
      audioEndTokenId: 11,
      imStartTokenId: 13,
      imEndTokenId: 14,
    },
  });
}

const transcribeContext = {
  modelId: 'qwen-dispose',
  classification: DEFAULT_QWEN3_ASR_CLASSIFICATION,
};

describe('Qwen decoder logits copy-then-dispose', () => {
  it('copies logits into an owned buffer before the Ort tensor is disposed', async () => {
    const logits = new TrackingTensor('float32', new Float32Array([0, 10, 1]), [1, 1, 3]);
    const copied = await copyQwenLogits(logits);
    logits.dispose();
    expect(copied[1]).toBe(10);
    expect((logits.data as Float32Array)[1]).toBe(0);
    expect(copied).not.toBe(logits.data);
  });

  it('disposes per-layer prefill/step logits and replaced KV while keeping next-step cache until replaced', async () => {
    const config = perLayerConfig();
    const logits: TrackingTensor[] = [];
    const presents: TrackingTensor[] = [];
    const extras: TrackingTensor[] = [];
    let decoderCall = 0;
    const encoderSession: QwenOrtSessionLike = {
      async run() {
        return {
          audio_embeddings: new TrackingTensor('float16', new Uint16Array([0x3c00, 0x3c00, 0, 0]), [1, 2, 2]),
          audio_token_mask: new TrackingTensor('bool', new Uint8Array([1, 0]), [1, 2]),
        };
      },
    };
    const decoderSession: QwenOrtSessionLike = {
      async run(feeds) {
        const sequence = Number((feeds.input_ids as TrackingTensor).dims[1]);
        const nextToken = decoderCall === 0 ? 1 : decoderCall === 1 ? 2 : decoderCall === 2 ? 3 : 4;
        const data = new Float32Array(sequence * 5);
        data[(sequence - 1) * 5 + nextToken] = 10;
        const logit = new TrackingTensor('float32', data, [1, sequence, 5]);
        const extra = new TrackingTensor('float32', new Float32Array(1), [1]);
        const totalSequence = Number((feeds.attention_mask as TrackingTensor).dims[3]);
        const key = new TrackingTensor('float16', new Uint16Array(totalSequence * 2), [1, 1, totalSequence, 2]);
        const value = new TrackingTensor('float16', new Uint16Array(totalSequence * 2), [1, 1, totalSequence, 2]);
        logits.push(logit);
        extras.push(extra);
        presents.push(key, value);
        decoderCall += 1;
        return { logits: logit, extra, 'present.0.key': key, 'present.0.value': value };
      },
    };
    const executor = new OrtQwen3AsrExecutor(
      'qwen-per-layer-dispose',
      config,
      'wasm',
      {
        source: {
          kind: 'direct',
          artifacts: { encoderUrl: 'encoder', decoderUrl: 'decoder', tokenizerUrl: 'tokenizer' },
        },
      },
      {
        ort,
        tokenizer: tokenizer(),
        featureProcessor: new Qwen3AsrFeatureProcessor(config),
        encoderSession,
        decoderSession,
      },
    );
    const result = await executor.transcribe(
      PcmAudioBuffer.fromMono(new Float32Array(8000), 16000),
      { language: 'tr', maxNewTokens: 4 },
      { ...transcribeContext, config },
    );
    expect(result.utteranceText).toBe('hi!');
    expect(logits.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(extras.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(presents.every((tensor) => tensor.disposed === 1)).toBe(true);
    await executor.dispose();
  });

  it('disposes per-layer logits and leftover present KV when a step throws', async () => {
    const config = perLayerConfig();
    const logits: TrackingTensor[] = [];
    const presents: TrackingTensor[] = [];
    let decoderCall = 0;
    const executor = new OrtQwen3AsrExecutor(
      'qwen-per-layer-throw',
      config,
      'wasm',
      {
        source: {
          kind: 'direct',
          artifacts: { encoderUrl: 'encoder', decoderUrl: 'decoder', tokenizerUrl: 'tokenizer' },
        },
      },
      {
        ort,
        tokenizer: tokenizer(),
        featureProcessor: new Qwen3AsrFeatureProcessor(config),
        encoderSession: {
          async run() {
            return {
              audio_embeddings: new TrackingTensor('float16', new Uint16Array([0x3c00, 0x3c00, 0, 0]), [1, 2, 2]),
              audio_token_mask: new TrackingTensor('bool', new Uint8Array([1, 0]), [1, 2]),
            };
          },
        },
        decoderSession: {
          async run(feeds) {
            const sequence = Number((feeds.input_ids as TrackingTensor).dims[1]);
            const data = new Float32Array(sequence * 5);
            data[(sequence - 1) * 5 + 1] = 10;
            const logit = new TrackingTensor('float32', data, [1, sequence, 5]);
            logits.push(logit);
            if (decoderCall === 0) {
              const totalSequence = Number((feeds.attention_mask as TrackingTensor).dims[3]);
              const key = new TrackingTensor('float16', new Uint16Array(totalSequence * 2), [1, 1, totalSequence, 2]);
              const value = new TrackingTensor('float16', new Uint16Array(totalSequence * 2), [1, 1, totalSequence, 2]);
              presents.push(key, value);
              decoderCall += 1;
              return { logits: logit, 'present.0.key': key, 'present.0.value': value };
            }
            decoderCall += 1;
            return { logits: logit };
          },
        },
      },
    );
    await expect(
      executor.transcribe(
        PcmAudioBuffer.fromMono(new Float32Array(8000), 16000),
        { language: 'tr', maxNewTokens: 4 },
        { ...transcribeContext, config },
      ),
    ).rejects.toThrow(/missing present\.0\.key\/value/);
    expect(logits.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(presents.every((tensor) => tensor.disposed === 1)).toBe(true);
    await executor.dispose();
  });

  it('copies official stacked prefill/step logits then disposes replaced stacked KV', async () => {
    const config = stackedConfig();
    expect(config.graph.kvLayout).toBe('stacked');
    const logits: TrackingTensor[] = [];
    const presents: TrackingTensor[] = [];
    const extras: TrackingTensor[] = [];
    let prefillCalls = 0;
    let stepCalls = 0;
    const executor = new OrtQwen3AsrExecutor(
      'qwen-stacked-dispose',
      config,
      'wasm',
      {
        source: {
          kind: 'direct',
          artifacts: {
            encoderUrl: 'encoder',
            decoderUrl: 'prefill',
            tokenizerUrl: 'tokenizer',
          },
        },
      },
      {
        ort,
        tokenizer: tokenizer(),
        featureProcessor: new Qwen3AsrFeatureProcessor(config),
        encoderSession: {
          async run() {
            return {
              audio_embeddings: new TrackingTensor('float32', new Float32Array(16), [8, 2]),
            };
          },
        },
        decoderSession: {
          async run() {
            prefillCalls += 1;
            const data = new Float32Array(5);
            data[1] = 10;
            const logit = new TrackingTensor('float32', data, [1, 1, 5]);
            const extra = new TrackingTensor('float32', new Float32Array(1), [1]);
            const keys = new TrackingTensor('float16', new Uint16Array(4), [1, 1, 2, 2]);
            const values = new TrackingTensor('float16', new Uint16Array(4), [1, 1, 2, 2]);
            logits.push(logit);
            extras.push(extra);
            presents.push(keys, values);
            return { logits: logit, extra, present_keys: keys, present_values: values };
          },
        },
        decoderStepSession: {
          async run() {
            stepCalls += 1;
            const nextToken = stepCalls === 1 ? 2 : stepCalls === 2 ? 3 : 4;
            const data = new Float32Array(5);
            data[nextToken] = 10;
            const logit = new TrackingTensor('float32', data, [1, 1, 5]);
            const extra = new TrackingTensor('float32', new Float32Array(1), [1]);
            const keys = new TrackingTensor('float16', new Uint16Array(4), [1, 1, 2, 2]);
            const values = new TrackingTensor('float16', new Uint16Array(4), [1, 1, 2, 2]);
            logits.push(logit);
            extras.push(extra);
            presents.push(keys, values);
            return { logits: logit, extra, present_keys: keys, present_values: values };
          },
        },
      },
    );
    const result = await executor.transcribe(
      PcmAudioBuffer.fromMono(new Float32Array(8000), 16000),
      { language: 'tr', maxNewTokens: 4 },
      { ...transcribeContext, modelId: 'qwen-stacked-dispose', config },
    );
    expect(result.utteranceText).toBe('hi!');
    expect(prefillCalls).toBe(1);
    expect(stepCalls).toBe(3);
    expect(logits.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(extras.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(presents.every((tensor) => tensor.disposed === 1)).toBe(true);
    await executor.dispose();
  });

  it('disposes official prefill logits when present KV is missing', async () => {
    const config = stackedConfig();
    const logit = new TrackingTensor('float32', Float32Array.from([0, 10, 0, 0, 0]), [1, 1, 5]);
    const extra = new TrackingTensor('float32', new Float32Array(1), [1]);
    const executor = new OrtQwen3AsrExecutor(
      'qwen-stacked-throw',
      config,
      'wasm',
      {
        source: {
          kind: 'direct',
          artifacts: { encoderUrl: 'encoder', decoderUrl: 'prefill', tokenizerUrl: 'tokenizer' },
        },
      },
      {
        ort,
        tokenizer: tokenizer(),
        featureProcessor: new Qwen3AsrFeatureProcessor(config),
        encoderSession: {
          async run() {
            return { audio_embeddings: new TrackingTensor('float32', new Float32Array(16), [8, 2]) };
          },
        },
        decoderSession: {
          async run() {
            return { logits: logit, extra };
          },
        },
        decoderStepSession: {
          async run() {
            throw new Error('step should not run');
          },
        },
      },
    );
    await expect(
      executor.transcribe(
        PcmAudioBuffer.fromMono(new Float32Array(8000), 16000),
        { language: 'tr', maxNewTokens: 4 },
        { ...transcribeContext, modelId: 'qwen-stacked-throw', config },
      ),
    ).rejects.toThrow(/present_keys\/present_values/);
    expect(logit.disposed).toBe(1);
    expect(extra.disposed).toBe(1);
    await executor.dispose();
  });
});
