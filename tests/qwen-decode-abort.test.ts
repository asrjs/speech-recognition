import { describe, expect, it } from 'vitest';

import { PcmAudioBuffer } from '../src/audio/index.js';
import {
  DEFAULT_QWEN3_ASR_CLASSIFICATION,
  OrtQwen3AsrExecutor,
  Qwen3AsrFeatureProcessor,
  Qwen3AsrTokenizer,
  parseOfficialQwen3AsrConfig,
  parseQwen3AsrConfig,
  type QwenOrtModuleLike,
  type QwenOrtSessionLike,
  type QwenOrtTensorLike,
} from '../src/models/qwen-asr/index.js';
import { DEFAULT_QWEN3_ASR_CONFIG } from '../src/models/qwen-asr/config.js';
import { PipelineAbortedError } from '../src/pipeline/composition.js';

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
  return parseQwen3AsrConfig('qwen-decode-abort', {
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
  return parseOfficialQwen3AsrConfig('qwen-stacked-abort', {
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

const pcm = () => PcmAudioBuffer.fromMono(new Float32Array(8000), 16000);

describe('Qwen in-flight decode abort', () => {
  it('stops official stacked steps on abort, disposes KV, and can decode again', async () => {
    const config = stackedConfig();
    const logits: TrackingTensor[] = [];
    const presents: TrackingTensor[] = [];
    const extras: TrackingTensor[] = [];
    let prefillCalls = 0;
    let stepCalls = 0;
    const signal = { aborted: false };
    const executor = new OrtQwen3AsrExecutor(
      'qwen-stacked-abort',
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
            return { audio_embeddings: new TrackingTensor('float32', new Float32Array(16), [8, 2]) };
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
            if (stepCalls === 1) signal.aborted = true;
            return { logits: logit, extra, present_keys: keys, present_values: values };
          },
        },
      },
    );
    const context = {
      modelId: 'qwen-stacked-abort',
      classification: DEFAULT_QWEN3_ASR_CLASSIFICATION,
      config,
    };

    await expect(
      executor.transcribe(pcm(), { language: 'tr', maxNewTokens: 8, signal }, context),
    ).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(prefillCalls).toBe(1);
    expect(stepCalls).toBe(1);
    expect(logits.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(extras.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(presents.every((tensor) => tensor.disposed === 1)).toBe(true);

    signal.aborted = false;
    prefillCalls = 0;
    stepCalls = 0;
    const result = await executor.transcribe(pcm(), { language: 'tr', maxNewTokens: 8 }, context);
    expect(result.utteranceText).toBe('hi!');
    expect(prefillCalls).toBe(1);
    expect(stepCalls).toBe(3);
    await executor.dispose();
  });

  it('stops per-layer decoder steps on abort, disposes KV, and can decode again', async () => {
    const config = perLayerConfig();
    const logits: TrackingTensor[] = [];
    const presents: TrackingTensor[] = [];
    let decoderCall = 0;
    const signal = { aborted: false };
    const executor = new OrtQwen3AsrExecutor(
      'qwen-per-layer-abort',
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
            const nextToken = decoderCall === 0 ? 1 : decoderCall === 1 ? 2 : decoderCall === 2 ? 3 : 4;
            const data = new Float32Array(sequence * 5);
            data[(sequence - 1) * 5 + nextToken] = 10;
            const logit = new TrackingTensor('float32', data, [1, sequence, 5]);
            const totalSequence = Number((feeds.attention_mask as TrackingTensor).dims[3]);
            const key = new TrackingTensor('float16', new Uint16Array(totalSequence * 2), [1, 1, totalSequence, 2]);
            const value = new TrackingTensor('float16', new Uint16Array(totalSequence * 2), [1, 1, totalSequence, 2]);
            logits.push(logit);
            presents.push(key, value);
            decoderCall += 1;
            if (decoderCall === 2) signal.aborted = true;
            return { logits: logit, 'present.0.key': key, 'present.0.value': value };
          },
        },
      },
    );
    const context = {
      modelId: 'qwen-per-layer-abort',
      classification: DEFAULT_QWEN3_ASR_CLASSIFICATION,
      config,
    };

    await expect(
      executor.transcribe(pcm(), { language: 'tr', maxNewTokens: 8, signal }, context),
    ).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(decoderCall).toBe(2);
    expect(logits.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(presents.every((tensor) => tensor.disposed === 1)).toBe(true);

    signal.aborted = false;
    decoderCall = 0;
    const result = await executor.transcribe(pcm(), { language: 'tr', maxNewTokens: 8 }, context);
    expect(result.utteranceText).toBe('hi!');
    expect(decoderCall).toBe(4);
    await executor.dispose();
  });
});
