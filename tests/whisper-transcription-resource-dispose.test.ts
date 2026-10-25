import { describe, expect, it } from 'vitest';
import { PcmAudioBuffer } from '../src/audio/index.js';
import { DEFAULT_WHISPER_CLASSIFICATION } from '../src/models/whisper-seq2seq/config.js';
import {
  WhisperOnnxExecutor,
} from '../src/models/whisper-seq2seq/executor.js';
import {
  parseWhisperGenerationConfig,
  parseWhisperModelConfig,
} from '../src/models/whisper-seq2seq/generation-config.js';
import type { OrtTensorLike } from '../src/models/whisper-seq2seq/ort.js';

class TrackingTensor implements OrtTensorLike {
  disposed = 0;

  constructor(
    readonly type: string,
    readonly data: ArrayBufferView,
    readonly dims: readonly number[],
    readonly location: string | undefined = 'cpu',
  ) {}

  async getData(): Promise<ArrayBufferView> {
    return this.data;
  }

  dispose(): void {
    this.disposed += 1;
  }
}

function createSplitTranscriptionFixture(
  failure: 'none' | 'decoder-init' | 'decoder-step' | 'alignment' = 'none',
  batched = false,
) {
  const created: TrackingTensor[] = [];
  const encoderOutputs: TrackingTensor[] = [];
  const scalarEncoderFeeds: TrackingTensor[] = [];
  let encoderRunCount = 0;

  class Tensor extends TrackingTensor {
    constructor(
      type: string,
      data: ArrayBufferView,
      dims: readonly number[],
      location: string | undefined = 'cpu',
    ) {
      super(type, data, dims, location);
      created.push(this);
    }
  }

  const tokenIds = new Map([
    ['<|endoftext|>', 2],
    ['<|startoftranscript|>', 3],
    ['<|en|>', 4],
    ['<|transcribe|>', 5],
    ['<|notimestamps|>', 6],
    ['<|nospeech|>', 7],
    ['<|0.00|>', 8],
  ]);
  const specialIds = new Set(tokenIds.values());
  const tokenizer = {
    getTokenId: (token: string) => tokenIds.get(token),
    isSpecialTokenId: (id: number) => specialIds.has(id),
    isTimestampTokenId: (id: number) => id === 8,
    timestampTokenIdToSeconds: (id: number) => id === 8 ? 0 : undefined,
    idsToTokens: (ids: readonly number[]) => ids.map((id) => [...tokenIds.entries()].find(([, value]) => value === id)?.[0] ?? 'text'),
    decode: (ids: readonly number[], options: { readonly skipSpecialTokens?: boolean } = {}) => ids
      .filter((id) => !options.skipSpecialTokens || !specialIds.has(id))
      .map((id) => id === 1 ? 'hello' : '')
      .join(''),
    encode: () => [1],
  };

  const encoderSession = {
    inputMetadata: [{ name: 'input_features', type: 'float32' }],
    async run() {
      const output = new Tensor(
        'float32',
        new Float32Array(16).fill(0.25),
        [1, 4, 4],
        failure === 'alignment' ? 'gpu-buffer' : 'cpu',
      );
      encoderOutputs.push(output);
      encoderRunCount += 1;
      return { last_hidden_state: output };
    },
  };

  const makeLogits = (tokenId: number): TrackingTensor => {
    const logits = new Float32Array(10);
    logits[tokenId] = 10;
    return new Tensor('float32', logits, [1, 1, logits.length]);
  };

  const decoderInitSession = {
    inputNames: ['input_ids', 'encoder_hidden_states'],
    inputMetadata: [{ name: 'encoder_hidden_states', type: 'float32' }],
    async run() {
      if (failure === 'decoder-init') throw new Error('decoder init failed');
      const initLogits = makeLogits(1);
      if (batched) (initLogits.data as Float32Array)[0] = 9;
      return {
        logits: initLogits,
        'present.0.decoder.key': new Tensor('float32', new Float32Array(16), [1, 1, 4, 4]),
        'present.0.decoder.value': new Tensor('float32', new Float32Array(16), [1, 1, 4, 4]),
        'present.0.encoder.key': new Tensor('float32', new Float32Array(16), [1, 1, 4, 4]),
        'present.0.encoder.value': new Tensor('float32', new Float32Array(16), [1, 1, 4, 4]),
      };
    },
  };

  const decoderStepSession = {
    inputNames: [
      'input_ids',
      'past_key_values.0.decoder.key',
      'past_key_values.0.decoder.value',
      'past_key_values.0.encoder.key',
      'past_key_values.0.encoder.value',
    ],
    inputMetadata: batched ? [{ name: 'input_ids', shape: [2, 'tokens'] }] : undefined,
    async run(feeds: Record<string, TrackingTensor>) {
      if (failure === 'decoder-step') throw new Error('decoder step failed');
      scalarEncoderFeeds.push(
        feeds['past_key_values.0.encoder.key']!,
        feeds['past_key_values.0.encoder.value']!,
      );
      const batchSize = feeds.input_ids?.dims[0] ?? 1;
      const nextLength = (feeds['past_key_values.0.decoder.key']?.dims[2] ?? 0) + 1;
      const logits = new Float32Array(batchSize * 10);
      for (let index = 0; index < batchSize; index++) logits[index * 10 + 2] = 10;
      return {
        logits: new Tensor('float32', logits, [batchSize, 1, 10]),
        'present.0.decoder.key': new Tensor('float32', new Float32Array(batchSize * nextLength * 4), [batchSize, 1, nextLength, 4]),
        'present.0.decoder.value': new Tensor('float32', new Float32Array(batchSize * nextLength * 4), [batchSize, 1, nextLength, 4]),
      };
    },
  };

  const decoderAlignSession = {
    inputNames: ['input_ids', 'encoder_hidden_states'],
    inputMetadata: [{ name: 'encoder_hidden_states', type: 'float16' }],
    async run() {
      if (failure === 'alignment') throw new Error('alignment failed');
      return { alignment: new Tensor('float32', new Float32Array(4), [1, 1, 1, 4]) };
    },
  };

  const loaded = {
    ort: { Tensor },
    tokenizer,
    encoderSession,
    decoderInitSession,
    decoderStepSession,
    decoderAlignSession,
    decoderAlignUrl: 'mock://decoder_align.onnx',
    decoderAlignCausalSelfAttention: true,
    decoderAlignAttentionValues: 'logits',
    decoderAlignAttentionLayout: 'selected_heads',
    decoderBackendForOrt: 'wasm',
    modelConfig: parseWhisperModelConfig({
      decoder_layers: 1,
      decoder_attention_heads: 1,
      d_model: 4,
      num_mel_bins: 80,
    }),
    generationConfig: parseWhisperGenerationConfig({}),
    warnings: [],
    isSplitGraph: true,
    sessionCreateMs: 0,
  };

  const executor = new WhisperOnnxExecutor(
    'whisper-resource-dispose',
    DEFAULT_WHISPER_CLASSIFICATION,
    {
      maxSourcePositions: 4,
      maxTargetPositions: 2,
      languages: ['en'],
    },
    'wasm',
    undefined,
  );
  (executor as unknown as { getLoadedState: () => Promise<unknown> }).getLoadedState = async () => loaded;

  return {
    executor,
    created,
    encoderOutputs,
    scalarEncoderFeeds,
    get encoderRunCount() { return encoderRunCount; },
  };
}

async function transcribeFixture(
  fixture: ReturnType<typeof createSplitTranscriptionFixture>,
  alignment = false,
  batched = false,
) {
  const audio = PcmAudioBuffer.fromMono(new Float32Array(320), 16000);
  return (fixture.executor as unknown as {
    transcribeWithSplitGraph(
      input: unknown,
      options: unknown,
      context: unknown,
    ): Promise<{ readonly utteranceText: string; readonly warnings: readonly { readonly message: string }[] }>;
  }).transcribeWithSplitGraph(
    audio,
    {
      language: 'en',
      noTimestamps: true,
      maxNewTokens: 2,
      ...(batched ? { numBeams: 2, experimentalBatchedBeam: true } : {}),
      returnWords: alignment,
    },
    { modelId: 'whisper-resource-dispose', config: {} },
  );
}

describe('Whisper split transcription resource ownership', () => {
  it('releases encoder outputs and scalar encoder-KV caches after repeated successful transcriptions', async () => {
    const fixture = createSplitTranscriptionFixture();

    const first = await transcribeFixture(fixture);
    const second = await transcribeFixture(fixture);

    expect(first.utteranceText).toBe('hello');
    expect(second.utteranceText).toBe('hello');
    expect(fixture.encoderRunCount).toBe(2);
    expect(fixture.encoderOutputs.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(fixture.scalarEncoderFeeds.length).toBeGreaterThan(0);
    expect(fixture.scalarEncoderFeeds.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(fixture.created.every((tensor) => tensor.disposed === 1)).toBe(true);
  });

  it.each(['decoder-init', 'decoder-step'] as const)(
    'releases owned tensors when %s throws',
    async (failure) => {
      const fixture = createSplitTranscriptionFixture(failure);
      await expect(transcribeFixture(fixture)).rejects.toThrow(/failed/);

      expect(fixture.encoderOutputs.every((tensor) => tensor.disposed === 1)).toBe(true);
      expect(fixture.scalarEncoderFeeds.every((tensor) => tensor.disposed === 1)).toBe(true);
      expect(fixture.created.every((tensor) => tensor.disposed === 1)).toBe(true);
    },
  );

  it('releases encoder materialization, recast, and alignment feeds after alignment throws', async () => {
    const fixture = createSplitTranscriptionFixture('alignment');
    const result = await transcribeFixture(fixture, true);

    expect(result.utteranceText).toBe('hello');
    expect(result.warnings.some((warning) => warning.message.includes('alignment failed'))).toBe(true);
    expect(fixture.created.some((tensor) => tensor.type === 'float16')).toBe(true);
    expect(fixture.created.filter((tensor) => tensor.dims.join(',') === '1,4,4').length).toBeGreaterThan(2);
    expect(fixture.encoderOutputs.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(fixture.created.every((tensor) => tensor.disposed === 1)).toBe(true);
  });

  it('releases batched encoder-KV caches after a successful beam transcription', async () => {
    const fixture = createSplitTranscriptionFixture('none', true);
    const result = await transcribeFixture(fixture, false, true);

    expect(result.utteranceText.length).toBeGreaterThan(0);
    expect(fixture.scalarEncoderFeeds.length).toBeGreaterThan(0);
    expect(fixture.scalarEncoderFeeds.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(fixture.created.every((tensor) => tensor.disposed === 1)).toBe(true);
  });
});
