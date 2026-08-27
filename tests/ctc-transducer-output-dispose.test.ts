import { describe, expect, it } from 'vitest';

import { OrtGigaAmCtcExecutor } from '../src/models/gigaam-ctc/executor.js';
import { GigaAmTokenizer } from '../src/models/gigaam-ctc/tokenizer.js';
import { OrtGigaAmRnntExecutor } from '../src/models/gigaam-rnnt/executor.js';
import { GigaAmRnntTokenizer } from '../src/models/gigaam-rnnt/tokenizer.js';
import { DEFAULT_LASR_CTC_CLASSIFICATION, parseLasrCtcConfig } from '../src/models/lasr-ctc/config.js';
import { OrtLasrCtcExecutor } from '../src/models/lasr-ctc/executor.js';
import { MedAsrJsPreprocessor } from '../src/models/lasr-ctc/mel.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/lasr-ctc/ort.js';
import { MedAsrTextTokenizer } from '../src/models/lasr-ctc/tokenizer.js';
import { OrtSenseVoiceExecutor } from '../src/models/sensevoice/executor.js';
import { SenseVoiceTokenizer } from '../src/models/sensevoice/tokenizer.js';
import { DEFAULT_WAV2VEC2_CLASSIFICATION, DEFAULT_WAV2VEC2_CONFIG } from '../src/models/wav2vec2/config.js';
import { OrtWav2Vec2Executor } from '../src/models/wav2vec2/executor.js';
import { Wav2Vec2CharTokenizer } from '../src/models/wav2vec2/tokenizer.js';
import { OrtXAsrExecutor } from '../src/models/x-asr/executor.js';
import { XAsrTokenizer } from '../src/models/x-asr/tokenizer.js';

class TrackingTensor implements OrtTensorLike {
  disposed = 0;

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

const ort: OrtModuleLike = {
  env: { wasm: {} },
  Tensor,
  InferenceSession: { create: async () => ({ async run() { return {}; } }) },
};

function pcm(frames = 16000) {
  return {
    sampleRate: 16000,
    numberOfChannels: 1,
    numberOfFrames: frames,
    durationSeconds: frames / 16000,
    channels: [new Float32Array(frames)],
  };
}

const gigaamCtcConfig = {
  ecosystem: 'gigaam' as const,
  architecture: 'gigaam-ctc' as const,
  processorArchitecture: 'gigaam-fbank' as const,
  encoderArchitecture: 'gigaam-conformer' as const,
  decoderArchitecture: 'ctc' as const,
  sampleRate: 16000,
  rawStride: 4,
  nMels: 64,
  featureHopSeconds: 0.01,
  vocabularySize: 71,
  languages: ['ru', 'en'],
  tokenizer: { kind: 'sentencepiece' as const, blankTokenId: 70 },
  nFft: 320,
  winLength: 320,
  hopLength: 160,
  featureLayout: 'mel-major' as const,
};

const gigaamRnntConfig = {
  ecosystem: 'gigaam' as const,
  architecture: 'gigaam-rnnt' as const,
  processorArchitecture: 'gigaam-fbank' as const,
  encoderArchitecture: 'gigaam-conformer' as const,
  decoderArchitecture: 'rnnt' as const,
  sampleRate: 16000,
  rawStride: 4,
  nMels: 64,
  featureHopSeconds: 0.01,
  vocabularySize: 35,
  languages: ['ru'],
  tokenizer: { kind: 'sentencepiece' as const, blankTokenId: 34 },
  nFft: 320 as const,
  winLength: 320 as const,
  hopLength: 160 as const,
  featureLayout: 'mel-major' as const,
  predictionHiddenSize: 320,
  predictionRnnLayers: 1,
  maxTokensPerFrame: 3,
};

const xAsrConfig = {
  ecosystem: 'x-asr' as const,
  architecture: 'zipformer2-streaming-rnnt' as const,
  processorArchitecture: 'kaldi-fbank' as const,
  encoderArchitecture: 'zipformer2' as const,
  decoderArchitecture: 'stateless-rnnt' as const,
  sampleRate: 16000 as const,
  featureDim: 80,
  featureHopSeconds: 0.01,
  rawStride: 1,
  languages: ['zh', 'en'] as const,
  chunkMs: 160 as const,
  graph: {
    encoderStateInputs: [],
    encoderFrameSize: 16,
    encoderFrameShift: 16,
    decoderContextSize: 2,
  },
};

describe('CTC/transducer session.run output dispose', () => {
  it('copies GigaAM CTC logits then disposes all graph outputs, including on throw', async () => {
    const logits = new TrackingTensor('float32', new Float32Array(99 * 71).fill(-10), [1, 99, 71]);
    const extra = new TrackingTensor('int64', BigInt64Array.from([99n]), [1]);
    for (let frame = 0; frame < 99; frame += 1) (logits.data as Float32Array)[frame * 71 + 2] = 10;
    const session: OrtSessionLike = {
      async run() {
        return { log_probs: logits, encoded_lengths: extra };
      },
    };
    const executor = new OrtGigaAmCtcExecutor('gigaam-ctc-dispose', 'wasm', gigaamCtcConfig, undefined);
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({
      ort, session, tokenizer: GigaAmTokenizer.fromText('▁ 0\na 2\n<blk> 70\n'), warnings: [],
    });

    const result = await executor.transcribe(pcm());
    expect(result.utteranceText).toBe('a');
    expect(logits.disposed).toBe(1);
    expect(extra.disposed).toBe(1);

    const bad = new TrackingTensor('float32', new Float32Array(10), [2, 5]);
    const failing: OrtSessionLike = {
      async run() {
        return { log_probs: bad };
      },
    };
    const boom = new OrtGigaAmCtcExecutor('gigaam-ctc-throw', 'wasm', gigaamCtcConfig, undefined);
    (boom as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({
      ort, session: failing, tokenizer: GigaAmTokenizer.fromText('▁ 0\na 2\n<blk> 70\n'), warnings: [],
    });
    await expect(boom.transcribe(pcm())).rejects.toThrow(/Unexpected GigaAM logits shape/);
    expect(bad.disposed).toBe(1);
  });

  it('disposes SenseVoice logits and length tensors after a copied CTC pass', async () => {
    const logits = new TrackingTensor('float32', new Float32Array(8 * 4).fill(-8), [1, 8, 4]);
    const lengths = new TrackingTensor('int64', BigInt64Array.from([8n]), [1]);
    (logits.data as Float32Array)[1] = 8;
    const executor = new OrtSenseVoiceExecutor('sensevoice-dispose', 'wasm', undefined);
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({
      ort,
      session: {
        async run() {
          return { logprobs: logits, logprobs_lens: lengths };
        },
      },
      tokenizer: SenseVoiceTokenizer.fromText('<blank> 0\na 1\nb 2\nc 3\n'),
      warnings: [],
      graph: 'folded',
    });
    await executor.transcribe(pcm());
    expect(logits.disposed).toBe(1);
    expect(lengths.disposed).toBe(1);
  });

  it('copies LASR logits before disposing session outputs', async () => {
    const logits = new TrackingTensor('float32', new Float32Array(10 * 32).fill(-8), [1, 10, 32]);
    (logits.data as Float32Array)[1] = 8;
    const extra = new TrackingTensor('int64', BigInt64Array.from([10n]), [1]);
    const executor = new OrtLasrCtcExecutor(
      'lasr-ctc-dispose',
      DEFAULT_LASR_CTC_CLASSIFICATION,
      parseLasrCtcConfig('lasr-ctc-dispose'),
      'wasm',
      undefined,
    );
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({
      ort,
      session: {
        async run() {
          return { logits, extra };
        },
      },
      tokenizer: MedAsrTextTokenizer.fromText(
        Array.from({ length: 32 }, (_, id) => `${id === 31 ? '<blk>' : 'a'} ${id}`).join('\n'),
      ),
      preprocessor: new MedAsrJsPreprocessor({
        nMels: 128,
        center: false,
        preemphasis: 0,
        melScale: 'kaldi',
        slaneyNorm: false,
        logZeroGuard: 1e-5,
        normalizeFeatures: false,
      }),
      warnings: [],
    });
    await executor.transcribe(pcm(), {});
    expect(logits.disposed).toBe(1);
    expect(extra.disposed).toBe(1);
  });

  it('disposes Wav2Vec2 extractLogits and transcribe outputs after copying', async () => {
    const extractLogits = new TrackingTensor('float32', new Float32Array(10 * 3).fill(-8), [1, 10, 3]);
    (extractLogits.data as Float32Array)[2] = 8;
    const transcribeLogits = new TrackingTensor('float32', new Float32Array(10 * 3).fill(-8), [1, 10, 3]);
    (transcribeLogits.data as Float32Array)[2] = 8;
    let extract = true;
    const executor = new OrtWav2Vec2Executor(
      'wav2vec2-dispose',
      DEFAULT_WAV2VEC2_CLASSIFICATION,
      DEFAULT_WAV2VEC2_CONFIG,
      'wasm',
      undefined,
    );
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({
      ort,
      session: {
        async run() {
          return { logits: extract ? extractLogits : transcribeLogits };
        },
      },
      tokenizer: new Wav2Vec2CharTokenizer({ '<pad>': 0, '|': 1, a: 2 }),
      config: DEFAULT_WAV2VEC2_CONFIG,
      warnings: [],
    });
    const extracted = await executor.extractLogits(pcm());
    expect(extracted.logits[2]).toBe(8);
    expect(extractLogits.disposed).toBe(1);
    extract = false;
    await executor.transcribe(pcm(), {});
    expect(transcribeLogits.disposed).toBe(1);
  });

  it('copies GigaAM RNN-T encoder features then disposes encoder, decoder, and joint outputs', async () => {
    const encoded = new TrackingTensor('float32', new Float32Array(320), [1, 320, 1]);
    const encodedLen = new TrackingTensor('int64', BigInt64Array.from([1n]), [1]);
    const unusedEnc = new TrackingTensor('float32', new Float32Array(1), [1]);
    const decoderTensors: TrackingTensor[] = [];
    const jointTensors: TrackingTensor[] = [];
    const unusedJoint = new TrackingTensor('float32', new Float32Array(1), [1]);
    const executor = new OrtGigaAmRnntExecutor('gigaam-rnnt-dispose', 'wasm', gigaamRnntConfig, undefined);
    (executor as unknown as { state: Promise<unknown> }).state = Promise.resolve({
      ort,
      encoder: {
        async run() {
          return { encoded, encoded_len: encodedLen, unused: unusedEnc };
        },
      },
      decoder: {
        async run() {
          const dec = new TrackingTensor('float32', new Float32Array(320), [1, 320, 1]);
          const ho = new TrackingTensor('float32', new Float32Array(320), [1, 1, 320]);
          const co = new TrackingTensor('float32', new Float32Array(320), [1, 1, 320]);
          decoderTensors.push(dec, ho, co);
          return { dec, ho, co };
        },
      },
      joint: {
        async run() {
          const logits = new Float32Array(35).fill(-8);
          logits[34] = 8;
          const joint = new TrackingTensor('float32', logits, [35]);
          jointTensors.push(joint);
          return { joint, unused: unusedJoint };
        },
      },
      tokenizer: GigaAmRnntTokenizer.fromText('  0\na 2\n<blk> 34\n'),
      warnings: [],
    });
    await executor.transcribe(pcm());
    expect(encoded.disposed).toBe(1);
    expect(encodedLen.disposed).toBe(1);
    expect(unusedEnc.disposed).toBe(1);
    expect(decoderTensors.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(jointTensors.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(unusedJoint.disposed).toBeGreaterThan(0);
  });

  it('copies X-ASR encoder/joiner outputs then disposes leftover Ort tensors', async () => {
    const unusedEnc: TrackingTensor[] = [];
    const joinerLogits: TrackingTensor[] = [];
    const unusedJoin: TrackingTensor[] = [];
    const executor = new OrtXAsrExecutor('x-asr-dispose', 'wasm', xAsrConfig);
    (executor as unknown as { source: object }).source = { kind: 'direct', artifacts: {} };
    (executor as unknown as { state: Promise<unknown> }).state = Promise.resolve({
      ort,
      encoder: {
        async run() {
          const encoded = new TrackingTensor('float32', new Float32Array(8), [1, 1, 8]);
          const extra = new TrackingTensor('float32', new Float32Array(1), [1]);
          unusedEnc.push(extra);
          return { encoder_out: encoded, unused: extra };
        },
      },
      decoder: {
        async run() {
          return { decoder_out: new TrackingTensor('float32', new Float32Array(8), [1, 8]) };
        },
      },
      joiner: {
        async run() {
          const logits = new Float32Array(2);
          logits[0] = 8;
          logits[1] = -8;
          const joint = new TrackingTensor('float32', logits, [2]);
          const extra = new TrackingTensor('float32', new Float32Array(1), [1]);
          joinerLogits.push(joint);
          unusedJoin.push(extra);
          return { logit: joint, unused: extra };
        },
      },
      tokenizer: XAsrTokenizer.fromText('<blk> 0\n▁a 1\n'),
      graph: {
        featureInputName: 'x',
        encoderOutputName: 'encoder_out',
        encoderFrameSize: 16,
        encoderFrameShift: 16,
        encoderStateInputs: [],
        decoderInputName: 'y',
        decoderOutputName: 'decoder_out',
        decoderContextSize: 2,
        decoderIndexType: 'int64',
        joinerEncoderInputName: 'encoder_out',
        joinerDecoderInputName: 'decoder_out',
        joinerOutputName: 'logit',
      },
    });
    await executor.transcribe(pcm());
    expect(unusedEnc.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(joinerLogits.every((tensor) => tensor.disposed === 1)).toBe(true);
    expect(unusedJoin.every((tensor) => tensor.disposed === 1)).toBe(true);
  });
});
