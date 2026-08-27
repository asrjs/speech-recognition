import { describe, expect, it } from 'vitest';

import { OrtGigaAmCtcExecutor } from '../src/models/gigaam-ctc/executor.js';
import { GigaAmTokenizer } from '../src/models/gigaam-ctc/tokenizer.js';
import { OrtGigaAmRnntExecutor } from '../src/models/gigaam-rnnt/executor.js';
import { GigaAmRnntTokenizer } from '../src/models/gigaam-rnnt/tokenizer.js';
import { DEFAULT_LASR_CTC_CLASSIFICATION, parseLasrCtcConfig } from '../src/models/lasr-ctc/config.js';
import { OrtLasrCtcExecutor } from '../src/models/lasr-ctc/executor.js';
import type { OrtSessionLike } from '../src/models/lasr-ctc/ort.js';
import { DEFAULT_NEMO_AED_CLASSIFICATION, parseNemoAedConfig } from '../src/models/nemo-aed/config.js';
import { OrtNemoAedExecutor } from '../src/models/nemo-aed/executor.js';
import { DEFAULT_NEMO_RNNT_CLASSIFICATION, parseNemoRnntConfig } from '../src/models/nemo-rnnt/config.js';
import { OrtNemoRnntExecutor } from '../src/models/nemo-rnnt/executor.js';
import { DEFAULT_NEMO_TDT_CLASSIFICATION, parseNemoTdtConfig } from '../src/models/nemo-tdt/config.js';
import { OrtNemoTdtExecutor } from '../src/models/nemo-tdt/executor.js';
import { OnnxNemoPreprocessor } from '../src/models/nemo-tdt/preprocessor.js';
import { OrtSenseVoiceExecutor } from '../src/models/sensevoice/executor.js';
import { DEFAULT_WAV2VEC2_CLASSIFICATION, DEFAULT_WAV2VEC2_CONFIG } from '../src/models/wav2vec2/config.js';
import { OrtWav2Vec2Executor } from '../src/models/wav2vec2/executor.js';
import { DEFAULT_WHISPER_CLASSIFICATION, parseWhisperSeq2SeqConfig } from '../src/models/whisper-seq2seq/config.js';
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/executor.js';
import { DEFAULT_QWEN3_ASR_CONFIG } from '../src/models/qwen-asr/config.js';
import { OrtQwen3AsrExecutor } from '../src/models/qwen-asr/executor.js';
import { OrtXAsrExecutor } from '../src/models/x-asr/executor.js';
import { XAsrTokenizer } from '../src/models/x-asr/tokenizer.js';

type LoadStateOwner = { loadStatePromise?: Promise<unknown> };

function setLoadState(executor: object, state: Promise<unknown> | unknown): void {
  (executor as LoadStateOwner).loadStatePromise =
    state instanceof Promise ? state : Promise.resolve(state);
}

function trackingSession(label: string, released: string[]): OrtSessionLike {
  return {
    async run() {
      return {};
    },
    release() {
      released.push(label);
    },
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

describe('experimental ORT session release', () => {
  it('releases the GigaAM CTC session even when load is still pending', async () => {
    const released: string[] = [];
    const session = trackingSession('ctc', released);
    let resolveLoad!: (value: unknown) => void;
    const executor = new OrtGigaAmCtcExecutor('gigaam-ctc-dispose', 'wasm', gigaamCtcConfig, undefined);
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = new Promise((resolve) => {
      resolveLoad = resolve;
    });

    void executor.dispose();
    expect(released).toEqual([]);
    resolveLoad({
      session,
      tokenizer: GigaAmTokenizer.fromText('▁ 0\na 2\n<blk> 70\n'),
      warnings: [],
    });
    await executor.dispose();
    expect(released).toEqual(['ctc']);
    await expect(
      executor.transcribe({
        sampleRate: 16000,
        numberOfChannels: 1,
        numberOfFrames: 160,
        durationSeconds: 0.01,
        channels: [new Float32Array(160)],
      }),
    ).rejects.toThrow(/disposed/);
  });

  it('releases GigaAM RNN-T encoder, decoder, and joint', async () => {
    const released: string[] = [];
    const executor = new OrtGigaAmRnntExecutor('gigaam-rnnt-dispose', 'wasm', gigaamRnntConfig, undefined);
    (executor as unknown as { state: Promise<unknown> }).state = Promise.resolve({
      encoder: trackingSession('encoder', released),
      decoder: trackingSession('decoder', released),
      joint: trackingSession('joint', released),
      tokenizer: GigaAmRnntTokenizer.fromText('  0\na 2\n<blk> 34\n'),
      warnings: [],
    });
    await executor.dispose();
    expect(released).toEqual(['encoder', 'decoder', 'joint']);
  });

  it('releases the SenseVoice session and still drops asset handles', async () => {
    const released: string[] = [];
    const executor = new OrtSenseVoiceExecutor('sensevoice-dispose', 'wasm', undefined);
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({
      session: trackingSession('sensevoice', released),
      warnings: [],
      graph: 'folded',
    });
    await executor.dispose();
    expect(released).toEqual(['sensevoice']);
  });

  it('releases X-ASR encoder, decoder, and joiner', async () => {
    const released: string[] = [];
    const executor = new OrtXAsrExecutor('x-asr-dispose', 'wasm', xAsrConfig);
    (executor as unknown as { state: Promise<unknown> }).state = Promise.resolve({
      encoder: trackingSession('encoder', released),
      decoder: trackingSession('decoder', released),
      joiner: trackingSession('joiner', released),
      tokenizer: XAsrTokenizer.fromText('<blk> 0\n▁a 1\n'),
      graph: xAsrConfig.graph,
    });
    await executor.dispose();
    expect(released).toEqual(['encoder', 'decoder', 'joiner']);
  });
});

describe('production ORT session release', () => {
  it('releases the LASR CTC session even when load is still pending', async () => {
    const released: string[] = [];
    const session = trackingSession('lasr', released);
    let resolveLoad!: (value: unknown) => void;
    const executor = new OrtLasrCtcExecutor(
      'lasr-ctc-dispose',
      DEFAULT_LASR_CTC_CLASSIFICATION,
      parseLasrCtcConfig('lasr-ctc-dispose'),
      'wasm',
      undefined,
    );
    setLoadState(
      executor,
      new Promise((resolve) => {
        resolveLoad = resolve;
      }),
    );

    const disposing = executor.dispose();
    expect(released).toEqual([]);
    resolveLoad({
      session,
      warnings: [],
    });
    await disposing;
    expect(released).toEqual(['lasr']);
    await expect(executor.ready()).rejects.toThrow(/disposed/);
  });

  it('releases Wav2Vec2 session on dispose', async () => {
    const released: string[] = [];
    const executor = new OrtWav2Vec2Executor(
      'wav2vec2-dispose',
      DEFAULT_WAV2VEC2_CLASSIFICATION,
      DEFAULT_WAV2VEC2_CONFIG,
      'wasm',
      undefined,
    );
    setLoadState(executor, {
      session: trackingSession('wav2vec2', released),
      warnings: [],
    });
    await executor.dispose();
    expect(released).toEqual(['wav2vec2']);
    await expect(executor.ready()).rejects.toThrow(/disposed/);
  });

  it('releases NeMo TDT encoder, decoder, and preprocessor', async () => {
    const released: string[] = [];
    const executor = new OrtNemoTdtExecutor(
      'nemo-tdt-dispose',
      DEFAULT_NEMO_TDT_CLASSIFICATION,
      parseNemoTdtConfig('nemo-tdt-dispose'),
      'wasm',
      undefined,
    );
    setLoadState(executor, {
      encoderSession: trackingSession('encoder', released),
      decoderSession: trackingSession('decoder', released),
      preprocessor: {
        async process() {
          return { features: new Float32Array(0), frameCount: 0, validLength: 0 };
        },
        async dispose() {
          released.push('preprocessor');
        },
      },
      warnings: [],
    });
    await executor.dispose();
    expect(released).toEqual(['encoder', 'decoder', 'preprocessor']);
    await expect(executor.ready()).rejects.toThrow(/disposed/);
  });

  it('releases NeMo AED encoder, decoder, and preprocessor', async () => {
    const released: string[] = [];
    const executor = new OrtNemoAedExecutor(
      'nemo-aed-dispose',
      DEFAULT_NEMO_AED_CLASSIFICATION,
      parseNemoAedConfig('nemo-aed-dispose'),
      'wasm',
      undefined,
    );
    setLoadState(executor, {
      encoderSession: trackingSession('encoder', released),
      decoderSession: trackingSession('decoder', released),
      preprocessor: {
        async process() {
          return { features: new Float32Array(0), frameCount: 0, validLength: 0 };
        },
        async dispose() {
          released.push('preprocessor');
        },
      },
      warnings: [],
    });
    await executor.dispose();
    expect(released).toEqual(['encoder', 'decoder', 'preprocessor']);
    await expect(executor.ready()).rejects.toThrow(/disposed/);
  });

  it('releases NeMo RNN-T encoder, decoder, and preprocessor', async () => {
    const released: string[] = [];
    const executor = new OrtNemoRnntExecutor(
      'nemo-rnnt-dispose',
      DEFAULT_NEMO_RNNT_CLASSIFICATION,
      parseNemoRnntConfig('nemo-rnnt-dispose'),
      'wasm',
      undefined,
    );
    setLoadState(executor, {
      encoderSession: trackingSession('encoder', released),
      decoderSession: trackingSession('decoder', released),
      preprocessor: {
        async process() {
          return { features: new Float32Array(0), frameCount: 0, validLength: 0 };
        },
        async dispose() {
          released.push('preprocessor');
        },
      },
      warnings: [],
    });
    await executor.dispose();
    expect(released).toEqual(['encoder', 'decoder', 'preprocessor']);
    await expect(executor.ready()).rejects.toThrow(/disposed/);
  });

  it('releases the ONNX NeMo preprocessor session even when create is still pending', async () => {
    const released: string[] = [];
    const preprocessor = new OnnxNemoPreprocessor(
      {
        env: { wasm: {} },
        InferenceSession: {
          create: async () => trackingSession('unused', released),
        },
      } as never,
      'preprocessor.onnx',
    );
    let resolveSession!: (session: OrtSessionLike) => void;
    (preprocessor as unknown as { sessionPromise?: Promise<OrtSessionLike> }).sessionPromise =
      new Promise((resolve) => {
        resolveSession = resolve;
      });

    const disposing = preprocessor.dispose();
    expect(released).toEqual([]);
    resolveSession(trackingSession('preprocessor', released));
    await disposing;
    expect(released).toEqual(['preprocessor']);
  });

  it('releases Whisper encoder, decoder graphs, and lazy alignment sessions once', async () => {
    const released: string[] = [];
    const decoderAlign = trackingSession('decoderAlign', released);
    const executor = new WhisperOnnxExecutor(
      'whisper-dispose',
      DEFAULT_WHISPER_CLASSIFICATION,
      parseWhisperSeq2SeqConfig('whisper-dispose'),
      'wasm',
      undefined,
    );
    setLoadState(executor, {
      encoderSession: trackingSession('encoder', released),
      decoderSession: trackingSession('decoder', released),
      decoderInitSession: trackingSession('decoderInit', released),
      decoderStepSession: trackingSession('decoderStep', released),
      decoderAlignSession: decoderAlign,
      warnings: [],
    });
    Object.assign(executor, {
      decoderAlignSession: decoderAlign,
      alignmentReferenceEncoderSession: trackingSession('alignEncoder', released),
      alignmentReferenceDecoderAlignSession: trackingSession('alignDecoder', released),
    });

    await executor.dispose();
    await executor.dispose();
    expect(released).toEqual([
      'encoder',
      'decoder',
      'decoderInit',
      'decoderStep',
      'decoderAlign',
      'alignEncoder',
      'alignDecoder',
    ]);
    await expect(executor.ready()).rejects.toThrow(/disposed/);
  });

  it('releases Whisper sessions even when load is still pending', async () => {
    const released: string[] = [];
    let resolveLoad!: (value: unknown) => void;
    const executor = new WhisperOnnxExecutor(
      'whisper-pending-dispose',
      DEFAULT_WHISPER_CLASSIFICATION,
      parseWhisperSeq2SeqConfig('whisper-pending-dispose'),
      'wasm',
      undefined,
    );
    setLoadState(
      executor,
      new Promise((resolve) => {
        resolveLoad = resolve;
      }),
    );

    const disposing = executor.dispose();
    expect(released).toEqual([]);
    resolveLoad({
      encoderSession: trackingSession('encoder', released),
      decoderInitSession: trackingSession('decoderInit', released),
      decoderStepSession: trackingSession('decoderStep', released),
      warnings: [],
    });
    await disposing;
    expect(released).toEqual(['encoder', 'decoderInit', 'decoderStep']);
  });
});

describe('Qwen ORT session release', () => {
  it('releases encoder, prefill, and step sessions once', async () => {
    const released: string[] = [];
    const executor = new OrtQwen3AsrExecutor(
      'qwen-dispose',
      DEFAULT_QWEN3_ASR_CONFIG,
      'wasm',
      undefined,
    );
    setLoadState(executor, {
      encoderSession: trackingSession('encoder', released),
      decoderSession: trackingSession('prefill', released),
      decoderStepSession: trackingSession('step', released),
      warnings: [],
    });
    await executor.dispose();
    await executor.dispose();
    expect(released).toEqual(['encoder', 'prefill', 'step']);
    await expect(executor.ready()).rejects.toThrow(/disposed/);
  });

  it('releases Qwen sessions even when load is still pending', async () => {
    const released: string[] = [];
    let resolveLoad!: (value: unknown) => void;
    const executor = new OrtQwen3AsrExecutor(
      'qwen-pending-dispose',
      DEFAULT_QWEN3_ASR_CONFIG,
      'wasm',
      undefined,
    );
    setLoadState(
      executor,
      new Promise((resolve) => {
        resolveLoad = resolve;
      }),
    );

    const disposing = executor.dispose();
    expect(released).toEqual([]);
    resolveLoad({
      encoderSession: trackingSession('encoder', released),
      decoderSession: trackingSession('prefill', released),
      decoderStepSession: trackingSession('step', released),
      warnings: [],
    });
    await disposing;
    expect(released).toEqual(['encoder', 'prefill', 'step']);
    await expect(executor.ready()).rejects.toThrow(/disposed/);
  });
});
