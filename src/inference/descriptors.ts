export type AcousticEncoderKind =
  | 'fastconformer'
  | 'conformer'
  | 'wav2vec2-conformer'
  | 'whisper-transformer'
  | 'zipformer'
  | 'dfsmn'
  | 'transformer'
  | 'speech-adapter';

export interface AcousticEncoderDescriptor {
  readonly kind: AcousticEncoderKind;
  readonly sharedModule: 'inference';
  readonly outputStride?: number;
  readonly streaming?: boolean;
  readonly notes?: readonly string[];
}

export type DecoderHeadKind =
  | 'ctc-head'
  | 'transducer-tdt'
  | 'transformer-decoder'
  | 'speech-llm'
  | 'vad-frame-classifier';

export interface DecoderHeadDescriptor {
  readonly kind: DecoderHeadKind;
  readonly sharedModule: 'inference';
  readonly supportsStreaming?: boolean;
  readonly notes?: readonly string[];
}

export type DecodingTopology =
  | 'ctc'
  | 'rnnt'
  | 'tdt'
  | 'stateless-rnnt'
  | 'aed'
  | 'speech-llm'
  | 'vad';

export type DecodingStrategyKind =
  | 'ctc-greedy'
  | 'ctc-beam'
  | 'rnnt-greedy'
  | 'rnnt-beam'
  | 'tdt-greedy'
  | 'aed-generate'
  | 'speech-llm-generate'
  | 'vad-threshold';

export interface DecodingDescriptor {
  readonly topology: DecodingTopology;
  readonly strategy: DecodingStrategyKind;
  readonly supportsStreaming?: boolean;
  readonly supportsBeamSearch?: boolean;
  readonly notes?: readonly string[];
}

export function createAcousticEncoderDescriptor(
  descriptor: AcousticEncoderDescriptor,
): AcousticEncoderDescriptor {
  return descriptor;
}

export function createDecoderHeadDescriptor(
  descriptor: DecoderHeadDescriptor,
): DecoderHeadDescriptor {
  return descriptor;
}

export function createDecodingDescriptor(descriptor: DecodingDescriptor): DecodingDescriptor {
  return descriptor;
}

export const FASTCONFORMER_ENCODER = createAcousticEncoderDescriptor({
  kind: 'fastconformer',
  sharedModule: 'inference',
  outputStride: 8,
  streaming: true,
});

export const CONFORMER_ENCODER = createAcousticEncoderDescriptor({
  kind: 'conformer',
  sharedModule: 'inference',
  outputStride: 160,
  streaming: false,
});

export const WAV2VEC2_CONFORMER_ENCODER = createAcousticEncoderDescriptor({
  kind: 'wav2vec2-conformer',
  sharedModule: 'inference',
  outputStride: 320,
  streaming: false,
});

export const WHISPER_TRANSFORMER_ENCODER = createAcousticEncoderDescriptor({
  kind: 'whisper-transformer',
  sharedModule: 'inference',
  outputStride: 2,
  streaming: false,
});

export const CTC_HEAD_DECODER = createDecoderHeadDescriptor({
  kind: 'ctc-head',
  sharedModule: 'inference',
  supportsStreaming: true,
});

export const TDT_TRANSDUCER_DECODER = createDecoderHeadDescriptor({
  kind: 'transducer-tdt',
  sharedModule: 'inference',
  supportsStreaming: true,
  notes: ['Prediction net plus joiner with duration outputs.'],
});

export const TRANSFORMER_SEQ2SEQ_DECODER = createDecoderHeadDescriptor({
  kind: 'transformer-decoder',
  sharedModule: 'inference',
  supportsStreaming: false,
});

export const TDT_GREEDY_DECODING: DecodingDescriptor = {
  topology: 'tdt',
  strategy: 'tdt-greedy',
  supportsStreaming: true,
  supportsBeamSearch: false,
  notes: [
    'Duration-aware greedy transducer decoding.',
    'Keeps decoding policy separate from the prediction network and joiner.',
  ],
};

export const CTC_GREEDY_DECODING: DecodingDescriptor = {
  topology: 'ctc',
  strategy: 'ctc-greedy',
  supportsStreaming: true,
  supportsBeamSearch: false,
  notes: ['Argmax plus blank collapse over frame-level logits.'],
};

export const AED_GENERATE_DECODING: DecodingDescriptor = {
  topology: 'aed',
  strategy: 'aed-generate',
  supportsStreaming: false,
  supportsBeamSearch: true,
  notes: ['Autoregressive decoding with prompt tokens and next-token generation.'],
};

export const WHISPER_GENERATE_DECODING: DecodingDescriptor = AED_GENERATE_DECODING;
