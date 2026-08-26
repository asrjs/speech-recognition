import type {
  AssetProvider,
  AudioBufferLike,
  BaseTranscriptionOptions,
  ModelClassification,
  SpeechRuntimeHooks,
  TranscriptMetrics,
  TranscriptWarning,
} from '../../types/index.js';
import type { LasrCtcFeatureBatch } from '../lasr-ctc/types.js';

export const SENSEVOICE_LANGUAGES = ['auto', 'zh', 'en', 'yue', 'ja', 'ko'] as const;
export type SenseVoiceLanguage = (typeof SENSEVOICE_LANGUAGES)[number];

export const SENSEVOICE_LANGUAGE_IDS: Readonly<Record<SenseVoiceLanguage, number>> = {
  auto: 0,
  zh: 3,
  en: 4,
  yue: 7,
  ja: 11,
  ko: 12,
};

export const SENSEVOICE_TEXTNORM_IDS = {
  withitn: 14,
  woitn: 15,
} as const;

export interface SenseVoiceFeatureBatch extends LasrCtcFeatureBatch {
  readonly validFrameCount: number;
}

export interface SenseVoicePrompt {
  readonly language: SenseVoiceLanguage;
  readonly languageId: number;
  readonly textnorm: 'withitn' | 'woitn';
  readonly textnormId: number;
}

export interface SenseVoiceNativeMetadata {
  readonly language?: string;
  readonly emotion?: string;
  readonly event?: string;
}

export interface SenseVoiceModelConfig {
  readonly ecosystem: 'funasr';
  readonly architecture: 'sensevoice';
  readonly processorArchitecture: 'kaldi-fbank';
  readonly encoderArchitecture: 'sensevoice-conformer';
  readonly decoderArchitecture: 'ctc';
  readonly sampleRate: number;
  readonly featureHopSeconds: number;
  readonly nMels: 80;
  readonly vocabularySize: number;
  readonly ctcBlankId: number;
  readonly languages: readonly SenseVoiceLanguage[];
}

export interface SenseVoiceDirectArtifacts {
  readonly modelUrl: string;
  readonly tokenizerUrl: string;
  readonly modelDataUrl?: string;
  readonly modelDataFilename?: string;
}

export interface SenseVoiceDirectSource {
  readonly kind: 'direct';
  readonly artifacts: SenseVoiceDirectArtifacts;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export interface SenseVoiceHuggingFaceSource {
  readonly kind: 'huggingface';
  readonly repoId: string;
  readonly revision?: string;
  readonly modelFilename?: string;
  readonly modelDataFilename?: string;
  readonly tokenizerFilename?: string;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export type SenseVoiceArtifactSource = SenseVoiceDirectSource | SenseVoiceHuggingFaceSource;

export interface SenseVoiceModelOptions {
  readonly config?: Partial<SenseVoiceModelConfig>;
  readonly source?: SenseVoiceArtifactSource;
}

export interface SenseVoiceTranscriptionOptions extends BaseTranscriptionOptions {
  readonly language?: SenseVoiceLanguage | string;
  readonly useItn?: boolean;
  readonly returnTokenIds?: boolean;
  readonly returnLogitIndices?: boolean;
}

export interface SenseVoiceNativeToken {
  readonly index: number;
  readonly id?: number;
  readonly text: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
}

export interface SenseVoiceNativeTranscript {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly language?: string;
  readonly metadata?: SenseVoiceNativeMetadata;
  readonly tokens?: readonly SenseVoiceNativeToken[];
  readonly confidence?: { readonly utterance?: number; readonly tokenAverage?: number };
  readonly metrics?: TranscriptMetrics;
  readonly warnings?: readonly TranscriptWarning[];
}

export interface SenseVoiceExecutor {
  ready?(): Promise<void> | void;
  transcribe(
    audio: AudioBufferLike,
    options: SenseVoiceTranscriptionOptions,
  ): Promise<SenseVoiceNativeTranscript>;
  dispose(): Promise<void> | void;
}

export interface SenseVoiceModelDependencies {
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
  readonly executor?: SenseVoiceExecutor;
}

export interface SenseVoiceModelFamilyOptions {
  readonly dependencies?: SenseVoiceModelDependencies;
  readonly family?: string;
}

export interface SenseVoiceModelClassification extends ModelClassification {
  readonly family: string;
}
