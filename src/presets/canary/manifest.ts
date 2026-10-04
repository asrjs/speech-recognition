import type { ModelClassification } from '../../types/index.js';
import type {
  NemoAedArtifactSource,
  NemoAedModelConfig,
  NemoAedPromptSettings,
} from '../../models/nemo-aed/index.js';

/**
 * Prompt-token contract used by the Canary runtime.
 *
 * The upstream model card describes task control via task/language/timestamp/PnC
 * tokens. In this repo the exported tokenizer metadata and prompt builder make
 * that contract explicit with the exact special-token sequence consumed by the
 * decoder.
 */
export interface CanaryPromptTokenDocumentation {
  readonly token: string;
  readonly purpose: string;
  readonly runtimeField?: keyof NemoAedPromptSettings | 'system';
}

/**
 * Human-readable metadata copied from the upstream model card and verified
 * against the NeMo checkpoint config used during the port.
 */
export interface CanaryPresetDocumentation {
  readonly modelCardUrl: string;
  readonly architecture: {
    readonly summary: string;
    readonly encoder: 'fastconformer';
    readonly decoder: 'transformer-decoder';
    readonly encoderLayers: number;
    readonly decoderLayers: number;
    readonly parameterCount: string;
    readonly tokenizer: string;
  };
  readonly capabilities: {
    readonly defaultBehavior: string;
    readonly supportedLanguages: readonly string[];
    readonly supportedTasks: readonly string[];
    readonly supportsWordTimestamps: boolean;
    readonly supportsSegmentTimestamps: boolean;
  };
  readonly input: {
    readonly acceptedFormats: readonly string[];
    readonly sampleRate: number;
    readonly channels: 'mono';
    readonly note: string;
  };
  readonly prompting: {
    readonly format: string;
    readonly tokens: readonly CanaryPromptTokenDocumentation[];
    readonly note: string;
  };
  readonly longform: {
    readonly preferredNeMoScript: string;
    readonly designedForAudioUpToSeconds: number;
    readonly recommendedTimestampChunkSeconds: number;
  };
  readonly preprocessing: {
    readonly nemoApiHandlesFrontendInternally: true;
    readonly runtimeStillNeedsFrontendArtifacts: true;
    readonly currentRuntimeDefault: string;
    readonly note: string;
  };
}

export interface CanaryPresetManifest {
  readonly preset: 'canary';
  readonly modelId: string;
  readonly aliases?: readonly string[];
  readonly description: string;
  readonly classification: ModelClassification;
  readonly config: Partial<NemoAedModelConfig>;
  readonly source?: NemoAedArtifactSource;
  readonly docs?: CanaryPresetDocumentation;
}

export const DEFAULT_MODEL = 'nvidia/canary-180m-flash';

/**
 * Ordered special-token prompt consumed by the Canary decoder in this runtime.
 *
 * The model card describes language/task/timestamp/PnC controls at a higher
 * level. The exported tokenizer and prompt builder resolve that into the exact
 * decoder prefix we feed before autoregressive generation starts.
 */
export const CANARY_PROMPT_TOKENS: readonly CanaryPromptTokenDocumentation[] = [
  {
    token: '<|startofcontext|>',
    purpose: 'Marks the beginning of the multitask prompt context.',
    runtimeField: 'system',
  },
  {
    token: '<|startoftranscript|>',
    purpose: 'Marks the beginning of the actual transcript generation request.',
    runtimeField: 'system',
  },
  {
    token: '<|emo:undefined|>',
    purpose: 'Emotion slot used by Canary prompt format `canary2`.',
    runtimeField: 'emotion',
  },
  {
    token: '<|en|> / <|de|> / <|es|> / <|fr|>',
    purpose: 'Source-language token for the input audio.',
    runtimeField: 'sourceLanguage',
  },
  {
    token: '<|en|> / <|de|> / <|es|> / <|fr|>',
    purpose: 'Target-language token for ASR or speech translation output.',
    runtimeField: 'targetLanguage',
  },
  {
    token: '<|pnc|> / <|nopnc|>',
    purpose: 'Toggles punctuation and capitalization generation.',
    runtimeField: 'punctuate',
  },
  {
    token: '<|itn|> / <|noitn|>',
    purpose: 'Toggles inverse text normalization requests.',
    runtimeField: 'inverseTextNormalization',
  },
  {
    token: '<|timestamp|> / <|notimestamp|>',
    purpose: 'Toggles timestamp generation.',
    runtimeField: 'timestamps',
  },
  {
    token: '<|diarize|> / <|nodiarize|>',
    purpose: 'Toggles diarization-aware prompting.',
    runtimeField: 'diarize',
  },
];

/**
 * Upstream model-card notes plus port-specific runtime guidance.
 *
 * Important nuance:
 * - "Pre-Processing Not Needed" on the model card means callers should pass raw
 *   16 kHz mono audio to NeMo and let NeMo run the frontend internally.
 * - It does not mean the model has no mel frontend. Our browser/Node runtime
 *   still needs a feature extractor, either ONNX or JS, before the encoder.
 */
export const CANARY_180M_FLASH_DOCS: CanaryPresetDocumentation = {
  modelCardUrl: 'https://huggingface.co/nvidia/canary-180m-flash',
  architecture: {
    summary:
      'FastConformer encoder + Transformer decoder encoder-decoder ASR/translation model with aggregate SentencePiece tokenization.',
    encoder: 'fastconformer',
    decoder: 'transformer-decoder',
    encoderLayers: 17,
    decoderLayers: 4,
    parameterCount: '182M',
    tokenizer:
      'Aggregate tokenizer built from per-language SentencePiece tokenizers plus Canary special prompt tokens.',
  },
  capabilities: {
    defaultBehavior: 'English ASR when callers pass plain audio paths without a manifest.',
    supportedLanguages: ['en', 'de', 'es', 'fr'],
    supportedTasks: ['asr', 'speech-translation'],
    supportsWordTimestamps: true,
    supportsSegmentTimestamps: true,
  },
  input: {
    acceptedFormats: ['wav', 'flac'],
    sampleRate: 16000,
    channels: 'mono',
    note: 'NeMo expects raw audio input and performs frontend feature extraction internally.',
  },
  prompting: {
    format: 'canary2',
    tokens: CANARY_PROMPT_TOKENS,
    note: 'Task control is encoded through special prompt tokens rather than a separate task enum in the runtime.',
  },
  longform: {
    preferredNeMoScript: 'scripts/speech_to_text_aed_chunked_infer.py',
    designedForAudioUpToSeconds: 40,
    recommendedTimestampChunkSeconds: 10,
  },
  preprocessing: {
    nemoApiHandlesFrontendInternally: true,
    runtimeStillNeedsFrontendArtifacts: true,
    currentRuntimeDefault: 'Shared NeMo 128-bin frontend via in-repo JS mel.',
    note: 'The repo now defaults Canary to the shared in-repo JS frontend. Keep `nemo128.onnx` as an optional comparison or fallback artifact while NeMo-reference feature parity continues to improve.',
  },
};

export const CANARY_PRESET_MANIFESTS: readonly CanaryPresetManifest[] = [
  {
    preset: 'canary',
    modelId: DEFAULT_MODEL,
    aliases: ['canary-180m-flash'],
    description: 'Canary preset over the shared NeMo AED implementation boundary.',
    classification: {
      ecosystem: 'nemo',
      processor: 'nemo-mel',
      encoder: 'fastconformer',
      decoder: 'transformer-decoder',
      topology: 'aed',
      family: 'canary',
      task: 'multitask-asr-translation',
    },
    config: {
      vocabularySize: 5248,
      languages: ['en', 'de', 'es', 'fr'],
      melBins: 128,
      promptFormat: 'canary2',
      promptDefaults: [
        {
          sourceLanguage: 'en',
          targetLanguage: 'en',
          decoderContext: '',
          emotion: '<|emo:undefined|>',
          punctuate: true,
          inverseTextNormalization: false,
          timestamps: false,
          diarize: false,
        },
      ],
    },
    source: {
      kind: 'huggingface',
      repoId: 'ysdede/canary-180m-flash-onnx',
      preprocessorBackend: 'js',
    },
    docs: CANARY_180M_FLASH_DOCS,
  },
];

function normalizePresetId(modelId: string): string {
  return modelId.trim().toLowerCase();
}

export function listCanaryPresetManifests(): readonly CanaryPresetManifest[] {
  return CANARY_PRESET_MANIFESTS;
}

export function resolveCanaryPresetManifest(modelId: string): CanaryPresetManifest | undefined {
  const normalizedModelId = normalizePresetId(modelId);
  return CANARY_PRESET_MANIFESTS.find((manifest) => {
    if (normalizePresetId(manifest.modelId) === normalizedModelId) {
      return true;
    }
    return (manifest.aliases ?? []).some((alias) => normalizePresetId(alias) === normalizedModelId);
  });
}

export function resolveCanaryArtifactSource(modelId: string): NemoAedArtifactSource | undefined {
  return resolveCanaryPresetManifest(modelId)?.source;
}
