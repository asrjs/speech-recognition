import {
  MODELS as CANARY_MODELS,
} from './canary/catalog.js';
import type { ModelInferenceLimits } from '../types/index.js';
import { resolveCanaryArtifactSource, resolveCanaryPresetManifest } from './canary/manifest.js';
import { resolveMedAsrPresetManifest } from './medasr/manifest.js';
import {
  MODELS as PARAKEET_MODELS,
} from './parakeet/catalog.js';
import {
  resolveParakeetArtifactSource,
  resolveParakeetPresetManifest,
} from './parakeet/manifest.js';
import { resolveWhisperPresetManifest } from './whisper/manifest.js';
import { resolveWav2Vec2PresetManifest } from './wav2vec2/manifest.js';
import {
  resolveNemotronArtifactSource,
  resolveNemotronPresetManifest,
} from './nemotron/factory.js';
import type { BuiltInModelDescriptor } from './descriptors.js';

function buildWarmupExpectedTexts(): readonly string[] {
  return [
    'The boy was there when the sun rose.',
    'The boy was there when the sun rose. A rod is used to catch pink salmon.',
  ];
}

function createParakeetInferenceLimits(config: {
  readonly topology?: 'tdt' | 'rnnt';
  readonly supportsWordTimestamps?: boolean;
}): ModelInferenceLimits {
  if (config.topology === 'rnnt') {
    return {
      sampleRate: 16000,
      maxInputDurationSec: 30,
      recommendedWindowDurationSec: 15,
      minWindowDurationSec: 5,
      maxWindowDurationSec: 30,
      autoWindowThresholdSec: 30,
      defaultOverlapSec: 2,
      preferVadSegmentWindowing: true,
      supportsWordTimestamps: config.supportsWordTimestamps ?? false,
      supportsTokenTimestamps: true,
      supportsSegmentTimestamps: true,
      supportsConfidence: true,
      defaultSegmentationStrategy: 'vad',
      defaultMergeStrategy: 'concat',
    };
  }

  return {
    sampleRate: 16000,
    maxInputDurationSec: 180,
    recommendedWindowDurationSec: 90,
    minWindowDurationSec: 20,
    maxWindowDurationSec: 180,
    autoWindowThresholdSec: 180,
    defaultOverlapSec: 10,
    preferSentenceBoundaryWindowing: true,
    supportsWordTimestamps: config.supportsWordTimestamps ?? true,
    supportsTokenTimestamps: true,
    supportsSegmentTimestamps: true,
    supportsConfidence: true,
    defaultSegmentationStrategy: 'word-punctuation',
    defaultMergeStrategy: 'word-dedupe',
  };
}

function createConservativeAedInferenceLimits(): ModelInferenceLimits {
  return {
    sampleRate: 16000,
    maxInputDurationSec: 60,
    recommendedWindowDurationSec: 30,
    minWindowDurationSec: 5,
    maxWindowDurationSec: 60,
    autoWindowThresholdSec: 60,
    defaultOverlapSec: 5,
    supportsWordTimestamps: false,
    supportsTokenTimestamps: true,
    supportsSegmentTimestamps: true,
    supportsConfidence: true,
    defaultSegmentationStrategy: 'word-punctuation',
    defaultMergeStrategy: 'concat',
  };
}

function createCtcInferenceLimits(): ModelInferenceLimits {
  return {
    sampleRate: 16000,
    maxInputDurationSec: 60,
    recommendedWindowDurationSec: 30,
    minWindowDurationSec: 5,
    maxWindowDurationSec: 60,
    autoWindowThresholdSec: 60,
    defaultOverlapSec: 5,
    supportsWordTimestamps: false,
    supportsTokenTimestamps: false,
    supportsSegmentTimestamps: false,
    supportsConfidence: false,
    defaultSegmentationStrategy: 'ctc-frame',
    defaultMergeStrategy: 'ctc-collapse',
  };
}

function createWhisperInferenceLimits(): ModelInferenceLimits {
  return {
    sampleRate: 16000,
    maxInputDurationSec: 30,
    recommendedWindowDurationSec: 30,
    minWindowDurationSec: 5,
    maxWindowDurationSec: 30,
    autoWindowThresholdSec: 30,
    defaultStrideSec: 5,
    supportsWordTimestamps: true,
    supportsTokenTimestamps: true,
    supportsSegmentTimestamps: true,
    supportsConfidence: true,
    defaultSegmentationStrategy: 'whisper-token',
    defaultMergeStrategy: 'whisper-stride',
  };
}

function createParakeetDescriptors(): BuiltInModelDescriptor[] {
  return Object.entries(PARAKEET_MODELS).map(([modelId, config]) => {
    const manifest = resolveParakeetPresetManifest(modelId);
    const source = resolveParakeetArtifactSource(modelId);
    const warmupExpectedTexts =
      'warmupExpectedTexts' in config ? config.warmupExpectedTexts : undefined;
    const warmupRequiredKeywordGroups =
      'warmupRequiredKeywordGroups' in config ? config.warmupRequiredKeywordGroups : undefined;

    return {
      modelId,
      aliases: manifest?.aliases ?? [],
      preset: 'parakeet',
      displayName: config.displayName,
      description:
        manifest?.description ??
        `Parakeet ${config.topology === 'rnnt' ? 'RNNT' : 'TDT'} model preset.`,
      classification: manifest?.classification ?? {},
      languages: config.languages,
      defaultSourceLanguage: config.defaultLanguage,
      defaultTargetLanguage: config.defaultLanguage,
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: config.supportsWordTimestamps ?? true,
        supportsSegmentTimestamps: true,
        supportsPromptControls: false,
      },
      inference: createParakeetInferenceLimits(config),
      loading: {
        repoId: config.repoId,
        supportsHubSource: config.supportsHubSource ?? true,
        supportsLocalSource: true,
        supportsSplitQuantization: true,
        supportsSplitBackends: true,
        availableEncoderBackends: ['webgpu', 'wasm'],
        availableDecoderBackends: ['wasm'],
        defaultEncoderBackend: 'webgpu',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp16', 'int8', 'fp32'],
        availableDecoderQuantizations: ['fp16', 'int8', 'fp32'],
        defaultEncoderQuantization: 'fp16',
        defaultDecoderQuantization: 'int8',
        supportsPreprocessorBackend: true,
        preprocessorBackends: ['js', 'onnx'],
        defaultPreprocessorBackend: source?.preprocessorBackend ?? 'js',
        defaultPreprocessorName: config.preprocessor,
        encoderArtifactBaseName: 'encoder-model',
        decoderArtifactBaseName: 'decoder_joint-model',
        defaultRevision: config.defaultRevision ?? 'main',
      },
      controls: [],
      warmup: {
        mode: 'expected-text',
        expectedTexts: warmupExpectedTexts ?? buildWarmupExpectedTexts(),
        requiredKeywordGroups: warmupRequiredKeywordGroups,
      },
    };
  });
}

function createCanaryDescriptors(): BuiltInModelDescriptor[] {
  return Object.entries(CANARY_MODELS).map(([modelId, config]) => {
    const manifest = resolveCanaryPresetManifest(modelId);
    const source = resolveCanaryArtifactSource(modelId);
    const docs = manifest?.docs;
    const supportsExtendedPrompt = manifest?.config.promptFormat === 'canary2';

    return {
      modelId,
      aliases: manifest?.aliases ?? [],
      preset: 'canary',
      displayName: config.displayName,
      description: docs?.architecture.summary ?? manifest?.description ?? config.displayName,
      classification: manifest?.classification ?? {},
      languages: config.languages,
      defaultSourceLanguage: config.defaultSourceLanguage,
      defaultTargetLanguage: config.defaultTargetLanguage,
      capabilities: {
        supportedTasks: ['asr', 'speech-translation'],
        supportsTranslation: true,
        supportsPunctuationCapitalization: true,
        supportsInverseTextNormalization: supportsExtendedPrompt,
        supportsWordTimestamps: docs?.capabilities.supportsWordTimestamps ?? false,
        supportsSegmentTimestamps: docs?.capabilities.supportsSegmentTimestamps ?? false,
        supportsPromptControls: true,
      },
      inference: createConservativeAedInferenceLimits(),
      loading: {
        repoId: config.repoId,
        supportsHubSource: source !== undefined,
        supportsLocalSource: true,
        supportsSplitQuantization: true,
        supportsSplitBackends: true,
        availableEncoderBackends: ['webgpu', 'wasm'],
        availableDecoderBackends: ['webgpu', 'wasm'],
        defaultEncoderBackend: 'webgpu',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp16', 'int8', 'fp32'],
        availableDecoderQuantizations: ['fp16', 'int8', 'fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'int8',
        supportsPreprocessorBackend: true,
        preprocessorBackends: ['js', 'onnx'],
        defaultPreprocessorBackend: source?.preprocessorBackend ?? 'js',
        defaultPreprocessorName: 'nemo128',
        encoderArtifactBaseName: 'encoder-model',
        decoderArtifactBaseName: 'decoder-model',
        defaultRevision: 'main',
      },
      controls: [
        {
          key: 'task',
          label: 'Task',
          description:
            'Controls whether the decoder performs same-language ASR or speech translation.',
          type: 'enum',
          defaultValue: 'asr',
          options: [
            { value: 'asr', label: 'ASR' },
            { value: 'translation', label: 'Speech Translation' },
          ],
        },
        {
          key: 'targetLanguage',
          label: 'Target Language',
          description: 'Target text language prompt token.',
          type: 'language',
          defaultValue: config.defaultTargetLanguage,
          visibleWhen: {
            key: 'task',
            notEquals: 'asr',
          },
        },
        {
          key: 'punctuate',
          label: 'Punctuation & Capitalization',
          description: 'Injects the Canary PnC prompt toggle.',
          type: 'boolean',
          defaultValue: true,
        },
        ...(supportsExtendedPrompt
          ? [{
              key: 'timestamps',
              label: 'Timestamps',
              description: 'Injects the Canary timestamp prompt toggle.',
              type: 'boolean' as const,
              defaultValue: false,
            }]
          : []),
      ],
      warmup: {
        mode: 'expected-text',
        expectedTexts: buildWarmupExpectedTexts(),
      },
      ...(docs
        ? {
            docs: {
              modelCardUrl: docs.modelCardUrl,
              architectureSummary: docs.architecture.summary,
              promptingNote: docs.prompting.note,
            },
          }
        : {}),
    };
  });
}

function createMedAsrDescriptors(): BuiltInModelDescriptor[] {
  const manifest = resolveMedAsrPresetManifest('google/medasr');
  const source = manifest?.source;
  return [
    {
      modelId: 'google/medasr',
      aliases: manifest?.aliases ?? [],
      preset: 'medasr',
      displayName: 'Google MedASR (Medical English)',
      description: manifest?.description ?? 'Google MedASR preset.',
      classification: manifest?.classification ?? {},
      languages: manifest?.config.languages ?? ['en'],
      defaultSourceLanguage: 'en',
      defaultTargetLanguage: 'en',
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: false,
        supportsSegmentTimestamps: false,
        supportsPromptControls: false,
      },
      inference: createCtcInferenceLimits(),
      loading: {
        repoId: source?.kind === 'huggingface' ? source.repoId : undefined,
        supportsHubSource: true,
        supportsLocalSource: true,
        supportsSplitQuantization: false,
        supportsSplitBackends: false,
        availableEncoderBackends: ['webgpu', 'wasm'],
        availableDecoderBackends: ['webgpu', 'wasm'],
        defaultEncoderBackend: 'webgpu',
        defaultDecoderBackend: 'webgpu',
        availableEncoderQuantizations: ['fp32'],
        availableDecoderQuantizations: ['fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'fp32',
        supportsPreprocessorBackend: false,
        preprocessorBackends: [],
        defaultPreprocessorBackend: null,
        defaultPreprocessorName: null,
        encoderArtifactBaseName: 'model',
        decoderArtifactBaseName: 'model',
        defaultRevision: 'main',
      },
      controls: [],
      warmup: {
        mode: 'non-empty',
      },
    },
  ];
}

function createParakeetCtcDescriptors(): BuiltInModelDescriptor[] {
  return [
    {
      modelId: 'ctaguchi/parakeet-hf-ctc-110m-en',
      aliases: ['parakeet-ctc-110m-en'],
      preset: 'parakeet-ctc',
      displayName: 'Parakeet HF CTC 110M (English)',
      description:
        'ParakeetForCTC student over the NeMo CTC boundary; local folder only because no validated Hub export exists.',
      classification: {
        ecosystem: 'nemo',
        processor: 'parakeet-mel-80',
        encoder: 'parakeet-ctc-encoder',
        decoder: 'ctc',
        topology: 'ctc',
        family: 'parakeet-ctc',
        task: 'asr',
      },
      languages: ['en'],
      defaultSourceLanguage: 'en',
      defaultTargetLanguage: 'en',
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsPromptControls: false,
      },
      inference: {
        ...createCtcInferenceLimits(),
        supportsWordTimestamps: true,
        supportsTokenTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsConfidence: true,
      },
      loading: {
        repoId: 'ctaguchi/parakeet-hf-ctc-110m-en',
        supportsHubSource: false,
        supportsLocalSource: true,
        supportsSplitQuantization: false,
        supportsSplitBackends: false,
        availableEncoderBackends: ['wasm', 'webgpu'],
        availableDecoderBackends: ['wasm'],
        defaultEncoderBackend: 'wasm',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp32'],
        availableDecoderQuantizations: ['fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'fp32',
        supportsPreprocessorBackend: true,
        preprocessorBackends: ['js'],
        defaultPreprocessorBackend: 'js',
        defaultPreprocessorName: 'parakeet-mel-80',
        encoderArtifactBaseName: 'model',
        decoderArtifactBaseName: 'model',
        defaultRevision: 'main',
      },
      controls: [],
      warmup: {
        mode: 'non-empty',
      },
      docs: {
        modelCardUrl: 'https://huggingface.co/ctaguchi/parakeet-hf-ctc-110m-en',
        architectureSummary:
          '16 kHz PCM → 80-bin Parakeet mel with 0.97 preemphasis and per-band normalization → ParakeetForCTC encoder → CTC greedy decode.',
      },
    },
  ];
}

function createOmnilingualCtcDescriptors(): BuiltInModelDescriptor[] {
  return [
    {
      modelId: 'OpenVoiceOS/omnilingual-asr-ctc-300m-onnx',
      aliases: ['omnilingual-ctc-300m', 'omni-asr-ctc-300m'],
      preset: 'omnilingual-ctc',
      displayName: 'Omnilingual ASR CTC 300M (1,600+ languages)',
      description:
        'Meta omniASR CTC line as an ONNX repack; raw-waveform wav2vec2-style CTC with no mel frontend.',
      classification: {
        ecosystem: 'wav2vec2',
        processor: 'raw-waveform-conv',
        encoder: 'wav2vec2-conv',
        decoder: 'ctc',
        topology: 'ctc',
        family: 'omnilingual-ctc',
        task: 'asr',
      },
      languages: ['mul'],
      defaultSourceLanguage: 'en',
      defaultTargetLanguage: 'en',
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsPromptControls: false,
      },
      inference: {
        ...createCtcInferenceLimits(),
        supportsWordTimestamps: true,
        supportsTokenTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsConfidence: true,
      },
      loading: {
        repoId: 'OpenVoiceOS/omnilingual-asr-ctc-300m-onnx',
        supportsHubSource: false,
        supportsLocalSource: true,
        supportsSplitQuantization: false,
        supportsSplitBackends: false,
        availableEncoderBackends: ['wasm', 'webgpu'],
        availableDecoderBackends: ['wasm'],
        defaultEncoderBackend: 'wasm',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['int8'],
        availableDecoderQuantizations: ['int8'],
        defaultEncoderQuantization: 'int8',
        defaultDecoderQuantization: 'int8',
        supportsPreprocessorBackend: false,
        preprocessorBackends: [],
        defaultPreprocessorBackend: null,
        defaultPreprocessorName: null,
        encoderArtifactBaseName: 'model',
        decoderArtifactBaseName: 'model',
        defaultRevision: 'main',
      },
      controls: [],
      warmup: { mode: 'non-empty' },
      docs: {
        modelCardUrl: 'https://huggingface.co/OpenVoiceOS/omnilingual-asr-ctc-300m-onnx',
        architectureSummary:
          'Raw 16 kHz waveform, zero-mean unit-variance, convolutional frontend inside the graph, CTC greedy decode over a 10288-token vocabulary.',
      },
    },
  ];
}

function createNewLocalAsrDescriptors(): BuiltInModelDescriptor[] {
  const commonCapabilities = {
    supportedTasks: ['asr'] as const,
    supportsTranslation: false,
    supportsPunctuationCapitalization: false,
    supportsInverseTextNormalization: false,
    supportsWordTimestamps: false,
    supportsSegmentTimestamps: false,
    supportsPromptControls: false,
  };
  const localLoading = (artifact: string) => ({
    supportsHubSource: false,
    supportsLocalSource: true,
    supportsSplitQuantization: false,
    supportsSplitBackends: false,
    availableEncoderBackends: ['wasm', 'webgpu'] as const,
    availableDecoderBackends: ['wasm'] as const,
    defaultEncoderBackend: 'wasm' as const,
    defaultDecoderBackend: 'wasm' as const,
    availableEncoderQuantizations: ['fp32'] as const,
    availableDecoderQuantizations: ['fp32'] as const,
    defaultEncoderQuantization: 'fp32' as const,
    defaultDecoderQuantization: 'fp32' as const,
    supportsPreprocessorBackend: false,
    preprocessorBackends: [] as const,
    defaultPreprocessorBackend: null,
    defaultPreprocessorName: null,
    encoderArtifactBaseName: artifact,
    decoderArtifactBaseName: artifact,
    defaultRevision: 'local',
  });
  const localInference = {
    sampleRate: 16000,
    maxInputDurationSec: 60,
    recommendedWindowDurationSec: 30,
    minWindowDurationSec: 1,
    maxWindowDurationSec: 60,
    autoWindowThresholdSec: 60,
    supportsWordTimestamps: false,
    supportsTokenTimestamps: false,
    supportsSegmentTimestamps: false,
    supportsConfidence: false,
    defaultSegmentationStrategy: 'ctc-frame' as const,
    defaultMergeStrategy: 'ctc-collapse' as const,
  };
  return [
    {
      status: 'experimental', modelId: 'gigaam-multilingual-ctc', aliases: [], preset: 'gigaam-ctc',
      displayName: 'GigaAM Multilingual CTC',
      description: 'Official GigaAM multilingual CTC graph; local artifacts only.',
      classification: { ecosystem: 'gigaam', processor: 'gigaam-fbank', encoder: 'conformer', decoder: 'ctc', topology: 'ctc', family: 'gigaam-ctc', task: 'asr' },
      languages: ['ru', 'en', 'kk', 'ky', 'uz'], defaultSourceLanguage: 'ru',
      capabilities: commonCapabilities,
      inference: { ...localInference, supportsTokenTimestamps: true, supportsConfidence: true },
      loading: localLoading('multilingual_ctc'), controls: [], warmup: { mode: 'non-empty' },
      docs: { architectureSummary: 'Official GigaAM multilingual CTC encoder with GigaAM fbank features.' },
    },
    {
      status: 'experimental', modelId: 'gigaam-v3-e2e-rnnt', aliases: [], preset: 'gigaam-rnnt',
      displayName: 'GigaAM v3 E2E RNN-T (Russian)',
      description: 'Official Russian GigaAM v3 RNN-T graphs; local artifacts only.',
      classification: { ecosystem: 'gigaam', processor: 'gigaam-fbank', encoder: 'conformer', decoder: 'rnnt', topology: 'rnnt', family: 'gigaam-rnnt', task: 'asr' },
      languages: ['ru'], defaultSourceLanguage: 'ru',
      capabilities: commonCapabilities,
      inference: { ...localInference, maxInputDurationSec: 30, recommendedWindowDurationSec: 15, maxWindowDurationSec: 30, autoWindowThresholdSec: 30, defaultSegmentationStrategy: 'vad', defaultMergeStrategy: 'concat' },
      loading: { ...localLoading('v3_e2e_rnnt_encoder'), supportsSplitBackends: true, availableDecoderBackends: ['wasm', 'webgpu'] as const },
      controls: [], warmup: { mode: 'non-empty' },
      docs: { architectureSummary: 'Russian-only official GigaAM v3 encoder, predictor and joint graphs.' },
    },
    {
      status: 'experimental', modelId: 'FunAudioLLM/SenseVoiceSmall', aliases: ['SenseVoiceSmall'], preset: 'sensevoice',
      displayName: 'SenseVoice Small',
      description: 'Official FunAudioLLM SenseVoiceSmall ONNX export; local artifacts only.',
      classification: { ecosystem: 'funasr', processor: 'kaldi-fbank', encoder: 'conformer', decoder: 'ctc', topology: 'ctc', family: 'sensevoice', task: 'asr' },
      languages: ['auto', 'zh', 'en', 'yue', 'ja', 'ko'], defaultSourceLanguage: 'auto',
      capabilities: commonCapabilities,
      inference: { ...localInference, maxInputDurationSec: 30, maxWindowDurationSec: 30, autoWindowThresholdSec: 30 },
      loading: { ...localLoading('model'), supportsPreprocessorBackend: true, preprocessorBackends: ['js'] as const, defaultPreprocessorBackend: 'js' as const },
      controls: [], warmup: { mode: 'non-empty' },
      docs: { architectureSummary: 'Official SenseVoice CTC graph with Kaldi fbank, LFR and CMVN features.' },
    },
    {
      status: 'experimental', modelId: 'x-asr-zh-en-160ms', aliases: [], preset: 'x-asr',
      displayName: 'X-ASR zh-en 160 ms Streaming',
      description: 'Experimental local-only Zipformer2 transducer with native encoder-cache streaming.',
      classification: { ecosystem: 'x-asr', processor: 'kaldi-fbank', encoder: 'zipformer2', decoder: 'stateless-rnnt', topology: 'stateless-rnnt', family: 'x-asr', task: 'asr' },
      languages: ['zh', 'en'], defaultSourceLanguage: 'zh',
      capabilities: commonCapabilities,
      inference: { ...localInference, defaultSegmentationStrategy: 'none', defaultMergeStrategy: 'concat' },
      loading: {
        ...localLoading('encoder-160ms'),
        availableDecoderBackends: ['wasm', 'webgpu'] as const,
        supportsPreprocessorBackend: true,
        preprocessorBackends: ['js'] as const,
        defaultPreprocessorBackend: 'js' as const,
        decoderArtifactBaseName: 'decoder-160ms',
      },
      controls: [], warmup: { mode: 'non-empty' },
      docs: { architectureSummary: 'Native 160 ms cache-aware Zipformer2 encoder streaming with stateless RNN-T decoding; graph cache contract is provided by the family default config.' },
    },
  ];
}

function createTurkishWav2Vec2Descriptors(): BuiltInModelDescriptor[] {
  return [
    {
      modelId: 'mpoyraz/wav2vec2-xls-r-300m-cv7-turkish',
      aliases: ['wav2vec2-xls-r-300m-cv7-turkish', 'wav2vec2-turkish'],
      preset: 'wav2vec2-tr',
      displayName: 'Wav2Vec2 XLS-R 300M (Turkish Common Voice v7)',
      description:
        'Turkish character CTC fine-tune of wav2vec2-large-xls-r, exported to ONNX from official weights.',
      classification: {
        ecosystem: 'wav2vec2',
        processor: 'raw-waveform-conv',
        encoder: 'wav2vec2-xls-r',
        decoder: 'ctc',
        topology: 'ctc',
        family: 'wav2vec2',
        task: 'asr',
      },
      languages: ['tr'],
      defaultSourceLanguage: 'tr',
      defaultTargetLanguage: 'tr',
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsPromptControls: false,
      },
      inference: {
        ...createCtcInferenceLimits(),
        supportsWordTimestamps: true,
        supportsTokenTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsConfidence: true,
      },
      loading: {
        repoId: 'mpoyraz/wav2vec2-xls-r-300m-cv7-turkish',
        supportsHubSource: false,
        supportsLocalSource: true,
        supportsSplitQuantization: false,
        supportsSplitBackends: false,
        availableEncoderBackends: ['wasm', 'webgpu'],
        availableDecoderBackends: ['wasm'],
        defaultEncoderBackend: 'wasm',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp32'],
        availableDecoderQuantizations: ['fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'fp32',
        supportsPreprocessorBackend: false,
        preprocessorBackends: [],
        defaultPreprocessorBackend: null,
        defaultPreprocessorName: null,
        encoderArtifactBaseName: 'model',
        decoderArtifactBaseName: 'model',
        defaultRevision: 'main',
      },
      controls: [],
      warmup: { mode: 'non-empty' },
      docs: {
        modelCardUrl: 'https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv7-turkish',
        architectureSummary:
          'Raw 16 kHz waveform, zero-mean unit-variance, convolutional frontend in graph, character CTC greedy decode over a 38-token vocabulary with | as the word separator.',
      },
    },
  ];
}

function createQwen3AsrDescriptors(): BuiltInModelDescriptor[] {
  return [
    {
      modelId: 'Qwen/Qwen3-ASR-0.6B-hf',
      aliases: ['qwen3-asr-0.6b'],
      preset: 'qwen3-asr',
      displayName: 'Qwen3-ASR 0.6B (multilingual, language-tagged)',
      description:
        'Conditional speech recognizer over a 128-bin mel frontend, with a 28-layer autoregressive decoder and language identification.',
      classification: {
        ecosystem: 'qwen',
        processor: 'mel-128',
        encoder: 'qwen3-asr-audio-tower',
        decoder: 'aed',
        topology: 'aed',
        family: 'qwen3-asr',
        task: 'asr',
      },
      languages: ['multilingual'],
      defaultSourceLanguage: 'en',
      defaultTargetLanguage: 'en',
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: false,
        supportsSegmentTimestamps: false,
        supportsPromptControls: false,
      },
      inference: {
        ...createCtcInferenceLimits(),
        supportsWordTimestamps: false,
        supportsTokenTimestamps: false,
        supportsSegmentTimestamps: false,
        supportsConfidence: false,
      },
      loading: {
        repoId: 'Qwen/Qwen3-ASR-0.6B-hf',
        supportsHubSource: false,
        supportsLocalSource: true,
        supportsSplitQuantization: false,
        supportsSplitBackends: true,
        availableEncoderBackends: ['wasm', 'webgpu'],
        availableDecoderBackends: ['wasm', 'webgpu'],
        defaultEncoderBackend: 'wasm',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp32'],
        availableDecoderQuantizations: ['fp16', 'fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'fp32',
        supportsPreprocessorBackend: false,
        preprocessorBackends: [],
        defaultPreprocessorBackend: null,
        defaultPreprocessorName: null,
        encoderArtifactBaseName: 'audio-encoder-dynamic',
        decoderArtifactBaseName: 'decoder-prefill',
        defaultRevision: 'main',
      },
      controls: [],
      warmup: { mode: 'expected-text' },
      docs: {
        modelCardUrl: 'https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf',
        architectureSummary:
          '128-bin mel at n_fft 400 and hop 160, an audio tower plus multimodal projector emitting one embedding per frame, then a 28-layer grouped-query decoder with a stacked KV cache. Output is a language tag, the language name, an asr_text marker, then the words.',
      },
    },
  ];
}

function createMoonshineDescriptors(): BuiltInModelDescriptor[] {
  return [
    {
      modelId: 'UsefulSensors/moonshine-tiny',
      aliases: ['moonshine-tiny'],
      preset: 'moonshine',
      displayName: 'Moonshine Tiny (English streaming-capable AED)',
      description:
        'English AED model over raw 16 kHz audio; conv subsampler plus Conformer encoder with rotary embeddings and an autoregressive decoder.',
      classification: {
        ecosystem: 'moonshine',
        processor: 'raw-waveform-conv',
        encoder: 'moonshine-conformer',
        decoder: 'aed',
        topology: 'aed',
        family: 'moonshine',
        task: 'asr',
      },
      languages: ['en'],
      defaultSourceLanguage: 'en',
      defaultTargetLanguage: 'en',
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsPromptControls: false,
      },
      inference: {
        ...createCtcInferenceLimits(),
        supportsWordTimestamps: true,
        supportsTokenTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsConfidence: true,
      },
      loading: {
        repoId: 'UsefulSensors/moonshine-tiny',
        supportsHubSource: false,
        supportsLocalSource: true,
        supportsSplitQuantization: false,
        supportsSplitBackends: false,
        availableEncoderBackends: ['wasm', 'webgpu'],
        availableDecoderBackends: ['wasm'],
        defaultEncoderBackend: 'wasm',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp32'],
        availableDecoderQuantizations: ['fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'fp32',
        supportsPreprocessorBackend: false,
        preprocessorBackends: [],
        defaultPreprocessorBackend: null,
        defaultPreprocessorName: null,
        encoderArtifactBaseName: 'encoder_model',
        decoderArtifactBaseName: 'decoder_with_past_model',
        defaultRevision: 'main',
      },
      controls: [],
      warmup: { mode: 'expected-text' },
      docs: {
        modelCardUrl: 'https://huggingface.co/UsefulSensors/moonshine-tiny',
        architectureSummary:
          'Raw 16 kHz samples, conv subsampler in graph, Conformer encoder with rotary embeddings, autoregressive Transformer decoder with self-attention history.',
      },
    },
    {
      status: 'experimental',
      modelId: 'moonshine-ai/moonshine-streaming-tiny',
      aliases: ['moonshine-streaming-tiny'],
      preset: 'moonshine',
      displayName: 'Moonshine Tiny Streaming',
      description: 'Local-only Moonshine Tiny export with native streaming graphs.',
      classification: {
        ecosystem: 'moonshine',
        processor: 'raw-waveform-conv',
        encoder: 'moonshine-conformer',
        decoder: 'aed',
        topology: 'aed',
        family: 'moonshine',
        task: 'asr',
      },
      languages: ['en'],
      defaultSourceLanguage: 'en',
      defaultTargetLanguage: 'en',
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: false,
        supportsSegmentTimestamps: false,
        supportsPromptControls: false,
      },
      inference: {
        ...createCtcInferenceLimits(),
        supportsWordTimestamps: false,
        supportsTokenTimestamps: false,
        supportsSegmentTimestamps: false,
        supportsConfidence: false,
        defaultSegmentationStrategy: 'word-punctuation',
        defaultMergeStrategy: 'concat',
      },
      loading: {
        supportsHubSource: false,
        supportsLocalSource: true,
        supportsSplitQuantization: false,
        supportsSplitBackends: false,
        availableEncoderBackends: ['wasm', 'webgpu'],
        availableDecoderBackends: ['wasm', 'webgpu'],
        defaultEncoderBackend: 'wasm',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp32'],
        availableDecoderQuantizations: ['fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'fp32',
        supportsPreprocessorBackend: false,
        preprocessorBackends: [],
        defaultPreprocessorBackend: null,
        defaultPreprocessorName: null,
        encoderArtifactBaseName: 'encoder_model',
        decoderArtifactBaseName: 'decoder_step_model',
        defaultRevision: 'local',
      },
      controls: [],
      warmup: { mode: 'expected-text' },
      docs: {
        architectureSummary:
          'Local four-graph Moonshine export. The same selected backend runs the whole and chunked encoders plus decoder prefill and step graphs; streaming is native encoder-cache streaming.',
      },
    },
  ];
}

function createWav2Vec2Descriptors(): BuiltInModelDescriptor[] {
  const manifest = resolveWav2Vec2PresetManifest('facebook/wav2vec2-base-960h');
  return [
    {
      modelId: 'facebook/wav2vec2-base-960h',
      aliases: manifest?.aliases ?? [],
      preset: 'wav2vec2',
      displayName: 'Wav2Vec2 Base 960h',
      description: manifest?.description ?? 'Wav2Vec2 CTC preset.',
      classification: manifest?.classification ?? {},
      languages: manifest?.config.languages ?? ['en'],
      defaultSourceLanguage: 'en',
      defaultTargetLanguage: 'en',
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsPromptControls: false,
      },
      inference: {
        ...createCtcInferenceLimits(),
        supportsWordTimestamps: true,
        supportsTokenTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsConfidence: true,
      },
      loading: {
        repoId: manifest?.source?.kind === 'huggingface' ? manifest.source.repoId : undefined,
        supportsHubSource: manifest?.source?.kind === 'huggingface' ? true : false,
        supportsLocalSource: true,
        supportsSplitQuantization: false,
        supportsSplitBackends: false,
        availableEncoderBackends: ['wasm', 'webgpu'],
        availableDecoderBackends: ['wasm', 'webgpu'],
        defaultEncoderBackend: 'wasm',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp32'],
        availableDecoderQuantizations: ['fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'fp32',
        supportsPreprocessorBackend: false,
        preprocessorBackends: [],
        defaultPreprocessorBackend: null,
        defaultPreprocessorName: null,
        encoderArtifactBaseName: 'model',
        decoderArtifactBaseName: 'model',
        defaultRevision: 'main',
      },
      controls: [],
      warmup: {
        mode: 'non-empty',
      },
      docs: {
        modelCardUrl: 'https://huggingface.co/facebook/wav2vec2-base-960h',
        architectureSummary: 'Raw waveform → Wav2Vec2 encoder → CTC greedy decode.',
      },
    },
  ];
}

function createWhisperDescriptors(): BuiltInModelDescriptor[] {
  return [
    'openai/whisper-base',
    'openai/whisper-large-v3',
    'ysdede/whisper-large-v3-turbo-onnx-4graph',
  ].map((modelId) => {
    const manifest = resolveWhisperPresetManifest(modelId);
    const isTurbo4Graph = modelId === 'ysdede/whisper-large-v3-turbo-onnx-4graph';
    const isLocal4Graph = manifest?.localGeometry !== undefined;
    return {
      modelId,
      aliases: manifest?.aliases ?? [],
      preset: 'whisper',
      displayName: isTurbo4Graph
        ? 'Whisper Large v3 Turbo (4-graph ONNX)'
        : modelId === 'openai/whisper-base'
          ? 'Whisper Base'
          : 'Whisper Large v3',
      description: manifest?.description ?? 'Whisper preset.',
      classification: manifest?.classification ?? {},
      languages: manifest?.config.languages ?? ['auto'],
      defaultSourceLanguage: 'auto',
      defaultTargetLanguage: 'auto',
      capabilities: {
        supportedTasks: ['asr', 'speech-translation'],
        supportsTranslation: true,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsPromptControls: false,
      },
      inference: createWhisperInferenceLimits(),
      loading: {
        supportsHubSource: manifest?.source !== undefined,
        supportsLocalSource: isLocal4Graph,
        supportsSplitQuantization: false,
        supportsSplitBackends: false,
        availableEncoderBackends: isLocal4Graph ? ['webgpu'] : ['webgpu', 'wasm'],
        availableDecoderBackends: isLocal4Graph ? ['webgpu'] : ['webgpu', 'wasm'],
        defaultEncoderBackend: 'webgpu',
        defaultDecoderBackend: 'webgpu',
        availableEncoderQuantizations: isLocal4Graph ? ['fp16'] : ['fp32'],
        availableDecoderQuantizations: isLocal4Graph ? ['fp16'] : ['fp32'],
        defaultEncoderQuantization: isLocal4Graph ? 'fp16' : 'fp32',
        defaultDecoderQuantization: isLocal4Graph ? 'fp16' : 'fp32',
        supportsPreprocessorBackend: false,
        preprocessorBackends: [],
        defaultPreprocessorBackend: null,
        defaultPreprocessorName: null,
        defaultRevision: 'main',
      },
      controls: [],
      warmup: {
        mode: 'non-empty',
      },
    };
  });
}

function createNemotronInferenceLimits(): ModelInferenceLimits {
  return {
    sampleRate: 16000,
    maxInputDurationSec: 3600,
    recommendedWindowDurationSec: 30,
    minWindowDurationSec: 5,
    maxWindowDurationSec: 60,
    autoWindowThresholdSec: 30,
    defaultOverlapSec: 2,
    preferVadSegmentWindowing: true,
    supportsWordTimestamps: true,
    supportsTokenTimestamps: true,
    supportsSegmentTimestamps: true,
    supportsConfidence: true,
    defaultSegmentationStrategy: 'vad',
    defaultMergeStrategy: 'concat',
  };
}

function createNemotronDescriptors(): BuiltInModelDescriptor[] {
  const modelId = 'nemotron-3.5-asr-streaming-0.6b';
  const manifest = resolveNemotronPresetManifest(modelId);
  const source = resolveNemotronArtifactSource(modelId);

  return [
    {
      modelId,
      aliases: manifest?.aliases ?? [],
      preset: 'nemotron',
      displayName: 'Nemotron 3.5 ASR Streaming 0.6B',
      description:
        manifest?.description ??
        'Nemotron streaming RNNT preset (cache-aware encoder, prompt tokens).',
      classification: manifest?.classification ?? {},
      languages: ['en', 'tr', 'auto'],
      defaultSourceLanguage: 'auto',
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: true,
        supportsSegmentTimestamps: true,
        supportsPromptControls: false,
      },
      inference: createNemotronInferenceLimits(),
      loading: {
        repoId: source?.kind === 'huggingface' ? source.repoId : undefined,
        supportsHubSource: true,
        supportsLocalSource: true,
        supportsSplitQuantization: false,
        supportsSplitBackends: true,
        availableEncoderBackends: ['webgpu', 'wasm'],
        availableDecoderBackends: ['wasm', 'webgpu'],
        defaultEncoderBackend: 'webgpu',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp16', 'fp32'],
        availableDecoderQuantizations: ['fp16', 'fp32'],
        defaultEncoderQuantization: 'fp16',
        defaultDecoderQuantization: 'fp16',
        supportsPreprocessorBackend: true,
        preprocessorBackends: ['js'],
        defaultPreprocessorBackend: 'js',
        defaultPreprocessorName: null,
        defaultRevision: 'main',
      },
      controls: [],
      warmup: {
        mode: 'non-empty',
      },
      docs: {
        modelCardUrl: 'https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b',
        architectureSummary:
          'FastConformer cache-aware streaming encoder (80/160/320/560/1120 ms windows) + LSTM predictor + joint; own fp16 ONNX is the default, with fp32 export parity; prompt ids auto=101/en=0/tr=18.',
      },
    },
  ];
}

export const BUILT_IN_MODEL_DESCRIPTORS = [
  ...createParakeetDescriptors(),
  ...createCanaryDescriptors(),
  ...createMedAsrDescriptors(),
  ...createNemotronDescriptors(),
  ...createWav2Vec2Descriptors(),
  ...createParakeetCtcDescriptors(),
  ...createOmnilingualCtcDescriptors(),
  ...createNewLocalAsrDescriptors(),
  ...createTurkishWav2Vec2Descriptors(),
  ...createMoonshineDescriptors(),
  ...createQwen3AsrDescriptors(),
  ...createWhisperDescriptors(),
] as const;
