import type { ModelClassification } from '../../types/index.js';
import type {
  WhisperArtifactSource,
  WhisperSeq2SeqModelConfig,
} from '../../models/whisper-seq2seq/index.js';

export interface WhisperPresetManifest {
  readonly preset: 'whisper';
  readonly modelId: string;
  readonly aliases?: readonly string[];
  readonly description: string;
  readonly classification: ModelClassification;
  readonly config: Partial<WhisperSeq2SeqModelConfig>;
  readonly source?: WhisperArtifactSource;
}

const WHISPER_4GRAPH_REPO_ID = 'ysdede/whisper-large-v3-turbo-onnx-4graph';

function hfResolve(repoId: string, filename: string, revision = 'main'): string {
  const repoPath = repoId.split('/').map(encodeURIComponent).join('/');
  const encodedRevision = encodeURIComponent(revision);
  const filePath = filename.split('/').map(encodeURIComponent).join('/');
  return `https://huggingface.co/${repoPath}/resolve/${encodedRevision}/${filePath}`;
}

export const WHISPER_PRESET_MANIFESTS: readonly WhisperPresetManifest[] = [
  {
    preset: 'whisper',
    modelId: WHISPER_4GRAPH_REPO_ID,
    aliases: [
      'whisper-large-v3-turbo-onnx-4graph',
      'whisper-large-v3-turbo-4graph',
    ],
    description:
      'Whisper Large-v3 Turbo ONNX 4-graph split decoder port with KV-cache decode for WebGPU.',
    classification: {
      ecosystem: 'openai',
      processor: 'whisper-mel',
      encoder: 'whisper-transformer',
      decoder: 'transformer-decoder',
      topology: 'aed',
      family: 'whisper',
      task: 'multitask-asr-translation',
    },
    config: {
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      melBins: 128,
      vocabularySize: 51866,
      languages: ['auto'],
    },
    source: {
      kind: 'splitgraph',
      artifacts: {
        encoderUrl: hfResolve(WHISPER_4GRAPH_REPO_ID, 'fp16_iofp32/encoder_model.onnx'),
        decoderInitUrl: hfResolve(WHISPER_4GRAPH_REPO_ID, 'fp16/decoder_init.onnx'),
        decoderStepUrl: hfResolve(WHISPER_4GRAPH_REPO_ID, 'fp16/decoder_step.onnx'),
        decoderAlignUrl: hfResolve(WHISPER_4GRAPH_REPO_ID, 'fp16/decoder_align.onnx'),
        tokenizerUrl: hfResolve(WHISPER_4GRAPH_REPO_ID, 'fp16/tokenizer.json'),
        manifestUrl: hfResolve(WHISPER_4GRAPH_REPO_ID, 'fp16/manifest.json'),
        externalDataUrls: {
          encoder: [{ path: './encoder_model.onnx.data', file: 'encoder_model.onnx.data' }],
          decoder_init: [{ path: './decoder_init.onnx.data', file: 'decoder_init.onnx.data' }],
          decoder_step: [{ path: './decoder_step.onnx.data', file: 'decoder_step.onnx.data' }],
          decoder_align: [{ path: './decoder_align.onnx.data', file: 'decoder_align.onnx.data' }],
        },
      },
      encoderBackend: 'webgpu',
      decoderBackend: 'webgpu',
    },
  },
  {
    preset: 'whisper',
    modelId: 'onnx-community/whisper-tiny',
    aliases: ['whisper-tiny', 'openai/whisper-tiny'],
    description:
      'Whisper Tiny multilingual preset. ~39M params. Fastest, lowest memory. Good for smoke tests and low-resource devices.',
    classification: {
      ecosystem: 'openai',
      processor: 'whisper-mel',
      encoder: 'whisper-transformer',
      decoder: 'transformer-decoder',
      topology: 'aed',
      family: 'whisper',
      task: 'multitask-asr-translation',
    },
    config: {
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      melBins: 80,
      vocabularySize: 51865,
      languages: ['auto'],
    },
    source: {
      kind: 'huggingface',
      repoId: 'onnx-community/whisper-tiny_timestamped',
    },
  },
  {
    preset: 'whisper',
    modelId: 'onnx-community/whisper-base',
    aliases: ['whisper-base', 'openai/whisper-base', 'whisper-base-multilingual'],
    description:
      'Whisper Base multilingual preset. ~74M params. Default baseline for browser Whisper. Good balance of quality and speed.',
    classification: {
      ecosystem: 'openai',
      processor: 'whisper-mel',
      encoder: 'whisper-transformer',
      decoder: 'transformer-decoder',
      topology: 'aed',
      family: 'whisper',
      task: 'multitask-asr-translation',
    },
    config: {
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      melBins: 80,
      vocabularySize: 51865,
      languages: ['auto'],
    },
    source: {
      kind: 'huggingface',
      repoId: 'onnx-community/whisper-base_timestamped',
    },
  },
  {
    preset: 'whisper',
    modelId: 'onnx-community/whisper-small',
    aliases: ['whisper-small', 'openai/whisper-small'],
    description:
      'Whisper Small multilingual preset. ~244M params. Better quality than base, heavier. Use when accuracy matters more than speed.',
    classification: {
      ecosystem: 'openai',
      processor: 'whisper-mel',
      encoder: 'whisper-transformer',
      decoder: 'transformer-decoder',
      topology: 'aed',
      family: 'whisper',
      task: 'multitask-asr-translation',
    },
    config: {
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      melBins: 80,
      vocabularySize: 51865,
      languages: ['auto'],
    },
    source: {
      kind: 'huggingface',
      repoId: 'onnx-community/whisper-small_timestamped',
    },
  },
  {
    preset: 'whisper',
    modelId: 'onnx-community/whisper-large-v3-turbo',
    aliases: ['whisper-large-v3-turbo', 'openai/whisper-large-v3-turbo'],
    description:
      'Whisper Large-v3 Turbo multilingual preset. ~809M params. Experimental desktop/WebGPU only. Do not use on mobile or low-memory devices.',
    classification: {
      ecosystem: 'openai',
      processor: 'whisper-mel',
      encoder: 'whisper-transformer',
      decoder: 'transformer-decoder',
      topology: 'aed',
      family: 'whisper',
      task: 'multitask-asr-translation',
    },
    config: {
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      melBins: 128,
      vocabularySize: 51866,
      languages: ['auto'],
    },
    source: {
      kind: 'huggingface',
      repoId: 'onnx-community/whisper-large-v3-turbo_timestamped',
    },
  },
];

function normalizePresetId(modelId: string): string {
  return modelId.trim().toLowerCase();
}

export function resolveWhisperPresetManifest(modelId: string): WhisperPresetManifest | undefined {
  const normalizedModelId = normalizePresetId(modelId);

  return WHISPER_PRESET_MANIFESTS.find((manifest) => {
    if (normalizePresetId(manifest.modelId) === normalizedModelId) {
      return true;
    }
    return (manifest.aliases ?? []).some((alias) => normalizePresetId(alias) === normalizedModelId);
  });
}

export function resolveWhisperArtifactSource(
  modelId: string,
): WhisperArtifactSource | undefined {
  return resolveWhisperPresetManifest(modelId)?.source;
}
