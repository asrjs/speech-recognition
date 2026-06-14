/**
 * Self-exported 4-graph Whisper local-file loader.
 *
 * Reads a directory containing the whisper-browser-self-export-v1 format:
 *   manifest.json, encoder_model.onnx, decoder_init.onnx,
 *   decoder_step.onnx, decoder_align.onnx, tokenizer files
 *
 * Returns a WhisperArtifactSource + WhisperSeq2SeqModelConfig ready for
 * createWhisperSeq2SeqModelFamily().createModel().
 */

import * as fs from 'fs';
import * as path from 'path';
import { pathToFileURL } from 'url';
import { parseWhisperManifest } from './manifest.js';
import type {
  ExternalDataEntry,
  WhisperArtifactSource,
  WhisperSeq2SeqModelConfig,
  WhisperSplitGraphArtifacts,
} from './types.js';

export interface SplitGraphLocalModel {
  readonly source: WhisperArtifactSource;
  readonly config: WhisperSeq2SeqModelConfig;
  readonly modelId: string;
}

export interface SplitGraphLocalOptions {
  /** Variant subdirectory to load (e.g., 'fp32', 'fp16', 'int8-dynamic').
   *  Defaults to 'fp32'.  Set to null to load from the given directory
   *  without variant subdirectory resolution (flat layout). */
  readonly variant?: string | null;
}

const DEFAULT_VARIANT = 'fp32';

type GraphName = keyof NonNullable<WhisperSplitGraphArtifacts['externalDataUrls']>;

const MANIFEST_GRAPH_KEYS: readonly GraphName[] = [
  'encoder',
  'decoder_init',
  'decoder_step',
  'decoder_align',
];

function readExternalDataUrls(
  manifestRaw: Record<string, unknown>,
): WhisperSplitGraphArtifacts['externalDataUrls'] | undefined {
  const artifacts = manifestRaw.artifacts;
  if (!artifacts || typeof artifacts !== 'object') return undefined;

  const result: Partial<Record<GraphName, readonly ExternalDataEntry[]>> = {};
  for (const graphName of MANIFEST_GRAPH_KEYS) {
    const graph = (artifacts as Record<string, unknown>)[graphName];
    if (!graph || typeof graph !== 'object') continue;
    const externalData = (graph as Record<string, unknown>).externalData;
    if (!Array.isArray(externalData) || externalData.length === 0) continue;

    const entries = externalData
      .filter((entry): entry is Record<string, unknown> => Boolean(entry) && typeof entry === 'object')
      .map((entry) => ({
        path: String(entry.path ?? entry.file ?? ''),
        file: String(entry.file ?? entry.path ?? ''),
        sizeBytes: typeof entry.sizeBytes === 'number' ? entry.sizeBytes : undefined,
        sha256: typeof entry.sha256 === 'string' ? entry.sha256 : undefined,
      }))
      .filter((entry) => entry.path.length > 0 && entry.file.length > 0);

    if (entries.length > 0) {
      result[graphName] = entries;
    }
  }

  return Object.keys(result).length > 0 ? result : undefined;
}

function readWhisperFeatureFrameCount(manifestRaw: Record<string, unknown>): number {
  const maxSourcePositions = (manifestRaw.max_source_positions as number) ?? 3000;
  // HF/OpenAI config max_source_positions is encoder time positions after the
  // conv downsample (1500 for 30s Whisper). The ONNX encoder input is mel
  // frames before downsample, so browser/WASM local comparison must feed 3000.
  return maxSourcePositions <= 1500 ? maxSourcePositions * 2 : maxSourcePositions;
}

export function loadSplitGraphLocalModel(
  dirPath: string,
  options: SplitGraphLocalOptions = {},
): SplitGraphLocalModel {
  const variant = options.variant === undefined ? DEFAULT_VARIANT : options.variant;
  let resolved = path.resolve(dirPath);

  // If variant is specified, resolve into the variant subdirectory
  if (variant !== null && variant !== '') {
    const variantDir = path.join(resolved, variant);
    if (fs.existsSync(variantDir) && fs.existsSync(path.join(variantDir, 'manifest.json'))) {
      resolved = variantDir;
    } else if (options.variant !== undefined) {
      // User explicitly requested a variant that doesn't exist
      const available = ['fp32', 'fp16', 'q8']
        .filter((v) => fs.existsSync(path.join(resolved, v, 'manifest.json')));
      throw new Error(
        `Variant "${variant}" not found in ${resolved}. ` +
        `Available variants: ${available.length > 0 ? available.join(', ') : 'none'}. ` +
        `Tip: export with --output-layout variant-dirs to create variant subdirectories.`,
      );
    }
  }

  const manifestPath = path.join(resolved, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`manifest.json not found in ${resolved}. Is this a self-exported Whisper directory?`);
  }

  const manifestRaw = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as Record<string, unknown>;
  const manifest = parseWhisperManifest(manifestRaw);

  // Warn on variant selection based on validation status
  const compat = (manifestRaw.runtime_compatibility ?? {}) as Record<string, Record<string, unknown>>;
  if (variant && compat[variant]) {
    const notes = String(compat[variant]?.notes ?? '');

    // Check detailed validation fields first (new schema), fall back to legacy 'status'
    const validation = (compat[variant]?.validation ?? {}) as Record<string, string>;
    const runtime = (compat[variant]?.runtimeCompatibility ?? {}) as Record<string, string>;

    const isPending = (key: string) => (validation[key] ?? runtime[key]) === 'pending';
    const isNotRecommended = (key: string) => (validation[key] ?? runtime[key]) === 'not_recommended';

    if (isNotRecommended('browserWebGpu') || isNotRecommended('browserWasm')) {
      console.warn(
        `[whisper-seq2seq] Variant "${variant}" is not recommended for browser/WebGPU. ${notes}`,
      );
    } else if (isPending('browserWebGpu') || isPending('webGpuSmokeDecode')) {
      console.warn(
        `[whisper-seq2seq] Variant "${variant}" is native-validated but browser/WebGPU validation is pending. ${notes}`,
      );
    } else {
      // Legacy status check
      const status = String(compat[variant]?.status ?? '');
      if (status === 'requires_validation' || status === 'requires_export_time_fp16') {
        console.warn(
          `[whisper-seq2seq] Variant "${variant}": ${status}. ${notes}`,
        );
      }
    }
  }

  const modelId = manifest.modelId || path.basename(resolved);

  const fileUrl = (name: string): string => {
    const fullPath = path.join(resolved, name);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`Missing artifact: ${name} in ${resolved}`);
    }
    return pathToFileURL(fullPath).href;
  };

  const config: WhisperSeq2SeqModelConfig = {
    ecosystem: 'openai',
    architecture: 'whisper-seq2seq',
    processorArchitecture: 'whisper-mel',
    encoderArchitecture: 'whisper-transformer',
    decoderArchitecture: 'transformer-decoder',
    sampleRate: 16000,
    melBins: (manifestRaw.num_mel_bins as number) ?? 80,
    maxSourcePositions: readWhisperFeatureFrameCount(manifestRaw),
    maxTargetPositions: (manifestRaw.max_target_positions as number) ?? 448,
    vocabularySize: (manifestRaw.vocab_size as number) ?? 51865,
    languages: ['en'],
    tokenizer: {
      kind: 'tiktoken',
      vocabSize: (manifestRaw.vocab_size as number) ?? 51865,
    },
  };

  const source: WhisperArtifactSource = {
    kind: 'splitgraph',
    artifacts: {
      encoderUrl: fileUrl('encoder_model.onnx'),
      decoderInitUrl: fileUrl('decoder_init.onnx'),
      decoderStepUrl: fileUrl('decoder_step.onnx'),
      decoderAlignUrl: fs.existsSync(path.join(resolved, 'decoder_align.onnx'))
        ? fileUrl('decoder_align.onnx')
        : undefined,
      tokenizerUrl: fileUrl('tokenizer.json'),
      manifestUrl: fileUrl('manifest.json'),
      externalDataUrls: readExternalDataUrls(manifestRaw),
    },
  };

  return { source, config, modelId };
}
