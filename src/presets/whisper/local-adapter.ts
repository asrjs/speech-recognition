import { createBlobAssetProvider } from '../../io/index.js';
import type { WhisperSplitGraphArtifacts } from '../../models/whisper-seq2seq/index.js';
import { parseWhisperManifest } from '../../models/whisper-seq2seq/manifest.js';
import type {
  BuiltInLocalModelAdapter,
  LoadSpeechModelFromLocalEntriesOptions,
  ResolvedBuiltInLocalArtifacts,
  SpeechModelLocalInspection,
} from '../../runtime/local-adapter-registry.js';
import {
  collectLocalEntriesFromDirectory,
  createLocalEntriesFromFiles,
  getLocalEntryFile,
  normalizeLocalRelativePath,
} from '../../runtime/local-entry-files.js';
import type {
  SpeechModelLocalDirectoryHandleLike,
  SpeechModelLocalEntry,
} from '../../runtime/local-types.js';
import { resolveWhisperPresetManifest } from './manifest.js';

const REQUIRED_FILES = [
  'encoder_model.onnx',
  'decoder_init.onnx',
  'decoder_step.onnx',
  'decoder_align.onnx',
  'tokenizer.json',
  'manifest.json',
  'config.json',
  'generation_config.json',
] as const;
const GRAPH_FILES = {
  encoder: 'encoder_model.onnx',
  decoder_init: 'decoder_init.onnx',
  decoder_step: 'decoder_step.onnx',
  decoder_align: 'decoder_align.onnx',
} as const;

interface LocalWhisperBundle {
  readonly root: string;
  readonly entries: ReadonlyMap<(typeof REQUIRED_FILES)[number], SpeechModelLocalEntry>;
}

function localPath(entry: SpeechModelLocalEntry): string {
  return normalizeLocalRelativePath(entry.path).toLowerCase();
}

function resolveLocalBundle(entries: readonly SpeechModelLocalEntry[]): LocalWhisperBundle {
  const encoderPaths = entries
    .map(localPath)
    .filter((path) => path.split('/').pop() === 'encoder_model.onnx');
  if (encoderPaths.length === 0) {
    throw new Error('Missing encoder_model.onnx for the Whisper 4-graph bundle.');
  }

  const roots = [...new Set(encoderPaths.map((path) => path.split('/').slice(0, -1).join('/')))];
  const complete: LocalWhisperBundle[] = [];
  const incomplete: { root: string; missing: string[]; ambiguous: string[] }[] = [];
  for (const root of roots) {
    const files = new Map<(typeof REQUIRED_FILES)[number], SpeechModelLocalEntry>();
    const missing: string[] = [];
    const ambiguous: string[] = [];
    for (const name of REQUIRED_FILES) {
      const expectedPath = root ? `${root}/${name}` : name;
      const matches = entries.filter((entry) => localPath(entry) === expectedPath);
      if (matches.length === 0) missing.push(name);
      else if (matches.length > 1) ambiguous.push(name);
      else files.set(name, matches[0]!);
    }
    if (missing.length === 0 && ambiguous.length === 0) {
      complete.push({ root, entries: files });
    } else {
      incomplete.push({ root, missing, ambiguous });
    }
  }

  if (complete.length > 1) {
    throw new Error('Multiple complete Whisper 4-graph bundles were found; select one folder.');
  }
  if (complete.length === 1) return complete[0]!;

  const issue = incomplete[0];
  if (issue?.ambiguous.length) {
    throw new Error(`Ambiguous Whisper 4-graph artifacts: ${issue.ambiguous.join(', ')}.`);
  }
  throw new Error(
    `Incomplete Whisper 4-graph bundle${issue?.missing.length ? `; missing ${issue.missing.join(', ')}` : ''}. Select one self-contained fp16 bundle folder.`,
  );
}

function inspectWhisperEntries(
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  try {
    resolveLocalBundle(entries);
    return {
      // These are structural candidates only; resolveEntries checks the
      // actual manifest precision before allocating or loading graph assets.
      encoderQuantizations: ['fp16'],
      decoderQuantizations: ['fp16'],
      tokenizerNames: ['tokenizer.json'],
      preprocessorNames: [],
    };
  } catch {
    return {
      encoderQuantizations: [],
      decoderQuantizations: [],
      tokenizerNames: [],
      preprocessorNames: [],
    };
  }
}

interface WhisperLocalExternalData {
  readonly graph: keyof typeof GRAPH_FILES;
  readonly path: string;
  readonly entry: SpeechModelLocalEntry;
}

function safeRelativePath(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0) return null;
  const normalized = normalizeLocalRelativePath(value);
  const relative = normalized.replace(/^\.\//, '');
  if (
    !relative ||
    relative.startsWith('/') ||
    /^[a-z][a-z\d+.-]*:/i.test(relative) ||
    relative.split('/').some((part) => part === '..' || part === '.')
  ) {
    return null;
  }
  return relative;
}

async function validateWhisperManifest(
  entry: SpeechModelLocalEntry,
  configEntry: SpeechModelLocalEntry,
  bundle: LocalWhisperBundle,
  entries: readonly SpeechModelLocalEntry[],
  expected: {
    readonly melBins?: number;
    readonly maxSourcePositions?: number;
    readonly maxTargetPositions?: number;
    readonly vocabularySize?: number;
    readonly decoderLayers?: number;
    readonly hiddenSize?: number;
    readonly decoderAttentionHeads?: number;
  },
): Promise<WhisperLocalExternalData[]> {
  const [manifestFile, configFile] = await Promise.all([
    getLocalEntryFile(entry),
    getLocalEntryFile(configEntry),
  ]);
  let manifest: Record<string, unknown>;
  let config: Record<string, unknown>;
  try {
    manifest = JSON.parse(await manifestFile.text()) as Record<string, unknown>;
  } catch {
    throw new Error('The local Whisper 4-graph manifest is not valid JSON.');
  }
  try {
    config = JSON.parse(await configFile.text()) as Record<string, unknown>;
  } catch {
    throw new Error('The local Whisper config.json is not valid JSON.');
  }
  if (manifest.format !== 'whisper-browser-self-export-v1' || config.model_type !== 'whisper') {
    throw new Error('The local files are not a supported Whisper 4-graph export.');
  }
  const parsed = parseWhisperManifest(manifest);
  if (parsed.alignmentExport?.encoderHiddenStateDtype !== 'float16') {
    throw new Error('The local Whisper manifest must declare float16 encoder hidden states.');
  }

  const geometryFields = [
    'd_model',
    'decoder_layers',
    'decoder_attention_heads',
    'max_source_positions',
    'max_target_positions',
    'num_mel_bins',
    'vocab_size',
  ] as const;
  for (const field of geometryFields) {
    const value = config[field];
    if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
      throw new Error(`The local Whisper config has invalid ${field}.`);
    }
    if (typeof manifest[field] === 'number' && manifest[field] !== value) {
      throw new Error(`The local Whisper manifest and config disagree on ${field}.`);
    }
  }
  const dModel = config.d_model as number;
  const decoderAttentionHeads = config.decoder_attention_heads as number;
  if (dModel % decoderAttentionHeads !== 0) {
    throw new Error('The local Whisper config has incompatible decoder dimensions.');
  }
  if (
    (config.head_dim !== undefined && config.head_dim !== dModel / decoderAttentionHeads) ||
    (expected.melBins !== undefined && config.num_mel_bins !== expected.melBins) ||
    (expected.maxSourcePositions !== undefined &&
      config.max_source_positions !== expected.maxSourcePositions / 2) ||
    (expected.maxTargetPositions !== undefined &&
      config.max_target_positions !== expected.maxTargetPositions) ||
    (expected.vocabularySize !== undefined && config.vocab_size !== expected.vocabularySize) ||
    (expected.decoderLayers !== undefined && config.decoder_layers !== expected.decoderLayers) ||
    (expected.hiddenSize !== undefined && config.d_model !== expected.hiddenSize) ||
    (expected.decoderAttentionHeads !== undefined &&
      config.decoder_attention_heads !== expected.decoderAttentionHeads)
  ) {
    throw new Error('The local Whisper bundle does not match the registered preset configuration.');
  }

  const artifacts = manifest.artifacts as
    | Record<
        string,
        { file?: string; externalData?: readonly { path?: string; file?: string }[] } | undefined
      >
    | undefined;
  const externalData: WhisperLocalExternalData[] = [];
  for (const [graph, graphPath] of Object.entries(GRAPH_FILES) as [
    keyof typeof GRAPH_FILES,
    (typeof GRAPH_FILES)[keyof typeof GRAPH_FILES],
  ][]) {
    const artifact = artifacts?.[graph];
    if (artifact?.file !== graphPath) {
      throw new Error(`The local manifest is missing the expected ${graphPath} graph entry.`);
    }
    if (artifact.externalData !== undefined && !Array.isArray(artifact.externalData)) {
      throw new Error(`The local ${graphPath} manifest has an invalid externalData list.`);
    }
    const declared = artifact.externalData ?? [];
    const seenPaths = new Set<string>();
    for (const sidecar of declared) {
      const path = safeRelativePath(sidecar.path);
      const file = safeRelativePath(sidecar.file);
      if (!path || !file) {
        throw new Error(`The local ${graphPath} manifest has an unsafe external-data path.`);
      }
      if (seenPaths.has(path.toLowerCase())) {
        throw new Error(`Duplicate external-data path declared for ${graphPath}.`);
      }
      seenPaths.add(path.toLowerCase());
      const expectedPath = bundle.root ? `${bundle.root}/${file}` : file;
      const matches = entries.filter(
        (candidate) => localPath(candidate) === expectedPath.toLowerCase(),
      );
      if (matches.length === 0) {
        throw new Error(`Missing declared external data for ${graphPath}: ${sidecar.file}.`);
      }
      if (matches.length > 1) {
        throw new Error(`Ambiguous declared external data for ${graphPath}: ${sidecar.file}.`);
      }
      externalData.push({ graph, path: sidecar.path!, entry: matches[0]! });
    }
  }
  return externalData;
}

export const whisperBuiltInLocalModelAdapter: BuiltInLocalModelAdapter = {
  preset: 'whisper',
  createEntries(files) {
    return createLocalEntriesFromFiles(files);
  },
  async collectEntries(dirHandle: SpeechModelLocalDirectoryHandleLike, prefix = '', signal) {
    return await collectLocalEntriesFromDirectory(dirHandle, prefix, signal);
  },
  inspectEntries(entries) {
    return inspectWhisperEntries(entries);
  },
  async resolveEntries(
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
  ): Promise<ResolvedBuiltInLocalArtifacts> {
    if (options.entries.length === 0) {
      throw new Error('Pick a local Whisper 4-graph bundle folder first.');
    }
    const manifest = resolveWhisperPresetManifest(options.modelId);
    if (!manifest) {
      throw new Error(`No registered Whisper preset matches "${options.modelId}".`);
    }
    if (
      options.backend === 'wasm' ||
      options.encoderBackend === 'wasm' ||
      options.decoderBackend === 'wasm'
    ) {
      throw new Error('This local Whisper 4-graph bundle is registered for WebGPU only.');
    }
    if (
      (options.encoderQuant && options.encoderQuant !== 'fp16') ||
      (options.decoderQuant && options.decoderQuant !== 'fp16')
    ) {
      throw new Error('This local Whisper 4-graph bundle contains fp16 graphs only.');
    }

    const bundle = resolveLocalBundle(options.entries);
    const externalData = await validateWhisperManifest(
      bundle.entries.get('manifest.json')!,
      bundle.entries.get('config.json')!,
      bundle,
      options.entries,
      { ...manifest.config, ...manifest.localGeometry },
    );
    const assetProvider = createBlobAssetProvider();
    const assetHandles: ResolvedBuiltInLocalArtifacts['assetHandles'][number][] = [];
    const toLocator = async (entry: SpeechModelLocalEntry): Promise<string> => {
      const file = await getLocalEntryFile(entry);
      const handle = await assetProvider.resolve({
        id: entry.path,
        provider: 'blob',
        blob: file instanceof Blob ? file : new Blob([file]),
        signal: options.signal,
      });
      assetHandles.push(handle);
      const locator = await handle.getLocator('url');
      if (!locator) {
        throw new Error(`Could not create a URL locator for local asset "${entry.path}".`);
      }
      return locator;
    };

    try {
      const urlFor = async (name: (typeof REQUIRED_FILES)[number]) =>
        await toLocator(bundle.entries.get(name)!);
      const encoderUrl = await urlFor('encoder_model.onnx');
      const decoderInitUrl = await urlFor('decoder_init.onnx');
      const decoderStepUrl = await urlFor('decoder_step.onnx');
      const decoderAlignUrl = await urlFor('decoder_align.onnx');
      const tokenizerUrl = await urlFor('tokenizer.json');
      const manifestUrl = await urlFor('manifest.json');
      const configUrl = await urlFor('config.json');
      const generationConfigUrl = await urlFor('generation_config.json');
      // The split-graph resolver resolves `file` as a URL. Absolute blob URLs
      // keep each local sidecar paired with its graph despite opaque blob URLs.
      const externalDataUrls: NonNullable<WhisperSplitGraphArtifacts['externalDataUrls']> = {};
      for (const sidecar of externalData) {
        const locator = await toLocator(sidecar.entry);
        externalDataUrls[sidecar.graph] = [
          ...(externalDataUrls[sidecar.graph] ?? []),
          { path: sidecar.path, file: locator },
        ];
      }

      return {
        modelId: options.modelId,
        preset: 'whisper',
        builtInLoadOptions: {
          source: {
            kind: 'splitgraph',
            artifacts: {
              encoderUrl,
              decoderInitUrl,
              decoderStepUrl,
              decoderAlignUrl,
              tokenizerUrl,
              manifestUrl,
              configUrl,
              generationConfigUrl,
              externalDataUrls,
            },
            encoderBackend: options.encoderBackend ?? 'webgpu',
            decoderBackend: options.decoderBackend ?? 'webgpu',
            cpuThreads: options.cpuThreads,
            wasmPaths: options.wasmPaths,
            enableProfiling: options.enableProfiling,
          },
        },
        assetHandles,
        selection: {
          encoderName: 'encoder_model.onnx',
          decoderName: 'decoder_step.onnx',
          tokenizerName: 'tokenizer.json',
          encoderQuant: 'fp16',
          decoderQuant: 'fp16',
        },
      };
    } catch (error) {
      await Promise.all(assetHandles.map(async (handle) => await handle.dispose()));
      throw error;
    }
  },
};
