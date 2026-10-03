import { createBlobAssetProvider } from '../../io/index.js';
import type { QuantizationMode } from '../../runtime/huggingface.js';
import {
  collectLocalEntriesFromDirectory,
  createLocalEntriesFromFiles,
  findLocalEntry,
  findLocalSidecar,
  getLocalEntryFile,
} from '../../runtime/local-entry-files.js';
import type {
  BuiltInLocalModelAdapter,
  LoadSpeechModelFromLocalEntriesOptions,
  ResolvedBuiltInLocalArtifacts,
  SpeechModelLocalInspection,
} from '../../runtime/local-adapter-registry.js';
import type {
  SpeechModelLocalDirectoryHandleLike,
  SpeechModelLocalEntry,
} from '../../runtime/local-types.js';
import type { ResolvedAssetHandle } from '../../types/index.js';

// Two export generations exist on disk: the library graph names
// (encoder.onnx / joint.onnx) and the older 320 ms window names
// (encoder_320ms_fp16.onnx with joiner.onnx). Support both instead of
// guessing from a model name.
const NEMOTRON_GRAPH_BASES = {
  encoder: ['encoder_320ms', 'encoder'],
  decoder: ['decoder'],
  joint: ['joint', 'joiner'],
} as const;

const NEMOTRON_TOKENIZER_NAMES = ['tokens.txt', 'vocab.txt'] as const;

const NEMOTRON_QUANT_SUFFIXES: Array<[QuantizationMode, string]> = [
  ['fp32', ''],
  ['fp16', '_fp16'],
  ['int8', '_int8'],
];

const NEMOTRON_DOTTED_QUANT_SUFFIXES: Array<[QuantizationMode, string]> = [
  ['fp32', ''],
  ['fp16', '.fp16'],
  ['int8', '.int8'],
];

function detectNemotronQuants(
  entries: readonly SpeechModelLocalEntry[],
  baseNames: readonly string[],
): QuantizationMode[] {
  const names = new Set(entries.map((entry) => entry.basename.toLowerCase()));
  const found: QuantizationMode[] = [];
  for (const baseName of baseNames) {
    for (const [quant, suffix] of [...NEMOTRON_QUANT_SUFFIXES, ...NEMOTRON_DOTTED_QUANT_SUFFIXES]) {
      if (names.has(`${baseName}${suffix}.onnx`) && !found.includes(quant)) {
        found.push(quant);
      }
    }
  }
  return found;
}

function findNemotronGraphEntry(
  entries: readonly SpeechModelLocalEntry[],
  baseNames: readonly string[],
  quant: QuantizationMode,
): SpeechModelLocalEntry | null {
  for (const baseName of baseNames) {
    for (const [mode, suffix] of [...NEMOTRON_QUANT_SUFFIXES, ...NEMOTRON_DOTTED_QUANT_SUFFIXES]) {
      if (mode === quant) {
        const match = findLocalEntry(entries, `${baseName}${suffix}.onnx`);
        if (match) {
          return match;
        }
      }
    }
  }
  return null;
}

function quantizedNames(baseNames: readonly string[], quant: QuantizationMode): string {
  const suffix = [...NEMOTRON_QUANT_SUFFIXES, ...NEMOTRON_DOTTED_QUANT_SUFFIXES].find(
    ([candidate]) => candidate === quant,
  )?.[1];
  return baseNames.map((baseName) => `${baseName}${suffix ?? ''}.onnx`).join(' or ');
}

function inspectNemotronEntries(
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  const tokenizers = NEMOTRON_TOKENIZER_NAMES.filter((name) =>
    Boolean(findLocalEntry(entries, name)),
  );
  const preprocessors = entries
    .filter((entry) => /^nemo\d+\.onnx$/.test(entry.basename.toLowerCase()))
    .map((entry) => entry.basename);
  return {
    encoderQuantizations: detectNemotronQuants(entries, NEMOTRON_GRAPH_BASES.encoder),
    decoderQuantizations: detectNemotronQuants(entries, NEMOTRON_GRAPH_BASES.decoder),
    tokenizerNames: tokenizers,
    preprocessorNames: preprocessors,
  };
}

export const nemotronBuiltInLocalModelAdapter: BuiltInLocalModelAdapter = {
  preset: 'nemotron',
  createEntries(files) {
    return createLocalEntriesFromFiles(files);
  },
  async collectEntries(dirHandle: SpeechModelLocalDirectoryHandleLike, prefix = '', signal) {
    return await collectLocalEntriesFromDirectory(dirHandle, prefix, signal);
  },
  inspectEntries(entries) {
    return inspectNemotronEntries(entries);
  },
  async resolveEntries(
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
  ): Promise<ResolvedBuiltInLocalArtifacts> {
    if (options.entries.length === 0) {
      throw new Error('Pick a local model folder first.');
    }

    const inspection = inspectNemotronEntries(options.entries);
    const encoderQuant: QuantizationMode =
      options.encoderQuant ?? inspection.encoderQuantizations[0] ?? 'fp32';
    const decoderQuant: QuantizationMode =
      options.decoderQuant ?? inspection.decoderQuantizations[0] ?? 'fp32';
    const encoderEntry = findNemotronGraphEntry(
      options.entries,
      NEMOTRON_GRAPH_BASES.encoder,
      encoderQuant,
    );
    const decoderEntry = findNemotronGraphEntry(
      options.entries,
      NEMOTRON_GRAPH_BASES.decoder,
      decoderQuant,
    );
    const jointEntry = findNemotronGraphEntry(
      options.entries,
      NEMOTRON_GRAPH_BASES.joint,
      decoderQuant,
    );
    const tokenizerName = options.tokenizerName ?? inspection.tokenizerNames[0];
    const tokenizerEntry = tokenizerName ? findLocalEntry(options.entries, tokenizerName) : null;

    if (!encoderEntry) {
      throw new Error(
        `Missing encoder file: ${quantizedNames(NEMOTRON_GRAPH_BASES.encoder, encoderQuant)}`,
      );
    }
    if (!decoderEntry) {
      throw new Error(
        `Missing decoder file: ${quantizedNames(NEMOTRON_GRAPH_BASES.decoder, decoderQuant)}`,
      );
    }
    if (!jointEntry) {
      throw new Error(
        `Missing joint file: ${quantizedNames(NEMOTRON_GRAPH_BASES.joint, decoderQuant)}`,
      );
    }
    if (!tokenizerEntry) {
      throw new Error(`Missing tokenizer file: ${tokenizerName ?? NEMOTRON_TOKENIZER_NAMES[0]}`);
    }

    const assetProvider = createBlobAssetProvider();
    const assetHandles: ResolvedAssetHandle[] = [];
    const toLocator = async (entry: SpeechModelLocalEntry): Promise<string> => {
      if (options.signal?.aborted) {
        throw new Error('Local model load aborted.');
      }

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
      const encoderDataEntry = findLocalSidecar(options.entries, encoderEntry.basename);
      const decoderDataEntry = findLocalSidecar(options.entries, decoderEntry.basename);
      const jointDataEntry = findLocalSidecar(options.entries, jointEntry.basename);

      return {
        modelId: options.modelId,
        preset: 'nemotron',
        builtInLoadOptions: {
          source: {
            kind: 'direct',
            artifacts: {
              encoderUrl: await toLocator(encoderEntry),
              decoderUrl: await toLocator(decoderEntry),
              jointUrl: await toLocator(jointEntry),
              tokenizerUrl: await toLocator(tokenizerEntry),
              encoderDataUrl: encoderDataEntry ? await toLocator(encoderDataEntry) : undefined,
              decoderDataUrl: decoderDataEntry ? await toLocator(decoderDataEntry) : undefined,
              jointDataUrl: jointDataEntry ? await toLocator(jointDataEntry) : undefined,
              encoderFilename: encoderEntry.basename,
              decoderFilename: decoderEntry.basename,
              jointFilename: jointEntry.basename,
            },
            encoderBackend: options.encoderBackend,
            decoderBackend: options.decoderBackend,
          },
          preprocessorBackend: options.preprocessorBackend ?? 'js',
          cpuThreads: options.cpuThreads,
          wasmPaths: options.wasmPaths,
          enableProfiling: options.enableProfiling,
        },
        assetHandles,
        selection: {
          encoderName: encoderEntry.basename,
          decoderName: decoderEntry.basename,
          tokenizerName: tokenizerEntry.basename,
          encoderQuant,
          decoderQuant,
        },
      };
    } catch (error) {
      await Promise.all(
        assetHandles.map(async (handle) => {
          await handle.dispose();
        }),
      );
      throw error;
    }
  },
};
