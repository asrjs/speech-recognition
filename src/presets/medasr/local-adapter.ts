import {
  collectLocalEntriesFromDirectory,
  createLocalEntriesFromFiles,
  getLocalEntryFile,
} from '../../runtime/local-entry-files.js';
import { createBlobAssetProvider } from '../../io/index.js';
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
import { resolveMedAsrPresetManifest } from './manifest.js';

const GRAPH = 'model.onnx';
const TOKENIZER = 'tokens.txt';
const EXTERNAL_DATA = 'model.onnx.data';
const CONFIG = 'config.json';
const PREPROCESSOR = 'preprocessor_config.json';
const VOCABULARY_SIZE = 512;

interface MedAsrConfigFile {
  readonly model_type?: string;
  readonly vocab_size?: number;
}

interface MedAsrPreprocessorFile {
  readonly feature_size?: number;
  readonly sampling_rate?: number;
  readonly n_fft?: number;
  readonly win_length?: number;
  readonly hop_length?: number;
}

function uniqueEntry(
  entries: readonly SpeechModelLocalEntry[],
  basename: string,
): SpeechModelLocalEntry | null {
  const matches = entries.filter(
    (entry) => entry.basename.toLowerCase() === basename.toLowerCase(),
  );
  if (matches.length > 1) {
    throw new Error(`Multiple local MedASR artifacts named ${basename}.`);
  }
  return matches[0] ?? null;
}

function parentDirectory(path: string): string {
  const normalized = path.replace(/\\/g, '/');
  const separator = normalized.lastIndexOf('/');
  return separator < 0 ? '' : normalized.slice(0, separator);
}

function inspectMedAsrEntries(
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  const graph = uniqueEntry(entries, GRAPH);
  const tokenizer = uniqueEntry(entries, TOKENIZER);
  return {
    encoderQuantizations: graph ? ['fp32'] : [],
    decoderQuantizations: graph ? ['fp32'] : [],
    tokenizerNames: tokenizer ? [TOKENIZER] : [],
    preprocessorNames: [],
  };
}

async function readJson<T>(entry: SpeechModelLocalEntry, label: string): Promise<T> {
  const file = await getLocalEntryFile(entry);
  try {
    return JSON.parse(await file.text()) as T;
  } catch (error) {
    throw new Error(`Invalid MedASR ${label} JSON in ${entry.path}: ${String(error)}`);
  }
}

function validateTokens(contents: string): void {
  const lines = contents.split(/\r?\n/).filter((line) => line.length > 0);
  for (let id = 0; id < lines.length; id += 1) {
    const match = /^(.*)\s+(\d+)$/.exec(lines[id] ?? '');
    if (!match || Number(match[2]) !== id || match[1] === '') {
      throw new Error(`Invalid MedASR token mapping at line ${id + 1}; expected contiguous IDs.`);
    }
  }
  if (lines.length < VOCABULARY_SIZE || lines[0] !== '<epsilon> 0') {
    throw new Error(
      'MedASR tokens.txt must include 512 contiguous model tokens with <epsilon> at ID 0.',
    );
  }
}

async function validateMetadata(
  configEntry: SpeechModelLocalEntry,
  preprocessorEntry: SpeechModelLocalEntry,
  tokenizerEntry: SpeechModelLocalEntry,
): Promise<void> {
  const [config, preprocessor, tokenizerFile] = await Promise.all([
    readJson<MedAsrConfigFile>(configEntry, 'config'),
    readJson<MedAsrPreprocessorFile>(preprocessorEntry, 'preprocessor'),
    getLocalEntryFile(tokenizerEntry),
  ]);
  if (config.model_type !== 'lasr_ctc' || config.vocab_size !== VOCABULARY_SIZE) {
    throw new Error('Unsupported MedASR config.json: expected lasr_ctc with vocab_size 512.');
  }
  if (
    preprocessor.feature_size !== 128 ||
    preprocessor.sampling_rate !== 16000 ||
    preprocessor.n_fft !== 512 ||
    preprocessor.win_length !== 400 ||
    preprocessor.hop_length !== 160
  ) {
    throw new Error(
      'Unsupported MedASR preprocessor_config.json: expected 16 kHz, 128 bins, FFT 512, window 400, and hop 160.',
    );
  }
  validateTokens(await tokenizerFile.text());
}

export const medAsrBuiltInLocalModelAdapter: BuiltInLocalModelAdapter = {
  preset: 'medasr',
  createEntries(files) {
    return createLocalEntriesFromFiles(files);
  },
  async collectEntries(dirHandle: SpeechModelLocalDirectoryHandleLike, prefix = '', signal) {
    return await collectLocalEntriesFromDirectory(dirHandle, prefix, signal);
  },
  inspectEntries(entries) {
    return inspectMedAsrEntries(entries);
  },
  async resolveEntries(
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
  ): Promise<ResolvedBuiltInLocalArtifacts> {
    const manifest = resolveMedAsrPresetManifest(options.modelId);
    if (!manifest) throw new Error(`Unsupported MedASR model "${options.modelId}".`);
    if (
      (options.encoderQuant && options.encoderQuant !== 'fp32') ||
      (options.decoderQuant && options.decoderQuant !== 'fp32')
    ) {
      throw new Error('This MedASR local bundle provides only FP32 model.onnx.');
    }
    if (options.entries.length === 0) {
      throw new Error('Pick a local model folder first.');
    }

    const graph = uniqueEntry(options.entries, GRAPH);
    const tokenizer = uniqueEntry(options.entries, TOKENIZER);
    const config = uniqueEntry(options.entries, CONFIG);
    const preprocessor = uniqueEntry(options.entries, PREPROCESSOR);
    if (!graph) throw new Error(`Missing MedASR graph: ${GRAPH}`);
    if (!tokenizer) throw new Error(`Missing MedASR tokenizer: ${TOKENIZER}`);
    if (!config) throw new Error(`Missing MedASR config: ${CONFIG}`);
    if (!preprocessor) throw new Error(`Missing MedASR preprocessor config: ${PREPROCESSOR}`);
    if (parentDirectory(graph.path) !== parentDirectory(tokenizer.path)) {
      throw new Error('MedASR model.onnx and tokens.txt must be in the same folder.');
    }
    if (
      parentDirectory(graph.path) !== parentDirectory(config.path) ||
      parentDirectory(graph.path) !== parentDirectory(preprocessor.path)
    ) {
      throw new Error('MedASR config and preprocessor metadata must be beside model.onnx.');
    }

    const sidecar = uniqueEntry(options.entries, EXTERNAL_DATA);
    if (!sidecar) throw new Error(`Missing MedASR external graph data: ${EXTERNAL_DATA}`);
    if (parentDirectory(graph.path) !== parentDirectory(sidecar.path)) {
      throw new Error('MedASR model.onnx.data must be in the same folder as model.onnx.');
    }
    await validateMetadata(config, preprocessor, tokenizer);

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
      return {
        modelId: options.modelId,
        preset: 'medasr',
        builtInLoadOptions: {
          source: {
            kind: 'direct',
            artifacts: {
              modelUrl: await toLocator(graph),
              tokenizerUrl: await toLocator(tokenizer),
              modelDataUrl: await toLocator(sidecar),
              modelDataFilename: EXTERNAL_DATA,
            },
            wasmPaths: options.wasmPaths,
            cpuThreads: options.cpuThreads,
            enableProfiling: options.enableProfiling,
          },
        },
        assetHandles,
        selection: {
          encoderName: GRAPH,
          decoderName: GRAPH,
          tokenizerName: TOKENIZER,
          encoderQuant: 'fp32',
          decoderQuant: 'fp32',
        },
      };
    } catch (error) {
      await Promise.all(assetHandles.map((handle) => handle.dispose()));
      throw error;
    }
  },
};
