import {
  collectLocalEntriesFromDirectory,
  createLocalEntriesFromFiles,
  findLocalEntry,
  findLocalSidecar,
  getLocalEntryFile,
} from '../runtime/local-entry-files.js';
import { createBlobAssetProvider } from '../io/index.js';
import type {
  BuiltInLocalModelAdapter,
  LoadSpeechModelFromLocalEntriesOptions,
  ResolvedBuiltInLocalArtifacts,
  SpeechModelLocalInspection,
} from '../runtime/local-adapter-registry.js';
import type {
  SpeechModelLocalDirectoryHandleLike,
  SpeechModelLocalEntry,
} from '../runtime/local-types.js';

const GRAPH_NAME = 'model.onnx';
const VOCAB_NAMES = ['vocab.json'];

function inspectWav2Vec2Entries(
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  const hasGraph = Boolean(findLocalEntry(entries, GRAPH_NAME));
  const vocab = VOCAB_NAMES.filter((name) => findLocalEntry(entries, name));

  return {
    encoderQuantizations: hasGraph ? ['fp32'] : [],
    decoderQuantizations: hasGraph ? ['fp32'] : [],
    tokenizerNames: vocab,
    preprocessorNames: [],
  };
}

export const wav2vec2TurkishBuiltInLocalModelAdapter: BuiltInLocalModelAdapter = {
  preset: 'wav2vec2-tr',
  createEntries(files) {
    return createLocalEntriesFromFiles(files);
  },
  async collectEntries(dirHandle: SpeechModelLocalDirectoryHandleLike, prefix = '', signal) {
    return await collectLocalEntriesFromDirectory(dirHandle, prefix, signal);
  },
  inspectEntries(entries) {
    return inspectWav2Vec2Entries(entries);
  },
  async resolveEntries(
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
  ): Promise<ResolvedBuiltInLocalArtifacts> {
    if (options.entries.length === 0) {
      throw new Error('Pick a local model folder first.');
    }

    const inspection = inspectWav2Vec2Entries(options.entries);
    const graphEntry = findLocalEntry(options.entries, GRAPH_NAME);
    const vocabName = options.tokenizerName ?? inspection.tokenizerNames[0];
    const vocabEntry = vocabName ? findLocalEntry(options.entries, vocabName) : null;
    if (!graphEntry) {
      throw new Error(`Missing encoder file: ${GRAPH_NAME}`);
    }
    if (!vocabEntry) {
      throw new Error(`Missing tokenizer file: ${vocabName ?? VOCAB_NAMES[0]}`);
    }

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
      const dataEntry = findLocalSidecar(options.entries, graphEntry.basename);

      return {
        modelId: options.modelId,
        preset: 'wav2vec2-tr',
        builtInLoadOptions: {
          source: {
            kind: 'direct',
            artifacts: {
              // Field names follow Wav2Vec2DirectArtifacts exactly.
              modelUrl: await toLocator(graphEntry),
              tokenizerUrl: await toLocator(vocabEntry),
              modelDataUrl: dataEntry ? await toLocator(dataEntry) : undefined,
              modelDataFilename: graphEntry.basename,
            },
            encoderBackend: options.encoderBackend,
            preprocessorBackend: 'js',
          },
          cpuThreads: options.cpuThreads,
          wasmPaths: options.wasmPaths,
        },
        assetHandles,
        selection: {
          encoderName: GRAPH_NAME,
          decoderName: GRAPH_NAME,
          tokenizerName: vocabEntry.basename,
          encoderQuant: 'fp32',
          decoderQuant: 'fp32',
        },
      };
    } catch (error) {
      await Promise.all(
        assetHandles.map((handle) => handle.dispose()),
      );
      throw error;
    }
  },
};
