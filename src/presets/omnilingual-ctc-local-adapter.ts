import {
  collectLocalEntriesFromDirectory,
  createLocalEntriesFromFiles,
  findLocalEntry,
  findLocalSidecar,
  getLocalEntryFile,
} from '../runtime/local-entry-files.js';
import { createBlobAssetProvider } from '../io/index.js';
import { DEFAULT_OMNILINGUAL_CTC_NORMALIZATION_EPSILON } from '../models/omnilingual-ctc/model.js';
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

const GRAPH_CANDIDATES = ['model.int8.onnx', 'model.onnx'];

/** Reports what a local omnilingual wav2vec2-CTC folder can actually provide. */
function inspectOmnilingualEntries(
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  const graph = GRAPH_CANDIDATES.filter((name) => findLocalEntry(entries, name));
  const quantizations = graph.some((name) => name.includes('int8')) ? ['int8'] : ['fp32'];

  return {
    encoderQuantizations: graph.length ? (quantizations as never) : [],
    decoderQuantizations: graph.length ? (quantizations as never) : [],
    tokenizerNames: findLocalEntry(entries, 'tokens.txt') ? ['tokens.txt'] : [],
    preprocessorNames: [],
  };
}

export const omnilingualCtcBuiltInLocalModelAdapter: BuiltInLocalModelAdapter = {
  preset: 'omnilingual-ctc',
  createEntries(files) {
    return createLocalEntriesFromFiles(files);
  },
  async collectEntries(dirHandle: SpeechModelLocalDirectoryHandleLike, prefix = '', signal) {
    return await collectLocalEntriesFromDirectory(dirHandle, prefix, signal);
  },
  inspectEntries(entries) {
    return inspectOmnilingualEntries(entries);
  },
  async resolveEntries(
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
  ): Promise<ResolvedBuiltInLocalArtifacts> {
    if (options.entries.length === 0) {
      throw new Error('Pick a local model folder first.');
    }

    const inspection = inspectOmnilingualEntries(options.entries);
    const graphName = inspection.encoderQuantizations.length
      ? GRAPH_CANDIDATES.find((name) => findLocalEntry(options.entries, name))!
      : undefined;
    const graphEntry = graphName ? findLocalEntry(options.entries, graphName) : null;
    const tokenizerEntry = findLocalEntry(options.entries, 'tokens.txt');
    if (!graphEntry || !graphName) {
      throw new Error(`Missing encoder file: expected one of ${GRAPH_CANDIDATES.join(', ')}`);
    }
    if (!tokenizerEntry) {
      throw new Error('Missing tokenizer file: tokens.txt');
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
        preset: 'omnilingual-ctc',
        builtInLoadOptions: {
          modelUrl: await toLocator(graphEntry),
          tokensUrl: await toLocator(tokenizerEntry),
          externalDataUrl: dataEntry ? await toLocator(dataEntry) : undefined,
          externalDataFilename: graphEntry.basename,
          sampleRate: 16000,
          // The exported Omnilingual graph's official LayerNorm uses eps=1e-5.
          normalizationEpsilon: DEFAULT_OMNILINGUAL_CTC_NORMALIZATION_EPSILON,
          vocabularySize: inspection.tokenizerNames.length ? 10288 : undefined,
          backendId: options.encoderBackend,
          cpuThreads: options.cpuThreads,
          wasmPaths: options.wasmPaths,
          preprocessorBackend: 'js',
        },
        assetHandles,
        selection: {
          encoderName: graphName,
          decoderName: graphName,
          tokenizerName: tokenizerEntry.basename,
          encoderQuant: graphName.includes('int8') ? 'int8' : 'fp32',
          decoderQuant: graphName.includes('int8') ? 'int8' : 'fp32',
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
