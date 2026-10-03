import {
  collectLocalEntriesFromDirectory,
  createLocalEntriesFromFiles,
  findLocalEntry,
  findLocalSidecar,
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

const ENCODER_BASE = 'model';
const TOKENIZER_CANDIDATES = ['tokenizer.json'];

/** Reports what a local ParakeetForCTC folder can actually provide. */
function inspectParakeetCtcEntries(
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  const encoders = [ENCODER_BASE, 'model_fp16', 'model_int8'];
  const tokenizer = TOKENIZER_CANDIDATES.filter((name) => findLocalEntry(entries, name));

  return {
    encoderQuantizations: encoders.some((name) => findLocalEntry(entries, `${name}.onnx`))
      ? ['fp32']
      : [],
    decoderQuantizations: ['fp32'],
    tokenizerNames: tokenizer,
    preprocessorNames: [],
  };
}

export const parakeetCtcBuiltInLocalModelAdapter: BuiltInLocalModelAdapter = {
  preset: 'parakeet-ctc',
  createEntries(files) {
    return createLocalEntriesFromFiles(files);
  },
  async collectEntries(dirHandle: SpeechModelLocalDirectoryHandleLike, prefix = '', signal) {
    return await collectLocalEntriesFromDirectory(dirHandle, prefix, signal);
  },
  inspectEntries(entries) {
    return inspectParakeetCtcEntries(entries);
  },
  async resolveEntries(
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
  ): Promise<ResolvedBuiltInLocalArtifacts> {
    if (options.entries.length === 0) {
      throw new Error('Pick a local model folder first.');
    }

    const inspection = inspectParakeetCtcEntries(options.entries);
    const encoderName = `${ENCODER_BASE}.onnx`;
    const tokenizerName = options.tokenizerName ?? inspection.tokenizerNames[0];
    const encoderEntry = findLocalEntry(options.entries, encoderName);
    const tokenizerEntry = tokenizerName ? findLocalEntry(options.entries, tokenizerName) : null;
    if (!encoderEntry) {
      throw new Error(`Missing encoder file: ${encoderName}`);
    }
    if (!tokenizerEntry) {
      throw new Error(`Missing tokenizer file: ${tokenizerName ?? TOKENIZER_CANDIDATES[0]}`);
    }

    const assetProvider = createBlobAssetProvider();
    const assetHandles: ResolvedBuiltInLocalArtifacts['assetHandles'][number][] = [];
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

      return {
        modelId: options.modelId,
        preset: 'parakeet-ctc',
        builtInLoadOptions: {
          source: {
            kind: 'direct',
            artifacts: {
              encoderUrl: await toLocator(encoderEntry),
              tokenizerUrl: await toLocator(tokenizerEntry),
              encoderDataUrl: encoderDataEntry ? await toLocator(encoderDataEntry) : undefined,
              encoderFilename: encoderEntry.basename,
            },
            encoderBackend: options.encoderBackend,
          },
          preprocessorBackend: options.preprocessorBackend ?? 'js',
          cpuThreads: options.cpuThreads,
          wasmPaths: options.wasmPaths,
        },
        assetHandles,
        selection: {
          encoderName,
          decoderName: encoderName,
          tokenizerName: tokenizerEntry.basename,
          encoderQuant: 'fp32',
          decoderQuant: 'fp32',
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
