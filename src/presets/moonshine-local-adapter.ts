import {
  collectLocalEntriesFromDirectory,
  createLocalEntriesFromFiles,
  findLocalEntry,
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

const REQUIRED = [
  'encoder_model.onnx',
  'decoder_cross_kv_model.onnx',
  'decoder_with_past_model.onnx',
  'tokenizer.json',
  'export-receipt.json',
];

function inspectMoonshineEntries(
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  const present = REQUIRED.filter((name) => findLocalEntry(entries, name));

  return {
    encoderQuantizations: present.includes('encoder_model.onnx') ? ['fp32'] : [],
    decoderQuantizations: present.includes('decoder_with_past_model.onnx') ? ['fp32'] : [],
    tokenizerNames: present.includes('tokenizer.json') ? ['tokenizer.json'] : [],
    preprocessorNames: [],
  };
}

export const moonshineBuiltInLocalModelAdapter: BuiltInLocalModelAdapter = {
  preset: 'moonshine',
  createEntries(files) {
    return createLocalEntriesFromFiles(files);
  },
  async collectEntries(dirHandle: SpeechModelLocalDirectoryHandleLike, prefix = '', signal) {
    return await collectLocalEntriesFromDirectory(dirHandle, prefix, signal);
  },
  inspectEntries(entries) {
    return inspectMoonshineEntries(entries);
  },
  async resolveEntries(
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
  ): Promise<ResolvedBuiltInLocalArtifacts> {
    if (options.entries.length === 0) {
      throw new Error('Pick a local model folder first.');
    }

    const inspection = inspectMoonshineEntries(options.entries);
    if (inspection.encoderQuantizations.length === 0) {
      throw new Error('Missing encoder file: encoder_model.onnx');
    }
    if (inspection.decoderQuantizations.length === 0) {
      throw new Error('Missing decoder file: decoder_with_past_model.onnx');
    }

    const assetProvider = createBlobAssetProvider();
    const assetHandles: ResolvedBuiltInLocalArtifacts['assetHandles'][number][] = [];
    const toLocator = async (name: string): Promise<string> => {
      const entry = findLocalEntry(options.entries, name);
      if (!entry) {
        throw new Error(`Missing local artifact: ${name}`);
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
      return {
        modelId: options.modelId,
        preset: 'moonshine',
        builtInLoadOptions: {
          encoderUrl: await toLocator('encoder_model.onnx'),
          crossKvUrl: await toLocator('decoder_cross_kv_model.onnx'),
          decoderUrl: await toLocator('decoder_with_past_model.onnx'),
          tokenizerUrl: await toLocator('tokenizer.json'),
          generationUrl: await toLocator('generation_config.json'),
          receiptUrl: await toLocator('export-receipt.json'),
          encoderBackend: options.encoderBackend,
          cpuThreads: options.cpuThreads,
        },
        assetHandles,
        selection: {
          encoderName: 'encoder_model.onnx',
          decoderName: 'decoder_with_past_model.onnx',
          tokenizerName: 'tokenizer.json',
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
