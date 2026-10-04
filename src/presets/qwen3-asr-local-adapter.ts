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
  'audio_front_model.onnx',
  'decoder_step_model.onnx',
  'embed_tokens_model.onnx',
  'tokenizer.json',
];

function inspectQwen3AsrEntries(
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  return {
    encoderQuantizations: findLocalEntry(entries, 'audio_front_model.onnx') ? ['fp32'] : [],
    decoderQuantizations: findLocalEntry(entries, 'decoder_step_model.onnx') ? ['fp32'] : [],
    tokenizerNames: findLocalEntry(entries, 'tokenizer.json') ? ['tokenizer.json'] : [],
    preprocessorNames: [],
  };
}

export const qwen3AsrBuiltInLocalModelAdapter: BuiltInLocalModelAdapter = {
  preset: 'qwen3-asr',
  createEntries(files) {
    return createLocalEntriesFromFiles(files);
  },
  async collectEntries(dirHandle: SpeechModelLocalDirectoryHandleLike, prefix = '', signal) {
    return await collectLocalEntriesFromDirectory(dirHandle, prefix, signal);
  },
  inspectEntries(entries) {
    return inspectQwen3AsrEntries(entries);
  },
  async resolveEntries(
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
  ): Promise<ResolvedBuiltInLocalArtifacts> {
    if (options.entries.length === 0) {
      throw new Error('Pick a local model folder first.');
    }
    for (const name of REQUIRED) {
      if (!findLocalEntry(options.entries, name)) {
        throw new Error(`Missing ${name} for the qwen3-asr preset.`);
      }
    }

    const assetProvider = createBlobAssetProvider();
    const assetHandles: ResolvedBuiltInLocalArtifacts['assetHandles'][number][] = [];
    const toLocator = async (name: string): Promise<string> => {
      const entry = findLocalEntry(options.entries, name)!;
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
      const frontUrl = await toLocator('audio_front_model.onnx');
      const decoderUrl = await toLocator('decoder_step_model.onnx');
      const embedUrl = await toLocator('embed_tokens_model.onnx');
      const tokenizerUrl = await toLocator('tokenizer.json');
      const cuSeqlensUrl = findLocalEntry(options.entries, 'jfk_cu_seqlens.bin')
        ? await toLocator('jfk_cu_seqlens.bin')
        : undefined;
      const prefixEmbedsUrl = findLocalEntry(options.entries, 'prompt_prefix_embeds.npy')
        ? await toLocator('prompt_prefix_embeds.npy')
        : undefined;
      const suffixEmbedsUrl = findLocalEntry(options.entries, 'prompt_suffix_embeds.npy')
        ? await toLocator('prompt_suffix_embeds.npy')
        : undefined;

      return {
        modelId: options.modelId,
        preset: 'qwen3-asr',
        builtInLoadOptions: {
          frontUrl,
          decoderUrl,
          embedUrl,
          tokenizerUrl,
          cuSeqlensUrl,
          prefixEmbedsUrl,
          suffixEmbedsUrl,
          encoderBackend: options.encoderBackend,
          cpuThreads: options.cpuThreads,
        },
        assetHandles,
        selection: {
          encoderName: 'audio_front_model.onnx',
          decoderName: 'decoder_step_model.onnx',
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
