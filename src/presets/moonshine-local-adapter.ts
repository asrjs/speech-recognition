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

const REQUIRED = [
  'encoder_model.onnx',
  'decoder_cross_kv_model.onnx',
  'decoder_with_past_model.onnx',
  'tokenizer.json',
  'export-receipt.json',
];

/** Streaming graph set; optional, resolved into loadOptions.streaming when present. */
const STREAMING = [
  'encoder_chunked_model.onnx',
  'decoder_prefill_model.onnx',
  'decoder_step_model.onnx',
];
const STREAMING_MODEL_IDS = new Set([
  'moonshine-ai/moonshine-streaming-tiny',
  'moonshine-streaming-tiny',
]);

function inspectMoonshineEntries(
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  const hasOffline = REQUIRED.every((name) => findLocalEntry(entries, name));
  const hasStreaming =
    Boolean(findLocalEntry(entries, 'encoder_model.onnx')) &&
    Boolean(findLocalEntry(entries, 'tokenizer.json')) &&
    STREAMING.every((name) => findLocalEntry(entries, name));

  return {
    encoderQuantizations: hasOffline || hasStreaming ? ['fp32'] : [],
    decoderQuantizations: hasOffline || hasStreaming ? ['fp32'] : [],
    tokenizerNames: findLocalEntry(entries, 'tokenizer.json') ? ['tokenizer.json'] : [],
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

    if (STREAMING_MODEL_IDS.has(options.modelId.toLowerCase())) {
      const missingStreaming = STREAMING.filter((name) => !findLocalEntry(options.entries, name));
      if (missingStreaming.length > 0) {
        throw new Error(`Moonshine streaming model requires local streaming graphs: ${missingStreaming.join(', ')}.`);
      }
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

    const toOptionalSidecarLocator = async (onnxName: string): Promise<string | undefined> => {
      const sidecar = findLocalSidecar(options.entries, onnxName);
      if (!sidecar) {
        return undefined;
      }
      return await toLocator(sidecar.basename);
    };
    const encoderBackend = options.encoderBackend ?? (
      options.backend === 'wasm'
        ? 'wasm'
        : options.backend === 'webgpu' ||
            options.backend === 'webgpu-hybrid' ||
            options.backend === 'webgpu-strict'
          ? 'webgpu'
          : undefined
    );

    try {
      const tokenizerUrl = await toLocator('tokenizer.json');
      const encoderUrl = await toLocator('encoder_model.onnx');
      const encoderDataUrl = await toOptionalSidecarLocator('encoder_model.onnx');
      const hasOffline =
        Boolean(findLocalEntry(options.entries, 'decoder_cross_kv_model.onnx')) &&
        Boolean(findLocalEntry(options.entries, 'decoder_with_past_model.onnx'));
      const crossKvDataUrl = hasOffline
        ? await toOptionalSidecarLocator('decoder_cross_kv_model.onnx')
        : undefined;
      const decoderDataUrl = hasOffline
        ? await toOptionalSidecarLocator('decoder_with_past_model.onnx')
        : undefined;
      const hasStreaming = STREAMING.every((name) => findLocalEntry(options.entries, name));
      const chunkedEncoderDataUrl = hasStreaming
        ? await toOptionalSidecarLocator('encoder_chunked_model.onnx')
        : undefined;
      const prefillDataUrl = hasStreaming
        ? await toOptionalSidecarLocator('decoder_prefill_model.onnx')
        : undefined;
      const stepDataUrl = hasStreaming
        ? await toOptionalSidecarLocator('decoder_step_model.onnx')
        : undefined;
      const streaming = hasStreaming
        ? {
            encoderUrl,
            ...(encoderDataUrl ? { encoderDataUrl } : {}),
            chunkedEncoderUrl: await toLocator('encoder_chunked_model.onnx'),
            ...(chunkedEncoderDataUrl ? { chunkedEncoderDataUrl } : {}),
            prefillUrl: await toLocator('decoder_prefill_model.onnx'),
            ...(prefillDataUrl ? { prefillDataUrl } : {}),
            stepUrl: await toLocator('decoder_step_model.onnx'),
            ...(stepDataUrl ? { stepDataUrl } : {}),
            tokenizerUrl,
          }
        : undefined;
      return {
        modelId: options.modelId,
        preset: 'moonshine',
        builtInLoadOptions: {
          encoderUrl,
          ...(encoderDataUrl ? { encoderDataUrl } : {}),
          ...(hasOffline
            ? {
                crossKvUrl: await toLocator('decoder_cross_kv_model.onnx'),
                ...(crossKvDataUrl ? { crossKvDataUrl } : {}),
                decoderUrl: await toLocator('decoder_with_past_model.onnx'),
                ...(decoderDataUrl ? { decoderDataUrl } : {}),
              }
            : {}),
          tokenizerUrl,
          generationUrl: findLocalEntry(options.entries, 'generation_config.json')
            ? await toLocator('generation_config.json')
            : undefined,
          receiptUrl: findLocalEntry(options.entries, 'export-receipt.json')
            ? await toLocator('export-receipt.json')
            : undefined,
          encoderBackend,
          cpuThreads: options.cpuThreads,
          ...(streaming ? { streaming } : {}),
        },
        assetHandles,
        selection: {
          encoderName: 'encoder_model.onnx',
          decoderName: hasOffline ? 'decoder_with_past_model.onnx' : 'decoder_step_model.onnx',
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
