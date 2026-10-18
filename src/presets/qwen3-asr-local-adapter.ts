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

const ENCODER = 'audio-encoder-dynamic.onnx';
const TOKENIZER = 'tokenizer.json';

type DecoderArtifacts = {
  readonly quant: 'fp16' | 'fp32';
  readonly prefill: string;
  readonly step: string;
  readonly prefillData: string;
  readonly stepData: string;
};

function decoderArtifactCandidates(): readonly DecoderArtifacts[] {
  return [
    {
      quant: 'fp16',
      prefill: 'decoder-prefill-fp16.onnx',
      step: 'decoder-step-fp16.onnx',
      prefillData: 'decoder-fp16.onnx.data',
      stepData: 'decoder-fp16.onnx.data',
    },
    {
      quant: 'fp32',
      prefill: 'decoder-prefill.onnx',
      step: 'decoder-step.onnx',
      prefillData: 'decoder-prefill.onnx.data',
      stepData: 'decoder-step.onnx.data',
    },
  ];
}

function findDecoderArtifacts(
  entries: readonly SpeechModelLocalEntry[],
  requestedQuant?: DecoderArtifacts['quant'],
): DecoderArtifacts | null {
  for (const candidate of decoderArtifactCandidates()) {
    if (requestedQuant && requestedQuant !== candidate.quant) continue;
    if (
      findLocalEntry(entries, candidate.prefill) &&
      findLocalEntry(entries, candidate.step) &&
      findLocalEntry(entries, candidate.prefillData) &&
      findLocalEntry(entries, candidate.stepData)
    ) {
      return candidate;
    }
  }
  return null;
}

function inspectQwen3AsrEntries(
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  const decoders = decoderArtifactCandidates().filter((candidate) =>
    findDecoderArtifacts(entries, candidate.quant),
  );
  return {
    encoderQuantizations: findLocalEntry(entries, ENCODER) ? ['fp32'] : [],
    decoderQuantizations: decoders.map((decoder) => decoder.quant),
    tokenizerNames: findLocalEntry(entries, TOKENIZER) ? [TOKENIZER] : [],
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
    if (
      findLocalEntry(options.entries, 'audio_front_model.onnx') &&
      findLocalEntry(options.entries, 'decoder_step_model.onnx')
    ) {
      throw new Error(
        'This Qwen folder contains the legacy audio_front/decoder_step/embed_tokens graph set, which is incompatible with the arbitrary-audio qwen-asr executor. Use the official audio-encoder-dynamic + decoder-prefill + decoder-step graph set.',
      );
    }
    if (!findLocalEntry(options.entries, ENCODER)) {
      throw new Error(`Missing ${ENCODER} for the official qwen3-asr preset.`);
    }
    const requestedQuant = options.decoderQuant;
    const decoder = requestedQuant
      ? requestedQuant === 'fp16' || requestedQuant === 'fp32'
        ? findDecoderArtifacts(options.entries, requestedQuant)
        : null
      : findDecoderArtifacts(options.entries, 'fp32') ?? findDecoderArtifacts(options.entries, 'fp16');
    if (!decoder) {
      throw new Error(
        `${requestedQuant ? `Missing the requested Qwen ${requestedQuant} decoder pair and external-data sidecar.` : 'No supported Qwen decoder pair and external-data sidecar was found.'} Use decoder-prefill-fp16.onnx + decoder-step-fp16.onnx + decoder-fp16.onnx.data for fp16, or the matching fp32 pair and sidecars.`,
      );
    }
    if (!findLocalEntry(options.entries, TOKENIZER)) {
      throw new Error(`Missing ${TOKENIZER} for the qwen3-asr preset.`);
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
      const encoderUrl = await toLocator(ENCODER);
      const decoderPrefillUrl = await toLocator(decoder.prefill);
      const decoderStepUrl = await toLocator(decoder.step);
      const decoderPrefillDataUrl = await toLocator(decoder.prefillData);
      const decoderStepDataUrl = decoder.stepData === decoder.prefillData
        ? decoderPrefillDataUrl
        : await toLocator(decoder.stepData);
      const tokenizerUrl = await toLocator(TOKENIZER);

      return {
        modelId: options.modelId,
        preset: 'qwen3-asr',
        builtInLoadOptions: {
          source: {
            kind: 'direct',
            artifacts: {
              encoderUrl,
              decoderUrl: decoderPrefillUrl,
              decoderStepUrl,
              tokenizerUrl,
              decoderPrefillDataUrl,
              decoderPrefillDataPath: decoder.prefillData,
              decoderStepDataUrl,
              decoderStepDataPath: decoder.stepData,
            },
            encoderBackend: options.encoderBackend,
            decoderBackend: options.decoderBackend,
            cpuThreads: options.cpuThreads,
            wasmPaths: options.wasmPaths,
            enableProfiling: options.enableProfiling,
          },
        },
        assetHandles,
        selection: {
          encoderName: ENCODER,
          decoderName: decoder.step,
          tokenizerName: TOKENIZER,
          encoderQuant: 'fp32',
          decoderQuant: decoder.quant,
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
