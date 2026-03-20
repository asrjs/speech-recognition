import {
  collectParakeetLocalEntries,
  createParakeetLocalEntries,
  inspectParakeetLocalEntries,
  resolveParakeetLocalEntries,
  type ResolvedParakeetLocalArtifacts,
} from './compat.js';
import type {
  BuiltInLocalModelAdapter,
  LoadSpeechModelFromLocalEntriesOptions,
  ResolvedBuiltInLocalArtifacts,
  SpeechModelLocalInspection,
} from '../../runtime/local-adapter-registry.js';

function normalizeSpeechModelLocalInspection(
  inspection: ReturnType<typeof inspectParakeetLocalEntries>,
): SpeechModelLocalInspection {
  return {
    encoderQuantizations: inspection.encoderQuantizations,
    decoderQuantizations: inspection.decoderQuantizations,
    tokenizerNames: inspection.tokenizerNames,
    preprocessorNames: inspection.preprocessorNames,
  };
}

function toBuiltInDirectLoadOptions(
  resolved: ResolvedParakeetLocalArtifacts,
): ResolvedBuiltInLocalArtifacts['builtInLoadOptions'] {
  return {
    source: {
      kind: 'direct',
      encoderBackend: resolved.config.encoderBackend,
      decoderBackend: resolved.config.decoderBackend,
      artifacts: {
        encoderUrl: resolved.config.encoderUrl,
        decoderUrl: resolved.config.decoderUrl,
        tokenizerUrl: resolved.config.tokenizerUrl,
        preprocessorUrl: resolved.config.preprocessorUrl,
        encoderDataUrl: resolved.config.encoderDataUrl ?? undefined,
        decoderDataUrl: resolved.config.decoderDataUrl ?? undefined,
        encoderFilename: resolved.config.filenames?.encoder,
        decoderFilename: resolved.config.filenames?.decoder,
      },
      preprocessorBackend: resolved.config.preprocessorBackend,
      cpuThreads: resolved.config.cpuThreads,
      enableProfiling: resolved.config.enableProfiling,
    },
  };
}

export const parakeetBuiltInLocalModelAdapter: BuiltInLocalModelAdapter = {
  preset: 'parakeet',
  createEntries(files) {
    return createParakeetLocalEntries(files);
  },
  async collectEntries(dirHandle, prefix = '') {
    return await collectParakeetLocalEntries(dirHandle, prefix);
  },
  inspectEntries(entries) {
    return normalizeSpeechModelLocalInspection(inspectParakeetLocalEntries(entries));
  },
  async resolveEntries(
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
  ): Promise<ResolvedBuiltInLocalArtifacts> {
    const resolved = await resolveParakeetLocalEntries(options.entries, {
      modelId: options.modelId,
      encoderBackend: options.encoderBackend,
      decoderBackend: options.decoderBackend,
      encoderQuant: options.encoderQuant,
      decoderQuant: options.decoderQuant,
      tokenizerName: options.tokenizerName,
      preprocessorName:
        options.preprocessorName === 'nemo80' || options.preprocessorName === 'nemo128'
          ? options.preprocessorName
          : undefined,
      preprocessorBackend: options.preprocessorBackend,
      backend: options.backend,
      verbose: options.verbose,
      cpuThreads: options.cpuThreads,
      enableProfiling: options.enableProfiling,
    });

    return {
      modelId: options.modelId,
      preset: 'parakeet',
      builtInLoadOptions: toBuiltInDirectLoadOptions(resolved),
      assetHandles: resolved.assetHandles,
      selection: resolved.selection,
    };
  },
};
