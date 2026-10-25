import { createBlobAssetProvider } from '../io/index.js';
import {
  collectLocalEntriesFromDirectory,
  createLocalEntriesFromFiles,
  getLocalEntryFile,
} from '../runtime/local-entry-files.js';
import type {
  BuiltInLocalModelAdapter,
  LoadSpeechModelFromLocalEntriesOptions,
  ResolvedBuiltInLocalArtifacts,
  SpeechModelLocalInspection,
} from '../runtime/local-adapter-registry.js';
import type { SpeechModelLocalDirectoryHandleLike, SpeechModelLocalEntry } from '../runtime/local-types.js';

interface LocalArtifactAdapterConfig {
  readonly preset: string;
  readonly required: readonly string[];
  readonly encoderArtifact: string;
  readonly decoderArtifact: string;
  readonly tokenizerArtifact: string;
  readonly resolveLoadOptions: (
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
    urls: Readonly<Record<string, string>>,
  ) => unknown;
}

function uniqueEntry(entries: readonly SpeechModelLocalEntry[], basename: string): SpeechModelLocalEntry | null {
  const matches = entries.filter((entry) => entry.basename.toLowerCase() === basename.toLowerCase());
  if (matches.length > 1) throw new Error(`Multiple local artifacts named ${basename}.`);
  return matches[0] ?? null;
}

export function createLocalArtifactAdapter(config: LocalArtifactAdapterConfig): BuiltInLocalModelAdapter {
  const inspect = (entries: readonly SpeechModelLocalEntry[]): SpeechModelLocalInspection => {
    const complete = config.required.every((name) => uniqueEntry(entries, name));
    return {
      encoderQuantizations: complete ? ['fp32'] : [],
      decoderQuantizations: complete ? ['fp32'] : [],
      tokenizerNames: complete ? [config.tokenizerArtifact] : [],
      preprocessorNames: [],
    };
  };
  return {
    preset: config.preset,
    createEntries(files) { return createLocalEntriesFromFiles(files); },
    async collectEntries(dirHandle: SpeechModelLocalDirectoryHandleLike, prefix = '', signal) {
      return await collectLocalEntriesFromDirectory(dirHandle, prefix, signal);
    },
    inspectEntries: inspect,
    async resolveEntries(options): Promise<ResolvedBuiltInLocalArtifacts> {
      if (options.entries.length === 0) throw new Error('Pick a local model folder first.');
      if ((options.encoderQuant && options.encoderQuant !== 'fp32') ||
          (options.decoderQuant && options.decoderQuant !== 'fp32')) {
        throw new Error(`This ${config.preset} bundle provides only FP32 graphs.`);
      }
      const entries = new Map<string, SpeechModelLocalEntry>();
      for (const name of config.required) {
        const entry = uniqueEntry(options.entries, name);
        if (!entry) throw new Error(`Missing local artifact: ${name}`);
        entries.set(name, entry);
      }
      const provider = createBlobAssetProvider();
      const handles: ResolvedBuiltInLocalArtifacts['assetHandles'][number][] = [];
      const urls: Record<string, string> = {};
      try {
        for (const [name, entry] of entries) {
          const file = await getLocalEntryFile(entry);
          const handle = await provider.resolve({
            id: entry.path,
            provider: 'blob',
            blob: file instanceof Blob ? file : new Blob([file]),
            signal: options.signal,
          });
          handles.push(handle);
          const url = await handle.getLocator('url');
          if (!url) throw new Error(`Could not create a URL locator for ${entry.path}.`);
          urls[name] = url;
        }
        return {
          modelId: options.modelId,
          preset: config.preset,
          builtInLoadOptions: config.resolveLoadOptions(options, urls),
          assetHandles: handles,
          selection: {
            encoderName: config.encoderArtifact,
            decoderName: config.decoderArtifact,
            tokenizerName: config.tokenizerArtifact,
            encoderQuant: 'fp32',
            decoderQuant: 'fp32',
          },
        };
      } catch (error) {
        await Promise.all(handles.map(async (handle) => handle.dispose()));
        throw error;
      }
    },
  };
}
