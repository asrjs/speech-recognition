import { createBlobAssetProvider } from '../../io/index.js';
import { getModelConfig } from './catalog.js';
import { collectLocalEntriesFromDirectory, createLocalEntriesFromFiles, getLocalEntryFile } from '../../runtime/local-entry-files.js';
import type { BuiltInLocalModelAdapter, LoadSpeechModelFromLocalEntriesOptions, ResolvedBuiltInLocalArtifacts, SpeechModelLocalInspection } from '../../runtime/local-adapter-registry.js';
import type { SpeechModelLocalDirectoryHandleLike, SpeechModelLocalEntry } from '../../runtime/local-types.js';
import type { NemoAedModelConfig, NemoAedPromptSettings } from '../../models/nemo-aed/index.js';

const ENCODER = 'encoder-model.onnx';
const DECODER = 'decoder-model.onnx';
const TOKENIZER = 'tokenizer.json';
const CONFIG = 'config.json';
const GRAPH_FILES = [ENCODER, DECODER] as const;

interface CanaryRuntimeConfig extends Partial<NemoAedModelConfig> {
  readonly model_id?: string;
  readonly family?: string;
  readonly preset?: string;
  readonly classification?: { readonly family?: string };
  readonly sample_rate?: number;
  readonly mel_bins?: number;
  readonly frame_shift_seconds?: number;
  readonly subsampling_factor?: number;
  readonly encoder_hidden_size?: number;
  readonly decoder_hidden_size?: number;
  readonly encoder_output_size?: number;
  readonly max_target_positions?: number;
  readonly languages?: readonly string[];
  readonly prompt_format?: string;
  readonly prompt_defaults?: readonly Record<string, unknown>[];
  readonly preprocessor_valid_length_mode?: 'onnx' | 'centered';
  readonly preprocessor_normalization?: 'per_feature' | 'none';
  readonly external_data?: Readonly<{ encoder?: string | null; decoder?: string | null }>;
}

interface CanaryTokenizerFile {
  readonly kind?: string;
  readonly version?: number;
  readonly prompt_format?: string;
  readonly vocab_size?: number;
  readonly langs?: readonly string[];
  readonly language_codes?: readonly string[];
  readonly bos_id?: number;
  readonly eos_id?: number;
  readonly pad_id?: number;
  readonly special_tokens?: Readonly<Record<string, number>>;
  readonly subtokenizers?: Readonly<Record<string, { readonly offset?: number; readonly size?: number; readonly pieces?: readonly string[] }>>;
}

function uniqueEntry(entries: readonly SpeechModelLocalEntry[], name: string): SpeechModelLocalEntry | null {
  const matches = entries.filter((entry) => entry.basename.toLowerCase() === name.toLowerCase());
  if (matches.length > 1) throw new Error(`Multiple local Canary artifacts named ${name}.`);
  return matches[0] ?? null;
}

function inspectCanaryEntries(entries: readonly SpeechModelLocalEntry[]): SpeechModelLocalInspection {
  const hasGraphs = GRAPH_FILES.every((name) => Boolean(uniqueEntry(entries, name)));
  const tokenizer = uniqueEntry(entries, TOKENIZER);
  const preprocessor = uniqueEntry(entries, 'nemo128.onnx');
  return {
    encoderQuantizations: hasGraphs ? ['fp32'] : [],
    decoderQuantizations: hasGraphs ? ['fp32'] : [],
    tokenizerNames: tokenizer ? [tokenizer.basename] : [],
    preprocessorNames: preprocessor ? ['nemo128'] : [],
  };
}

async function readJson<T>(entry: SpeechModelLocalEntry): Promise<T> {
  const file = await getLocalEntryFile(entry);
  try {
    return JSON.parse(await file.text()) as T;
  } catch (error) {
    throw new Error(`Invalid JSON in local Canary artifact ${entry.path}: ${String(error)}`);
  }
}

function requiredPositiveInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || Number(value) <= 0) throw new Error(`Invalid Canary ${label}.`);
  return Number(value);
}

function tokenizerPieceAtId(ranges: readonly NonNullable<CanaryTokenizerFile['subtokenizers']>[string][], id: number): string | undefined {
  const range = ranges.find((candidate) => id >= Number(candidate.offset) && id < Number(candidate.offset) + Number(candidate.size));
  return range?.pieces?.[id - Number(range.offset)];
}

function localParentPath(entry: SpeechModelLocalEntry): string {
  return entry.path.replace(/\\/g, '/').split('/').slice(0, -1).join('/').toLowerCase();
}

function mapPromptDefaults(
  values: readonly Record<string, unknown>[] | undefined,
  promptFormat: 'canary' | 'canary2',
): readonly NemoAedPromptSettings[] {
  if (!values?.length) throw new Error('Canary config.json is missing prompt_defaults.');
  return values.map((value) => {
    if (typeof value.source_lang !== 'string' || typeof value.target_lang !== 'string' ||
      typeof value.decoder_context !== 'string' || typeof value.pnc !== 'boolean') {
      throw new Error('Canary config.json has invalid prompt_defaults.');
    }
    if (promptFormat === 'canary2' &&
      (typeof value.emotion !== 'string' || typeof value.itn !== 'boolean' ||
        typeof value.timestamp !== 'boolean' || typeof value.diarize !== 'boolean')) {
      throw new Error('Canary2 config.json has invalid prompt_defaults.');
    }
    const rawTask = value.task;
    let task: NemoAedPromptSettings['task'];
    if (rawTask === undefined) {
      task = promptFormat === 'canary' ? 'asr' : undefined;
    } else if (typeof rawTask === 'string') {
      const normalized = rawTask.trim().toLowerCase();
      task = normalized === 'asr' || normalized === 'transcribe'
        ? 'asr'
        : normalized === 'ast' || normalized === 'translate' || normalized === 'translation' ||
            normalized === 'speech-translation' || normalized === 's2t_translation'
          ? 'translation'
          : undefined;
      if (!task) throw new Error(`Canary config.json has invalid prompt task "${rawTask}".`);
    } else {
      throw new Error('Canary config.json has invalid prompt task.');
    }
    return {
      sourceLanguage: value.source_lang,
      targetLanguage: value.target_lang,
      ...(task ? { task } : {}),
      decoderContext: value.decoder_context,
      emotion: typeof value.emotion === 'string' ? value.emotion : '<|emo:undefined|>',
      punctuate: value.pnc,
      inverseTextNormalization: typeof value.itn === 'boolean' ? value.itn : false,
      timestamps: typeof value.timestamp === 'boolean' ? value.timestamp : false,
      diarize: typeof value.diarize === 'boolean' ? value.diarize : false,
    };
  });
}

function validateArtifacts(
  modelId: string,
  metadata: CanaryRuntimeConfig,
  tokenizer: CanaryTokenizerFile,
): NemoAedModelConfig {
  const expected = getModelConfig(modelId);
  if (!expected || metadata.model_id !== modelId) throw new Error('Local Canary config model_id does not match the selected model.');
  if (metadata.family !== 'nemo-aed' || metadata.preset !== 'canary' || metadata.classification?.family !== 'canary') {
    throw new Error('Local config.json is not a Canary NeMo AED export.');
  }
  const sampleRate = requiredPositiveInteger(metadata.sample_rate, 'sample_rate');
  const melBins = requiredPositiveInteger(metadata.mel_bins, 'mel_bins');
  const vocabularySize = requiredPositiveInteger(tokenizer.vocab_size, 'tokenizer vocab_size');
  if (sampleRate !== 16000 || melBins !== expected.featuresSize) throw new Error('Local Canary sample rate or mel-bin count does not match the selected preset.');
  const promptFormat = metadata.prompt_format;
  if (promptFormat !== expected.promptFormat || tokenizer.prompt_format !== promptFormat) {
    throw new Error(`Local Canary prompt format must match the selected preset (${expected.promptFormat}).`);
  }
  if (tokenizer.kind !== 'canary-aggregate-tokenizer' || tokenizer.version !== 1 || vocabularySize !== expected.vocabSize) {
    throw new Error('Local Canary tokenizer vocabulary does not match the selected preset.');
  }
  const languages = metadata.languages;
  const tokenizerLanguages = tokenizer.language_codes;
  if (!languages?.length || !tokenizerLanguages?.length ||
    languages.length !== expected.languages.length ||
    tokenizerLanguages.length !== languages.length ||
    languages.some((language, index) => language !== expected.languages[index] || tokenizerLanguages[index] !== language)) {
    throw new Error('Local Canary language metadata does not match the selected preset.');
  }
  const ranges = Object.values(tokenizer.subtokenizers ?? {});
  const orderedRanges = [...ranges].sort((left, right) => Number(left.offset) - Number(right.offset));
  let nextOffset = 0;
  if (orderedRanges.length === 0 || orderedRanges.some((range) => {
    const valid = Number.isSafeInteger(range.offset) && Number.isSafeInteger(range.size) &&
      Number(range.offset) === nextOffset && Number(range.size) > 0 && range.pieces?.length === range.size;
    if (valid) nextOffset += Number(range.size);
    return !valid;
  }) || nextOffset !== vocabularySize) {
    throw new Error('Local Canary tokenizer ranges must cover the full vocabulary without gaps or overlap.');
  }
  for (const [label, id] of Object.entries({ bos_id: tokenizer.bos_id, eos_id: tokenizer.eos_id, pad_id: tokenizer.pad_id })) {
    if (!Number.isSafeInteger(id) || Number(id) < 0 || Number(id) >= vocabularySize) throw new Error(`Invalid Canary tokenizer ${label}.`);
  }
  const specialTokens = tokenizer.special_tokens;
  if (!specialTokens || Object.values(specialTokens).some((id) => !Number.isSafeInteger(id) || id < 0 || id >= vocabularySize) ||
    new Set(Object.values(specialTokens)).size !== Object.keys(specialTokens).length) {
    throw new Error('Invalid Canary tokenizer special_tokens.');
  }
  const requiredSpecialTokens = promptFormat === 'canary'
    ? [
        '<|startoftranscript|>', '<|transcribe|>', '<|translate|>', '<|pnc|>', '<|nopnc|>',
        '<|endoftext|>', '<|nospeech|>', '<pad>', ...languages.map((language) => `<|${language}|>`),
      ]
    : [
        '<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>', '<|pnc|>', '<|nopnc|>',
        '<|itn|>', '<|noitn|>', '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
        '<|endoftext|>', '<|nospeech|>', '<pad>', ...languages.map((language) => `<|${language}|>`),
      ];
  if (requiredSpecialTokens.some((token) => !Object.prototype.hasOwnProperty.call(specialTokens, token))) {
    throw new Error('Local Canary tokenizer is missing required prompt tokens.');
  }
  for (const [token, id] of Object.entries(specialTokens)) {
    if (tokenizerPieceAtId(orderedRanges, id) !== token) {
      throw new Error(`Canary special token ID does not point to its token piece: ${token}.`);
    }
  }
  const vocabularySpecialTokens = new Set(
    orderedRanges.flatMap((range) => (range.pieces ?? []).filter((piece) =>
      piece === '<pad>' || (piece.startsWith('<|') && piece.endsWith('|>')),
    )),
  );
  if (vocabularySpecialTokens.size !== Object.keys(specialTokens).length ||
    [...vocabularySpecialTokens].some((token) => !Object.prototype.hasOwnProperty.call(specialTokens, token))) {
    throw new Error('Canary special_tokens map does not cover every special token in the vocabulary.');
  }
  if (tokenizer.bos_id !== specialTokens['<|startoftranscript|>'] ||
    tokenizer.eos_id !== specialTokens['<|endoftext|>'] || tokenizer.pad_id !== specialTokens['<pad>']) {
    throw new Error('Canary BOS, EOS, and PAD IDs do not match tokenizer special tokens.');
  }
  if (typeof metadata.frame_shift_seconds !== 'number' || !Number.isFinite(metadata.frame_shift_seconds) || metadata.frame_shift_seconds <= 0 ||
    !Number.isSafeInteger(metadata.subsampling_factor) || Number(metadata.subsampling_factor) <= 0 ||
    !Number.isSafeInteger(metadata.encoder_hidden_size) || Number(metadata.encoder_hidden_size) <= 0 ||
    !Number.isSafeInteger(metadata.decoder_hidden_size) || Number(metadata.decoder_hidden_size) <= 0 ||
    !Number.isSafeInteger(metadata.encoder_output_size) || Number(metadata.encoder_output_size) <= 0 ||
    !Number.isSafeInteger(metadata.max_target_positions) || Number(metadata.max_target_positions) <= 0) {
    throw new Error('Local Canary config.json is missing model runtime dimensions.');
  }
  if (metadata.preprocessor_valid_length_mode !== 'onnx' && metadata.preprocessor_valid_length_mode !== 'centered') {
    throw new Error('Local Canary config.json is missing preprocessor_valid_length_mode.');
  }
  if (metadata.preprocessor_normalization !== 'per_feature' && metadata.preprocessor_normalization !== 'none') {
    throw new Error('Local Canary config.json is missing preprocessor_normalization.');
  }
  return {
    ecosystem: 'nemo', architecture: 'nemo-aed', encoderArchitecture: 'fastconformer',
    decoderArchitecture: 'transformer-decoder', sampleRate, frameShiftSeconds: metadata.frame_shift_seconds,
    subsamplingFactor: Number(metadata.subsampling_factor), melBins,
    preprocessorValidLengthMode: metadata.preprocessor_valid_length_mode,
    preprocessorNormalization: metadata.preprocessor_normalization,
    vocabularySize, languages, encoderHiddenSize: Number(metadata.encoder_hidden_size),
    decoderHiddenSize: Number(metadata.decoder_hidden_size), encoderOutputSize: Number(metadata.encoder_output_size),
    maxTargetPositions: Number(metadata.max_target_positions), promptFormat,
    promptDefaults: mapPromptDefaults(metadata.prompt_defaults, promptFormat),
    tokenizer: { kind: 'aggregate', vocabSize: vocabularySize, bosTokenId: tokenizer.bos_id, eosTokenId: tokenizer.eos_id, padTokenId: tokenizer.pad_id },
  };
}

export const canaryBuiltInLocalModelAdapter: BuiltInLocalModelAdapter = {
  preset: 'canary',
  createEntries(files) { return createLocalEntriesFromFiles(files); },
  async collectEntries(dirHandle: SpeechModelLocalDirectoryHandleLike, prefix = '', signal) {
    return await collectLocalEntriesFromDirectory(dirHandle, prefix, signal);
  },
  inspectEntries(entries) { return inspectCanaryEntries(entries); },
  async resolveEntries(options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string }): Promise<ResolvedBuiltInLocalArtifacts> {
    if (!options.entries.length) throw new Error('Pick a local model folder first.');
    const encoder = uniqueEntry(options.entries, ENCODER);
    const decoder = uniqueEntry(options.entries, DECODER);
    const tokenizerEntry = uniqueEntry(options.entries, TOKENIZER);
    const configEntry = uniqueEntry(options.entries, CONFIG);
    if (!encoder || !decoder || !tokenizerEntry || !configEntry) throw new Error('Canary local bundle requires encoder-model.onnx, decoder-model.onnx, tokenizer.json, and config.json.');
    const bundleParent = localParentPath(encoder);
    if ([decoder, tokenizerEntry, configEntry].some((entry) => localParentPath(entry) !== bundleParent)) {
      throw new Error('Canary graphs, tokenizer, and config must share one bundle directory.');
    }
    if ((options.encoderQuant && options.encoderQuant !== 'fp32') || (options.decoderQuant && options.decoderQuant !== 'fp32')) {
      throw new Error('Canary local export contains fp32 graphs only.');
    }
    if (options.tokenizerName && options.tokenizerName !== TOKENIZER) throw new Error(`Canary local export only provides ${TOKENIZER}.`);
    if (options.preprocessorName && options.preprocessorName !== 'nemo128') throw new Error('Canary local export only supports the nemo128 preprocessor.');
    for (const entry of options.entries) {
      const path = entry.path.replace(/\\/g, '/');
      if (path.startsWith('/') || /^[a-z]+:/i.test(path) || path.split('/').some((part) => part === '..')) throw new Error(`Unsafe local Canary artifact path: ${entry.path}`);
    }
    const metadata = await readJson<CanaryRuntimeConfig>(configEntry);
    const tokenizer = await readJson<CanaryTokenizerFile>(tokenizerEntry);
    const config = validateArtifacts(options.modelId, metadata, tokenizer);
    const preprocessorBackend = options.preprocessorBackend ?? 'js';
    const preprocessorEntry = preprocessorBackend === 'onnx' ? uniqueEntry(options.entries, 'nemo128.onnx') : null;
    if (preprocessorBackend === 'onnx' && !preprocessorEntry) {
      throw new Error('Canary ONNX preprocessing was selected, but nemo128.onnx is missing.');
    }
    if (preprocessorEntry && localParentPath(preprocessorEntry) !== bundleParent) {
      throw new Error('Canary preprocessor must be in the same bundle directory as its graphs.');
    }
    const entriesBySidecar = new Map<string, SpeechModelLocalEntry>();
    for (const [role, graphName] of [['encoder', ENCODER], ['decoder', DECODER]] as const) {
      const sidecarName = `${graphName}.data`;
      if (!metadata.external_data || !Object.prototype.hasOwnProperty.call(metadata.external_data, role)) {
        throw new Error('Canary config.json is missing external_data declarations.');
      }
      const declared = metadata.external_data[role];
      const sidecar = uniqueEntry(options.entries, sidecarName);
      if (declared !== undefined && declared !== null && declared !== sidecarName) throw new Error(`Unsafe or unsupported external-data path for ${graphName}.`);
      if (declared && !sidecar) throw new Error(`Missing declared external data for ${graphName}.`);
      if (!declared && sidecar) throw new Error(`Undeclared external-data sidecar for ${graphName}.`);
      if (sidecar && (sidecar.basename !== sidecarName || localParentPath(sidecar) !== bundleParent)) {
        throw new Error(`Canary sidecar for ${graphName} must use its declared filename in the bundle directory.`);
      }
      if (sidecar) entriesBySidecar.set(graphName, sidecar);
    }
    if (Object.keys(metadata.external_data ?? {}).some((key) => key !== 'encoder' && key !== 'decoder')) {
      throw new Error('Canary config.json contains unsupported external_data entries.');
    }
    const knownSidecars = new Set(GRAPH_FILES.map((name) => `${name}.data`));
    if (options.entries.some((entry) => entry.basename.toLowerCase().endsWith('.onnx.data') && !knownSidecars.has(entry.basename.toLowerCase()))) {
      throw new Error('Unsupported or ambiguous external-data sidecar in Canary bundle.');
    }
    const provider = createBlobAssetProvider();
    const assetHandles: ResolvedBuiltInLocalArtifacts['assetHandles'][number][] = [];
    const toLocator = async (entry: SpeechModelLocalEntry): Promise<string> => {
      if (options.signal?.aborted) throw new Error('Local model load aborted.');
      const file = await getLocalEntryFile(entry);
      const handle = await provider.resolve({ id: entry.path, provider: 'blob', blob: file instanceof Blob ? file : new Blob([file]), signal: options.signal });
      assetHandles.push(handle);
      const locator = await handle.getLocator('url');
      if (!locator) throw new Error(`Could not create a URL locator for local asset "${entry.path}".`);
      return locator;
    };
    try {
      const encoderUrl = await toLocator(encoder);
      const decoderUrl = await toLocator(decoder);
      const encoderDataUrl = entriesBySidecar.has(ENCODER) ? await toLocator(entriesBySidecar.get(ENCODER)!) : undefined;
      const decoderDataUrl = entriesBySidecar.has(DECODER) ? await toLocator(entriesBySidecar.get(DECODER)!) : undefined;
      return {
        modelId: options.modelId,
        preset: 'canary',
        builtInLoadOptions: {
          config,
          source: { kind: 'direct', preprocessorBackend, encoderBackend: options.encoderBackend, decoderBackend: options.decoderBackend,
            artifacts: { encoderUrl, decoderUrl, tokenizerUrl: await toLocator(tokenizerEntry), configUrl: await toLocator(configEntry),
              ...(preprocessorEntry ? { preprocessorUrl: await toLocator(preprocessorEntry), preprocessorFilename: 'nemo128.onnx' } : {}),
              ...(encoderDataUrl ? { encoderDataUrl } : {}), ...(decoderDataUrl ? { decoderDataUrl } : {}),
              encoderFilename: ENCODER, decoderFilename: DECODER, tokenizerFilename: TOKENIZER, configFilename: CONFIG } },
          preprocessorBackend, cpuThreads: options.cpuThreads, wasmPaths: options.wasmPaths, enableProfiling: options.enableProfiling,
        },
        assetHandles,
        selection: { encoderName: ENCODER, decoderName: DECODER, tokenizerName: TOKENIZER,
          ...(preprocessorEntry ? { preprocessorName: preprocessorEntry.basename } : {}), encoderQuant: 'fp32', decoderQuant: 'fp32' },
      };
    } catch (error) {
      await Promise.all(assetHandles.map((handle) => handle.dispose()));
      throw error;
    }
  },
};
