import type { NemoTokenizer } from '../nemo-common/index.js';
import { importNodeModule, isNodeLikeRuntime } from '../../io/node.js';
import type {
  NemoAedModelConfig,
  NemoAedPromptSettings,
  NemoAedTranscriptionOptions,
} from './types.js';

export interface CanaryAggregateSubtokenizerSpec {
  readonly offset: number;
  readonly size: number;
  readonly pieces: readonly string[];
}

export interface CanaryTokenizerPayload {
  readonly kind: 'canary-aggregate-tokenizer';
  readonly version: number;
  readonly prompt_format: string;
  readonly vocab_size: number;
  readonly langs: readonly string[];
  readonly language_codes: readonly string[];
  readonly bos_id: number;
  readonly eos_id: number;
  readonly pad_id: number;
  readonly special_tokens: Readonly<Record<string, number>>;
  readonly subtokenizers: Readonly<Record<string, CanaryAggregateSubtokenizerSpec>>;
}

interface TokenRange {
  readonly start: number;
  readonly end: number;
  readonly pieces: readonly string[];
}

async function fetchText(url: string): Promise<string> {
  if (isNodeLikeRuntime() && /^file:/i.test(url)) {
    const [{ fileURLToPath }, fs] = await Promise.all([
      importNodeModule<typeof import('node:url')>('node:url'),
      importNodeModule<typeof import('node:fs/promises')>('node:fs/promises'),
    ]);
    return fs.readFile(fileURLToPath(url), 'utf8');
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch Canary tokenizer metadata from ${url}: ${response.status} ${response.statusText}`,
    );
  }

  return response.text();
}

function normalizeLanguageToken(language: string): string {
  const value = String(language).trim();
  if (value.startsWith('<|') && value.endsWith('|>')) {
    return value;
  }
  return `<|${value}|>`;
}

function normalizeEmotionToken(emotion: string): string {
  const value = String(emotion).trim();
  if (value.startsWith('<|') && value.endsWith('|>')) {
    return value;
  }
  if (value.startsWith('emo:')) {
    return `<|${value}|>`;
  }
  return `<|emo:${value}|>`;
}

function normalizeBooleanToken(name: string, value: boolean): string {
  return value ? `<|${name}|>` : `<|no${name}|>`;
}

function normalizeBooleanOption(
  value: boolean | string | undefined,
  fallback: boolean,
  fieldName: string,
): boolean {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value !== 'string') {
    return fallback;
  }

  const normalized = value.trim().toLowerCase();
  if (normalized === 'yes' || normalized === 'true') {
    return true;
  }
  if (normalized === 'no' || normalized === 'false') {
    return false;
  }

  throw new Error(
    `Unsupported Canary boolean option "${fieldName}" value "${value}". Use boolean, "yes"/"no", or "true"/"false".`,
  );
}

function normalizeTaskOption(task: NemoAedTranscriptionOptions['task']): 'asr' | 'translation' | null {
  if (!task) {
    return null;
  }

  const normalized = String(task).trim().toLowerCase();
  if (normalized === 'asr') {
    return 'asr';
  }
  if (
    normalized === 'ast' ||
    normalized === 'translate' ||
    normalized === 'translation' ||
    normalized === 'speech-translation'
  ) {
    return 'translation';
  }

  return null;
}

function normalizeDecodedPiece(piece: string): string {
  return piece.replace(/\u2581/g, ' ').replace(/^\s+/, '');
}

function normalizeDecodedText(text: string): string {
  return text.replace(/\u2581/g, ' ').replace(/\s+(?=[!,.?:;])/g, '').replace(/\s+/g, ' ').trim();
}

export class CanaryTokenizer implements NemoTokenizer {
  readonly kind = 'aggregate' as const;
  readonly vocabSize: number;
  readonly bosId: number;
  readonly eosId: number;
  readonly padId: number;
  readonly promptFormat: string;
  readonly langs: readonly string[];
  readonly languageCodes: readonly string[];
  readonly specialTokens: Readonly<Record<string, number>>;
  private readonly reverseSpecialTokens = new Map<number, string>();
  private readonly tokenRanges: readonly TokenRange[];
  private readonly specialIdSet: ReadonlySet<number>;

  constructor(readonly payload: CanaryTokenizerPayload) {
    this.vocabSize = payload.vocab_size;
    this.bosId = payload.bos_id;
    this.eosId = payload.eos_id;
    this.padId = payload.pad_id;
    this.promptFormat = payload.prompt_format;
    this.langs = payload.langs;
    this.languageCodes = payload.language_codes;
    this.specialTokens = payload.special_tokens;
    for (const [token, id] of Object.entries(payload.special_tokens)) {
      this.reverseSpecialTokens.set(id, token);
    }
    this.specialIdSet = new Set(Object.values(payload.special_tokens));
    this.tokenRanges = Object.values(payload.subtokenizers)
      .map((spec) => ({
        start: spec.offset,
        end: spec.offset + spec.size,
        pieces: spec.pieces,
      }))
      .sort((left, right) => left.start - right.start);
  }

  static fromPayload(payload: CanaryTokenizerPayload): CanaryTokenizer {
    return new CanaryTokenizer(payload);
  }

  static async fromUrl(url: string): Promise<CanaryTokenizer> {
    const payload = JSON.parse(await fetchText(url)) as CanaryTokenizerPayload;
    return CanaryTokenizer.fromPayload(payload);
  }

  isSpecialId(id: number): boolean {
    return this.specialIdSet.has(id);
  }

  decode(ids: readonly number[]): string {
    const pieces: string[] = [];
    for (const id of ids) {
      if (this.isSpecialId(id)) {
        continue;
      }
      const token = this.resolveToken(id);
      if (token) {
        pieces.push(token);
      }
    }
    return normalizeDecodedText(pieces.join(''));
  }

  idsToTokens(ids: readonly number[]): readonly string[] {
    return ids.map((id) => this.resolveToken(id) ?? '');
  }

  resolvePromptSettings(
    config: NemoAedModelConfig,
    options: NemoAedTranscriptionOptions = {},
  ): NemoAedPromptSettings {
    const defaults = config.promptDefaults[0];
    const fallbackSource = defaults?.sourceLanguage ?? 'en';
    const task = normalizeTaskOption(options.task);
    const requestedSource = options.sourceLanguage ?? options.source_lang ?? options.language;
    const fallbackTarget =
      task === 'asr'
        ? requestedSource ?? defaults?.targetLanguage ?? fallbackSource
        : defaults?.targetLanguage ?? fallbackSource;
    const resolvedSource = requestedSource ?? fallbackSource;
    const resolvedTarget =
      options.targetLanguage ??
      options.target_lang ??
      (task === 'asr' ? resolvedSource : options.language) ??
      fallbackTarget;
    return {
      sourceLanguage: resolvedSource,
      targetLanguage: resolvedTarget,
      decoderContext: options.decoderContext ?? defaults?.decoderContext ?? '',
      emotion: normalizeEmotionToken(options.emotion ?? defaults?.emotion ?? 'undefined'),
      punctuate: normalizeBooleanOption(
        options.punctuate ?? options.pnc,
        defaults?.punctuate ?? true,
        'pnc',
      ),
      inverseTextNormalization:
        options.inverseTextNormalization ?? defaults?.inverseTextNormalization ?? false,
      timestamps: normalizeBooleanOption(
        options.timestamps ?? options.timestamp,
        defaults?.timestamps ?? false,
        'timestamp',
      ),
      diarize: options.diarize ?? defaults?.diarize ?? false,
    };
  }

  buildPromptIds(settings: NemoAedPromptSettings): readonly number[] {
    if (settings.decoderContext.trim().length > 0) {
      throw new Error(
        'Canary decoder_context prompt text is not supported yet. Use the default empty decoder context.',
      );
    }

    // Canary `canary2` prompt layout:
    //   <|startofcontext|>
    //   <|startoftranscript|>
    //   <emotion>
    //   <source_lang>
    //   <target_lang>
    //   <pnc toggle>
    //   <itn toggle>
    //   <timestamp toggle>
    //   <diarize toggle>
    return [
      this.requireTokenId('<|startofcontext|>'),
      this.requireTokenId('<|startoftranscript|>'),
      this.requireTokenId(normalizeEmotionToken(settings.emotion)),
      this.requireTokenId(normalizeLanguageToken(settings.sourceLanguage)),
      this.requireTokenId(normalizeLanguageToken(settings.targetLanguage)),
      this.requireTokenId(normalizeBooleanToken('pnc', settings.punctuate)),
      this.requireTokenId(normalizeBooleanToken('itn', settings.inverseTextNormalization)),
      this.requireTokenId(normalizeBooleanToken('timestamp', settings.timestamps)),
      this.requireTokenId(normalizeBooleanToken('diarize', settings.diarize)),
    ];
  }

  toNativeTokenText(piece: string): string {
    return normalizeDecodedPiece(piece);
  }

  private requireTokenId(token: string): number {
    const id = this.specialTokens[token];
    if (typeof id !== 'number') {
      throw new Error(`Canary tokenizer is missing special token "${token}".`);
    }
    return id;
  }

  private resolveToken(id: number): string | undefined {
    const special = this.reverseSpecialTokens.get(id);
    if (special) {
      return special;
    }
    for (const range of this.tokenRanges) {
      if (id >= range.start && id < range.end) {
        return range.pieces[id - range.start];
      }
    }
    return undefined;
  }
}
