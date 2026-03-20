import type { NemoTokenizer } from '../nemo-common/index.js';
import { importNodeModule, isNodeLikeRuntime } from '../../io/node.js';

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
      `Failed to fetch tokenizer vocabulary from ${url}: ${response.status} ${response.statusText}`,
    );
  }

  return response.text();
}

export class ParakeetTokenizer implements NemoTokenizer {
  readonly kind = 'sentencepiece' as const;
  readonly vocabSize: number;
  readonly blankId: number;
  readonly controlTokenIds: ReadonlySet<number>;
  private readonly sanitizedTokens: readonly string[];
  private readonly tokenToId: ReadonlyMap<string, number>;

  constructor(
    readonly idToToken: readonly string[],
    options: {
      readonly blankId?: number;
    } = {},
  ) {
    this.vocabSize = idToToken.length;
    const discoveredBlankId = idToToken.findIndex((token) => token === '<blk>');
    this.blankId =
      options.blankId ?? (discoveredBlankId >= 0 ? discoveredBlankId : idToToken.length);
    this.sanitizedTokens = idToToken.map((token) => token.replace(/\u2581/g, ' '));
    this.controlTokenIds = new Set(
      idToToken.flatMap((token, index) =>
        /^<[^>\s]+>$/.test(token) && token !== '<blk>' ? [index] : [],
      ),
    );
    this.tokenToId = new Map(idToToken.map((token, index) => [token, index] as const));
  }

  static async fromUrl(
    url: string,
    options: {
      readonly blankId?: number;
    } = {},
  ): Promise<ParakeetTokenizer> {
    const text = await fetchText(url);
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);

    const indexedVocabulary = lines.every((line) => {
      const parts = line.trim().split(/\s+/);
      if (parts.length !== 2) {
        return false;
      }

      return Number.isInteger(Number.parseInt(parts[1] ?? '', 10));
    });
    const idToToken: string[] = [];

    if (indexedVocabulary) {
      for (const line of lines) {
        const [token, idText] = line.trim().split(/\s+/);
        const id = Number.parseInt(idText ?? '', 10);
        if (!token || !Number.isInteger(id) || id < 0) {
          continue;
        }

        idToToken[id] = token;
      }
    } else {
      idToToken.push(...lines.map((line) => line.trim()));
    }

    return new ParakeetTokenizer(idToToken, options);
  }

  getTokenId(token: string): number | undefined {
    return this.tokenToId.get(token);
  }

  isControlTokenId(id: number): boolean {
    return this.controlTokenIds.has(id);
  }

  decode(
    ids: readonly number[],
    options: {
      readonly skipControlTokens?: boolean;
    } = {},
  ): string {
    return ids
      .filter((id) => id !== this.blankId)
      .filter((id) => !options.skipControlTokens || !this.isControlTokenId(id))
      .map((id) => this.sanitizedTokens[id] ?? '')
      .join('')
      .replace(/^\s+/, '')
      .replace(/\s+(?=[^\w\s])/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  idsToTokens(ids: readonly number[]): readonly string[] {
    return ids.map((id) => this.idToToken[id] ?? '');
  }
}
