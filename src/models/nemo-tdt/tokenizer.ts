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
  private readonly sanitizedTokens: readonly string[];

  constructor(readonly idToToken: readonly string[]) {
    this.vocabSize = idToToken.length;
    this.blankId = Math.max(
      0,
      idToToken.findIndex((token) => token === '<blk>'),
    );
    this.sanitizedTokens = idToToken.map((token) => token.replace(/\u2581/g, ' '));
  }

  static async fromUrl(url: string): Promise<ParakeetTokenizer> {
    const text = await fetchText(url);
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
    const idToToken: string[] = [];

    for (const line of lines) {
      const [token, idText] = line.trim().split(/\s+/);
      const id = Number.parseInt(idText ?? '', 10);
      if (!token || !Number.isInteger(id) || id < 0) {
        continue;
      }

      idToToken[id] = token;
    }

    return new ParakeetTokenizer(idToToken);
  }

  decode(ids: readonly number[]): string {
    return ids
      .filter((id) => id !== this.blankId)
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
