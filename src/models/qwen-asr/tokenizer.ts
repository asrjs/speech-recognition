import { isNodeLikeRuntime, importNodeModule } from '../../io/node.js';
import type { TextTokenizer } from '../../tokenizers/index.js';

interface QwenTokenizerJson {
  readonly model?: {
    readonly vocab?: Record<string, number>;
    readonly merges?: readonly string[];
  };
  readonly added_tokens?: ReadonlyArray<{
    readonly id: number;
    readonly content: string;
    readonly special?: boolean;
  }>;
}

async function fetchText(url: string): Promise<string> {
  if (isNodeLikeRuntime()) {
    const [{ fileURLToPath }, fs] = await Promise.all([
      importNodeModule<typeof import('node:url')>('node:url'),
      importNodeModule<typeof import('node:fs/promises')>('node:fs/promises'),
    ]);
    if (/^file:/i.test(url)) return fs.readFile(fileURLToPath(url), 'utf8');
    if (/^(?:\/|[A-Za-z]:\\|\.\.?[\\/])/.test(url)) {
      const { existsSync } = await importNodeModule<typeof import('node:fs')>('node:fs');
      if (existsSync(url)) return fs.readFile(url, 'utf8');
    }
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch Qwen tokenizer from ${url}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

function createByteToUnicodeMap(): ReadonlyMap<number, string> {
  const bytes: number[] = [];
  for (let value = 33; value <= 126; value += 1) bytes.push(value);
  for (let value = 161; value <= 172; value += 1) bytes.push(value);
  for (let value = 174; value <= 255; value += 1) bytes.push(value);
  const codePoints = bytes.slice();
  let extra = 0;
  for (let value = 0; value < 256; value += 1) {
    if (!bytes.includes(value)) {
      bytes.push(value);
      codePoints.push(256 + extra);
      extra += 1;
    }
  }
  return new Map(bytes.map((value, index) => [value, String.fromCharCode(codePoints[index]!) ]));
}

function createUnicodeToByteMap(byteToUnicode: ReadonlyMap<number, string>): ReadonlyMap<string, number> {
  return new Map([...byteToUnicode].map(([byte, character]) => [character, byte] as const));
}

const GPT2_REGEX = new RegExp(
  "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
  'gu',
);

export class Qwen3AsrTokenizer implements TextTokenizer {
  readonly kind = 'bpe' as const;
  readonly vocabSize: number;
  private readonly idToToken: ReadonlyMap<number, string>;
  private readonly tokenToId: ReadonlyMap<string, number>;
  private readonly specialTokenIds: ReadonlySet<number>;
  private readonly specialTokens: readonly string[];
  private readonly byteToUnicode: ReadonlyMap<number, string>;
  private readonly unicodeToByte: ReadonlyMap<string, number>;
  private readonly bpeMerges: ReadonlyMap<string, number>;

  constructor(data: QwenTokenizerJson) {
    const idToToken = new Map<number, string>();
    const tokenToId = new Map<string, number>();
    const specialIds = new Set<number>();
    for (const [token, id] of Object.entries(data.model?.vocab ?? {})) {
      idToToken.set(id, token);
      tokenToId.set(token, id);
    }
    for (const entry of data.added_tokens ?? []) {
      idToToken.set(entry.id, entry.content);
      tokenToId.set(entry.content, entry.id);
      if (entry.special || /^<[^>]+>$/.test(entry.content)) specialIds.add(entry.id);
    }
    this.idToToken = idToToken;
    this.tokenToId = tokenToId;
    this.specialTokenIds = specialIds;
    this.specialTokens = [...specialIds]
      .map((id) => idToToken.get(id))
      .filter((token): token is string => token !== undefined)
      .sort((left, right) => right.length - left.length);
    this.vocabSize = Math.max(...idToToken.keys(), 0) + 1;
    this.byteToUnicode = createByteToUnicodeMap();
    this.unicodeToByte = createUnicodeToByteMap(this.byteToUnicode);
    this.bpeMerges = new Map(
      (data.model?.merges ?? []).map((merge, index) => [merge, index] as const),
    );
  }

  static fromJson(text: string): Qwen3AsrTokenizer {
    return new Qwen3AsrTokenizer(JSON.parse(text) as QwenTokenizerJson);
  }

  static async fromUrl(url: string): Promise<Qwen3AsrTokenizer> {
    return Qwen3AsrTokenizer.fromJson(await fetchText(url));
  }

  getTokenId(token: string): number | undefined {
    return this.tokenToId.get(token);
  }

  isSpecialTokenId(id: number): boolean {
    return this.specialTokenIds.has(id);
  }

  idsToTokens(ids: readonly number[]): readonly string[] {
    return ids.map((id) => this.idToToken.get(id) ?? '');
  }

  encode(text: string): readonly number[] {
    const ids: number[] = [];
    let cursor = 0;
    while (cursor < text.length) {
      const special = this.specialTokens.find((token) => text.startsWith(token, cursor));
      if (special) {
        const id = this.tokenToId.get(special);
        if (id !== undefined) ids.push(id);
        cursor += special.length;
        continue;
      }
      const nextSpecial = this.specialTokens
        .map((token) => text.indexOf(token, cursor))
        .filter((index) => index >= 0)
        .sort((left, right) => left - right)[0] ?? text.length;
      if (nextSpecial > cursor) ids.push(...this.encodePlainText(text.slice(cursor, nextSpecial)));
      cursor = nextSpecial;
    }
    return ids;
  }

  private encodePlainText(text: string): number[] {
    const ids: number[] = [];
    for (const token of text.match(GPT2_REGEX) ?? []) {
      const bytes = new TextEncoder().encode(token);
      let word = '';
      for (const byte of bytes) word += this.byteToUnicode.get(byte) ?? '\uFFFD';
      let symbols = Array.from(word);
      while (symbols.length > 1) {
        let bestRank = Infinity;
        let bestIndex = -1;
        for (let index = 0; index < symbols.length - 1; index += 1) {
          const left = symbols[index];
          const right = symbols[index + 1];
          if (left === undefined || right === undefined) continue;
          const rank = this.bpeMerges.get(`${left} ${right}`);
          if (rank !== undefined && rank < bestRank) {
            bestRank = rank;
            bestIndex = index;
          }
        }
        if (bestIndex < 0) break;
        symbols = [
          ...symbols.slice(0, bestIndex),
          `${symbols[bestIndex]!}${symbols[bestIndex + 1]!}`,
          ...symbols.slice(bestIndex + 2),
        ];
      }
      for (const symbol of symbols) {
        const id = this.tokenToId.get(symbol);
        if (id !== undefined) ids.push(id);
      }
    }
    return ids;
  }

  decode(ids: readonly number[], options: { readonly skipSpecialTokens?: boolean } = {}): string {
    const pieces: string[] = [];
    for (const id of ids) {
      if (options.skipSpecialTokens && this.isSpecialTokenId(id)) continue;
      const token = this.idToToken.get(id);
      if (token !== undefined) pieces.push(token);
    }
    const bytes: number[] = [];
    for (const character of Array.from(pieces.join(''))) {
      const byte = this.unicodeToByte.get(character);
      if (byte !== undefined) bytes.push(byte);
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }
}
