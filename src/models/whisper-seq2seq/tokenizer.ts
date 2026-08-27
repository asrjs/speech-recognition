import { isNodeLikeRuntime, importNodeModule } from '../../io/node-compat.js';
import type { TextTokenizer } from '../../tokenizers/index.js';
import {
  fetchTextHonoringAbort,
  rethrowIfAssetAborted,
  throwIfAssetAborted,
  type AssetAbortSignalLike,
} from '../../io/abort.js';

interface WhisperTokenizerJson {
  readonly model?: {
    readonly type?: string;
    readonly vocab?: Record<string, number>;
    readonly merges?: readonly string[];
  };
  readonly added_tokens?: ReadonlyArray<{
    readonly id: number;
    readonly content: string;
    readonly special?: boolean;
  }>;
}

export async function fetchText(
  url: string,
  signal?: AssetAbortSignalLike | null,
): Promise<string> {
  throwIfAssetAborted(signal);
  if (isNodeLikeRuntime()) {
    const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
    const fs = await importNodeModule<typeof import('node:fs/promises')>('node:fs/promises');

    if (/^file:/i.test(url)) {
      try {
        const text = await fs.readFile(fileURLToPath(url), 'utf8');
        throwIfAssetAborted(signal);
        return text;
      } catch (error) {
        rethrowIfAssetAborted(error);
        throw error;
      }
    }

    // Handle bare file paths (no protocol) — check filesystem directly
    if (/^(?:\/|[A-Za-z]:\\|\.\.?[\\/])/.test(url)) {
      const { existsSync } = await importNodeModule<typeof import('node:fs')>('node:fs');
      if (existsSync(url)) {
        try {
          const text = await fs.readFile(url, 'utf8');
          throwIfAssetAborted(signal);
          return text;
        } catch (error) {
          rethrowIfAssetAborted(error);
          throw error;
        }
      }
    }
  }
  return fetchTextHonoringAbort(url, signal, {
    errorMessage: `Failed to fetch tokenizer from ${url}`,
  });
}

// GPT-2 style byte-to-unicode mapping
function createByteToUnicodeMap(): ReadonlyMap<number, string> {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const map = new Map<number, string>();
  for (let i = 0; i < bs.length; i++) {
    map.set(bs[i]!, String.fromCharCode(cs[i]!));
  }
  return map;
}

function createUnicodeToByteMap(byteToUnicode: ReadonlyMap<number, string>): ReadonlyMap<string, number> {
  const map = new Map<string, number>();
  for (const [byte, char] of byteToUnicode) {
    map.set(char, byte);
  }
  return map;
}

// GPT-2 ByteLevel pre-tokenizer regex
const GPT2_REGEX = new RegExp(
  "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
  "gu"
);

export class WhisperTokenizer implements TextTokenizer {
  readonly kind = 'tiktoken' as const;
  readonly vocabSize: number;
  private readonly idToToken: ReadonlyMap<number, string>;
  private readonly tokenToId: ReadonlyMap<string, number>;
  private readonly specialTokenIds: ReadonlySet<number>;
  private readonly timestampStartId: number;
  private readonly timestampEndId: number;
  private readonly byteToUnicode: ReadonlyMap<number, string>;
  private readonly unicodeToByte: ReadonlyMap<string, number>;
  private readonly bpeMerges: ReadonlyMap<string, number>;

  constructor(data: WhisperTokenizerJson) {
    const vocab = data.model?.vocab ?? {};
    const addedTokens = data.added_tokens ?? [];
    const merges = data.model?.merges ?? [];

    const idToToken = new Map<number, string>();
    const tokenToId = new Map<string, number>();
    const specialIds = new Set<number>();

    for (const [token, id] of Object.entries(vocab)) {
      idToToken.set(id, token);
      tokenToId.set(token, id);
    }

    for (const entry of addedTokens) {
      idToToken.set(entry.id, entry.content);
      tokenToId.set(entry.content, entry.id);
      if (entry.special) {
        specialIds.add(entry.id);
      }
    }

    this.idToToken = idToToken;
    this.tokenToId = tokenToId;
    this.specialTokenIds = specialIds;
    this.vocabSize = Math.max(...idToToken.keys(), 0) + 1;

    this.timestampStartId = tokenToId.get('<|0.00|>') ?? 50364;
    this.timestampEndId = tokenToId.get('<|30.00|>') ?? 51864;

    this.byteToUnicode = createByteToUnicodeMap();
    this.unicodeToByte = createUnicodeToByteMap(this.byteToUnicode);
    const bpeMerges = new Map<string, number>();
    for (let i = 0; i < merges.length; i++) {
      const merge = merges[i];
      if (merge !== undefined) bpeMerges.set(merge, i);
    }
    this.bpeMerges = bpeMerges;
  }

  static async fromUrl(
    url: string,
    signal?: AssetAbortSignalLike | null,
  ): Promise<WhisperTokenizer> {
    const text = await fetchText(url, signal);
    const data = JSON.parse(text) as WhisperTokenizerJson;
    return new WhisperTokenizer(data);
  }

  getTokenId(token: string): number | undefined {
    return this.tokenToId.get(token);
  }

  isSpecialTokenId(id: number): boolean {
    return this.specialTokenIds.has(id);
  }

  isTimestampTokenId(id: number): boolean {
    if (id >= this.timestampStartId && id <= this.timestampEndId) {
      return true;
    }
    const token = this.idToToken.get(id);
    return !!token && /^<\|\d+(?:\.\d+)?\|>$/.test(token);
  }

  timestampTokenIdToSeconds(id: number): number | undefined {
    if (!this.isTimestampTokenId(id)) {
      return undefined;
    }
    const token = this.idToToken.get(id);
    if (!token) return undefined;
    const match = token.match(/<\|(\d+(?:\.\d+)?)\|>/);
    if (!match) return undefined;
    return Number.parseFloat(match[1]!);
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    const pattern = /(<\|[^|]+\|>)|([^<]+)/g;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(text)) !== null) {
      const special = match[1];
      const plain = match[2];
      if (special) {
        const id = this.tokenToId.get(special);
        if (id !== undefined) {
          ids.push(id);
        }
      } else if (plain) {
        ids.push(...this.bpeEncode(plain));
      }
    }
    return ids;
  }

  private bpeEncode(text: string): number[] {
    const ids: number[] = [];
    const tokens = text.match(GPT2_REGEX) ?? [];

    for (const token of tokens) {
      // Convert to bytes via byte-to-unicode mapping
      const bytes = new TextEncoder().encode(token);
      let word = '';
      for (const b of bytes) {
        word += this.byteToUnicode.get(b) ?? '\uFFFD';
      }

      // BPE merge loop
      let wordTokens: string[] = Array.from(word);
      while (wordTokens.length > 1) {
        let minRank = Infinity;
        let minIndex = -1;
        for (let i = 0; i < wordTokens.length - 1; i++) {
          const left = wordTokens[i];
          const right = wordTokens[i + 1];
          if (left === undefined || right === undefined) continue;
          const pair = left + ' ' + right;
          const rank = this.bpeMerges.get(pair);
          if (rank !== undefined && rank < minRank) {
            minRank = rank;
            minIndex = i;
          }
        }
        if (minIndex === -1) break;
        wordTokens = [
          ...wordTokens.slice(0, minIndex),
          wordTokens[minIndex]! + wordTokens[minIndex + 1]!,
          ...wordTokens.slice(minIndex + 2),
        ];
      }

      for (const t of wordTokens) {
        const id = this.tokenToId.get(t);
        if (id !== undefined) ids.push(id);
      }
    }

    return ids;
  }

  decode(ids: readonly number[], options: { readonly skipSpecialTokens?: boolean } = {}): string {
    const parts: string[] = [];
    for (const id of ids) {
      if (options.skipSpecialTokens && this.isSpecialTokenId(id)) {
        continue;
      }
      const token = this.idToToken.get(id);
      if (token === undefined) {
        continue;
      }
      parts.push(token);
    }

    const bytes: number[] = [];
    for (const char of Array.from(parts.join(''))) {
      const byte = this.unicodeToByte.get(char);
      if (byte !== undefined) {
        bytes.push(byte);
      }
    }
    return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
  }

  idsToTokens(ids: readonly number[]): readonly string[] {
    return ids.map((id) => this.idToToken.get(id) ?? '');
  }
}
