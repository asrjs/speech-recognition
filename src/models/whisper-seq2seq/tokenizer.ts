import { isNodeLikeRuntime, importNodeModule } from '../../io/node.js';
import type { TextTokenizer } from '../../tokenizers/index.js';

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
    throw new Error(`Failed to fetch tokenizer from ${url}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

export class WhisperTokenizer implements TextTokenizer {
  readonly kind = 'tiktoken' as const;
  readonly vocabSize: number;
  private readonly idToToken: ReadonlyMap<number, string>;
  private readonly tokenToId: ReadonlyMap<string, number>;
  private readonly specialTokenIds: ReadonlySet<number>;
  private readonly timestampStartId: number;
  private readonly timestampEndId: number;

  constructor(data: WhisperTokenizerJson) {
    const vocab = data.model?.vocab ?? {};
    const addedTokens = data.added_tokens ?? [];

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
  }

  static async fromUrl(url: string): Promise<WhisperTokenizer> {
    const text = await fetchText(url);
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
    // For Whisper prompts, we mainly need special token lookup.
    // Full BPE encode is complex; this handles exact special token matches.
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
        // Naive fallback: character-level encoding for non-special text
        for (const char of plain) {
          const id = this.tokenToId.get(char);
          if (id !== undefined) {
            ids.push(id);
          } else {
            // Try with GPT-2 space prefix
            const spaceId = this.tokenToId.get('Ġ' + char);
            if (spaceId !== undefined) {
              ids.push(spaceId);
            }
          }
        }
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

    // GPT-2 style BPE decode cleanup
    let text = parts.join('');
    // Replace continuation marker with nothing ( Whisper uses no marker, but GPT-2 vocab does )
    text = text.replace(/Ġ/g, ' ');
    text = text.replace(/\s+/g, ' ');
    return text.trim();
  }

  idsToTokens(ids: readonly number[]): readonly string[] {
    return ids.map((id) => this.idToToken.get(id) ?? '');
  }
}
