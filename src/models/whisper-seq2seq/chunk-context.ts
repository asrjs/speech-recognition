/**
 * Condition-on-Previous-Text for Whisper chunking.
 *
 * When processing long audio in chunks, Whisper can use the transcription
 * of previous chunks as context to improve continuity. This module provides
 * the prompt builder that injects previous tokens into the next chunk's prompt.
 *
 * Algorithm (matches faster-whisper):
 *   1. After each chunk, collect the generated tokens (excluding prompt)
 *   2. When building the next chunk's prompt, insert previous tokens after
 *      a <|0.00|> timestamp token (50364)
 *   3. Crop previous tokens to maxContextTokens (default: 224, half of max_target_positions)
 *   4. Reset context on transcription reset or fallback
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Token for <|0.00|> — marks start of context in conditioned prompts. */
export const TIMESTAMP_0_TOKEN = 50364;

// ---------------------------------------------------------------------------
// ChunkContextBuilder
// ---------------------------------------------------------------------------

export class ChunkContextBuilder {
  private allTokens: number[] = [];
  private maxContextTokens: number;

  constructor(maxContextTokens: number) {
    this.maxContextTokens = maxContextTokens;
  }

  /** Get previous segment tokens, cropped to maxContextTokens from the end. */
  getPreviousTokens(): readonly number[] {
    return this.allTokens.slice(-this.maxContextTokens);
  }

  /** Add a segment's generated tokens to the context. */
  addSegmentTokens(tokens: readonly number[]): void {
    if (tokens.length === 0) return;
    this.allTokens.push(...tokens);
    // Keep only the tail within the limit
    if (this.allTokens.length > this.maxContextTokens) {
      this.allTokens = this.allTokens.slice(-this.maxContextTokens);
    }
  }

  /** Reset all accumulated context. */
  reset(): void {
    this.allTokens = [];
  }

  /** Get total number of tokens accumulated (before cropping). */
  getTotalTokenCount(): number {
    return this.allTokens.length;
  }
}

// ---------------------------------------------------------------------------
// Prompt Builder
// ---------------------------------------------------------------------------

/**
 * Build a prompt that includes previous segment context.
 *
 * Structure:
 *   [base_prompt..., <|0.00|>, ...previous_tokens_tail]
 *
 * This mirrors the prompt format used in faster-whisper's condition_on_previous_text.
 *
 * @param basePrompt — the standard prompt [SOT, lang, task, notimestamps]
 * @param previousTokens — tokens from previous segments
 * @param maxContextTokens — max tokens to include (crops from end)
 */
export function buildPromptWithContext(
  basePrompt: readonly number[],
  previousTokens: readonly number[],
  maxContextTokens: number,
): number[] {
  if (previousTokens.length === 0 || maxContextTokens <= 0) {
    return [...basePrompt];
  }

  // Crop to the last maxContextTokens
  const contextTokens = previousTokens.slice(-maxContextTokens);

  return [...basePrompt, TIMESTAMP_0_TOKEN, ...contextTokens];
}
