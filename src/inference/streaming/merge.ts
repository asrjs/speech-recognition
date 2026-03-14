import type {
  TranscriptMeta,
  TranscriptResult,
  TranscriptSegment,
  TranscriptToken,
  TranscriptWarning,
  TranscriptWord
} from '../../types/index.js';

function joinTranscriptText(parts: readonly string[]): string {
  return parts
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function reindexWords(words: readonly TranscriptWord[], offset: number): TranscriptWord[] {
  return words.map((word, index) => ({
    ...word,
    index: offset + index
  }));
}

function reindexSegments(segments: readonly TranscriptSegment[], offset: number): TranscriptSegment[] {
  return segments.map((segment, index) => ({
    ...segment,
    index: offset + index
  }));
}

function reindexTokens(tokens: readonly TranscriptToken[], offset: number): TranscriptToken[] {
  return tokens.map((token, index) => ({
    ...token,
    index: offset + index
  }));
}

function mergeWarnings(results: readonly TranscriptResult[]): TranscriptWarning[] {
  return results.flatMap((result) => result.warnings);
}

function mergeMeta(results: readonly TranscriptResult[]): TranscriptMeta {
  const last = results.at(-1);
  const tokenCount = results.reduce((sum, result) => sum + (result.tokens?.length ?? 0), 0);
  const wordCount = results.reduce((sum, result) => sum + (result.words?.length ?? 0), 0);
  const segmentCount = results.reduce((sum, result) => sum + (result.segments?.length ?? 0), 0);

  return {
    detailLevel: last?.meta.detailLevel ?? 'text',
    isFinal: results.every((result) => result.meta.isFinal),
    modelFamily: last?.meta.modelFamily,
    modelId: last?.meta.modelId,
    backendId: last?.meta.backendId,
    language: last?.meta.language,
    sampleRate: last?.meta.sampleRate,
    durationSeconds: results.reduce((sum, result) => sum + (result.meta.durationSeconds ?? 0), 0),
    tokenCount,
    wordCount,
    segmentCount,
    averageConfidence: last?.meta.averageConfidence,
    averageSegmentConfidence: last?.meta.averageSegmentConfidence,
    averageWordConfidence: last?.meta.averageWordConfidence,
    averageTokenConfidence: last?.meta.averageTokenConfidence,
    nativeAvailable: last?.meta.nativeAvailable,
    producedAt: last?.meta.producedAt,
    backendNotes: last?.meta.backendNotes,
    metrics: last?.meta.metrics
  };
}

export function mergeTranscriptResults(results: readonly TranscriptResult[]): TranscriptResult {
  if (results.length === 0) {
    return {
      text: '',
      warnings: [],
      meta: {
        detailLevel: 'text',
        isFinal: true,
        tokenCount: 0,
        wordCount: 0,
        segmentCount: 0
      }
    };
  }

  const segments = results.flatMap((result, index) => reindexSegments(
    result.segments ?? [],
    results.slice(0, index).reduce((sum, previous) => sum + (previous.segments?.length ?? 0), 0)
  ));
  const words = results.flatMap((result, index) => reindexWords(
    result.words ?? [],
    results.slice(0, index).reduce((sum, previous) => sum + (previous.words?.length ?? 0), 0)
  ));
  const tokens = results.flatMap((result, index) => reindexTokens(
    result.tokens ?? [],
    results.slice(0, index).reduce((sum, previous) => sum + (previous.tokens?.length ?? 0), 0)
  ));

  return {
    text: joinTranscriptText(results.map((result) => result.text)),
    warnings: mergeWarnings(results),
    meta: mergeMeta(results),
    segments: segments.length > 0 ? segments : undefined,
    words: words.length > 0 ? words : undefined,
    tokens: tokens.length > 0 ? tokens : undefined
  };
}

export function joinTranscriptFragments(left: string, right: string): string {
  return joinTranscriptText([left, right]);
}

export interface FrameAlignedToken {
  readonly id: number;
  readonly frameIndex: number;
  readonly absTime: number;
  readonly logProb: number;
  readonly text: string;
  readonly vignetteWeight?: number;
}

export interface TokenAlignmentChunk {
  readonly tokenIds: readonly number[];
  readonly frameIndices: readonly number[];
  readonly logProbs?: readonly number[];
  readonly tokens?: readonly { readonly text?: string; readonly token?: string }[];
}

export interface FrameAlignedTokenMergerOptions {
  readonly frameTimeStride?: number;
  readonly timeTolerance?: number;
  readonly stabilityThreshold?: number;
}

export interface LcsPtfaTokenMergerOptions {
  readonly frameTimeStride?: number;
  readonly timeTolerance?: number;
  readonly sequenceAnchorLength?: number;
  readonly vignetteSigmaFactor?: number;
}

function buildAlignedTokens(
  result: TokenAlignmentChunk,
  chunkStartTime: number,
  frameTimeStride: number,
  totalTokens = result.tokenIds.length,
  withVignette = false,
  vignetteSigmaFactor = 0.25
): FrameAlignedToken[] {
  return result.tokenIds.map((id, index) => {
    const frameIndex = result.frameIndices[index] ?? 0;
    const absTime = chunkStartTime + (frameIndex * frameTimeStride);
    const text = result.tokens?.[index]?.text ?? result.tokens?.[index]?.token ?? '';
    if (!withVignette) {
      return {
        id,
        frameIndex,
        absTime,
        logProb: result.logProbs?.[index] ?? 0,
        text
      };
    }

    const midpoint = (Math.max(totalTokens, 1) - 1) / 2;
    const sigma = Math.max(1, totalTokens) * vignetteSigmaFactor;
    const vignetteWeight = Math.exp(-Math.pow(index - midpoint, 2) / (2 * sigma * sigma));
    return {
      id,
      frameIndex,
      absTime,
      logProb: result.logProbs?.[index] ?? 0,
      text,
      vignetteWeight
    };
  });
}

export class FrameAlignedTokenMerger {
  private readonly frameTimeStride: number;
  private readonly timeTolerance: number;
  private readonly stabilityThreshold: number;
  private readonly confirmedTokens: FrameAlignedToken[] = [];
  private pendingTokens: FrameAlignedToken[] = [];
  private readonly stabilityMap = new Map<string, number>();

  constructor(options: FrameAlignedTokenMergerOptions = {}) {
    this.frameTimeStride = options.frameTimeStride ?? 0.08;
    this.timeTolerance = options.timeTolerance ?? 0.2;
    this.stabilityThreshold = options.stabilityThreshold ?? 2;
  }

  processChunk(result: TokenAlignmentChunk, chunkStartTime: number, overlapDuration = 0) {
    const tokens = buildAlignedTokens(result, chunkStartTime, this.frameTimeStride);
    const overlapEndTime = chunkStartTime + overlapDuration;
    const overlapTokens = tokens.filter((token) => token.absTime < overlapEndTime);
    const newTokens = tokens.filter((token) => token.absTime >= overlapEndTime);
    const anchors = this.findAnchors(overlapTokens);

    if (anchors.length > 0) {
      const anchorTime = anchors[0]!.absTime;
      const toConfirm = this.pendingTokens.filter((token) => token.absTime < anchorTime);
      this.confirmedTokens.push(...toConfirm);

      for (const token of overlapTokens) {
        const key = this.tokenKey(token.id, token.absTime);
        const count = (this.stabilityMap.get(key) ?? 0) + 1;
        this.stabilityMap.set(key, count);

        if (count >= this.stabilityThreshold) {
          const alreadyConfirmed = this.confirmedTokens.some((confirmed) => (
            Math.abs(confirmed.absTime - token.absTime) < this.timeTolerance
            && confirmed.id === token.id
          ));
          if (!alreadyConfirmed) {
            this.confirmedTokens.push(token);
          }
        }
      }
    }

    this.pendingTokens = newTokens;

    return {
      confirmed: this.confirmedTokens.slice(),
      pending: this.pendingTokens.slice(),
      anchorsFound: anchors.length,
      totalTokens: this.confirmedTokens.length + this.pendingTokens.length
    };
  }

  reset(): void {
    this.confirmedTokens.length = 0;
    this.pendingTokens = [];
    this.stabilityMap.clear();
  }

  getAllTokens(): FrameAlignedToken[] {
    return [...this.confirmedTokens, ...this.pendingTokens].sort((left, right) => left.absTime - right.absTime);
  }

  getState(): { confirmedCount: number; pendingCount: number; stabilityMapSize: number } {
    return {
      confirmedCount: this.confirmedTokens.length,
      pendingCount: this.pendingTokens.length,
      stabilityMapSize: this.stabilityMap.size
    };
  }

  private tokenKey(tokenId: number, absTime: number): string {
    return `${tokenId}@${Math.round(absTime * 10)}`;
  }

  private findAnchors(overlapTokens: readonly FrameAlignedToken[]): FrameAlignedToken[] {
    const anchors: FrameAlignedToken[] = [];
    for (const newToken of overlapTokens) {
      for (const pendingToken of this.pendingTokens) {
        if (
          newToken.id === pendingToken.id
          && Math.abs(newToken.absTime - pendingToken.absTime) < this.timeTolerance
        ) {
          anchors.push(newToken);
          break;
        }
      }
    }

    return anchors.sort((left, right) => left.absTime - right.absTime);
  }
}

export class LcsPtfaTokenMerger {
  private readonly frameTimeStride: number;
  private readonly timeTolerance: number;
  private readonly sequenceAnchorLength: number;
  private readonly vignetteSigmaFactor: number;
  private readonly confirmedTokens: FrameAlignedToken[] = [];
  private pendingTokens: FrameAlignedToken[] = [];
  private lcsBuffer = new Int32Array(1024);

  constructor(options: LcsPtfaTokenMergerOptions = {}) {
    this.frameTimeStride = options.frameTimeStride ?? 0.08;
    this.timeTolerance = options.timeTolerance ?? 0.15;
    this.sequenceAnchorLength = options.sequenceAnchorLength ?? 3;
    this.vignetteSigmaFactor = options.vignetteSigmaFactor ?? 0.25;
  }

  processChunk(result: TokenAlignmentChunk, chunkStartTime: number, overlapDuration = 0) {
    const tokens = buildAlignedTokens(
      result,
      chunkStartTime,
      this.frameTimeStride,
      result.tokenIds.length,
      true,
      this.vignetteSigmaFactor
    );
    const overlapEnd = chunkStartTime + overlapDuration;
    const overlapTokens = tokens.filter((token) => token.absTime < overlapEnd);
    const newTokens = tokens.filter((token) => token.absTime >= overlapEnd);

    if (this.pendingTokens.length === 0) {
      this.pendingTokens = tokens;
      return {
        confirmed: this.confirmedTokens.slice(),
        pending: this.pendingTokens.slice(),
        lcsLength: 0,
        anchorValid: false,
        isFirstChunk: true
      };
    }

    const previousIds = this.pendingTokens.map((token) => token.id);
    const overlapIds = overlapTokens.map((token) => token.id);
    const [startX, startY, lcsLength] = this.findLongestCommonSubstring(previousIds, overlapIds);

    let anchorValid = false;
    if (lcsLength >= this.sequenceAnchorLength) {
      anchorValid = this.verifyFrameAlignment(
        this.pendingTokens.slice(startX, startX + lcsLength),
        overlapTokens.slice(startY, startY + lcsLength)
      );
    }

    if (anchorValid) {
      this.confirmedTokens.push(...this.pendingTokens.slice(0, startX + lcsLength));
    } else if (lcsLength > 0) {
      const pathA = this.pendingTokens.slice(startX, startX + lcsLength);
      const pathB = overlapTokens.slice(startY, startY + lcsLength);
      const bestPath = this.arbitrateByLogProb(pathA, pathB);
      this.confirmedTokens.push(...this.pendingTokens.slice(0, startX));
      this.confirmedTokens.push(...bestPath);
    } else {
      this.confirmedTokens.push(...this.pendingTokens);
    }

    this.pendingTokens = newTokens;

    return {
      confirmed: this.confirmedTokens.slice(),
      pending: this.pendingTokens.slice(),
      lcsLength,
      anchorValid,
      overlapTokenCount: overlapTokens.length
    };
  }

  reset(): void {
    this.confirmedTokens.length = 0;
    this.pendingTokens = [];
  }

  getAllTokens(): FrameAlignedToken[] {
    return [...this.confirmedTokens, ...this.pendingTokens].sort((left, right) => left.absTime - right.absTime);
  }

  getState(): { confirmedCount: number; pendingCount: number } {
    return {
      confirmedCount: this.confirmedTokens.length,
      pendingCount: this.pendingTokens.length
    };
  }

  private findLongestCommonSubstring(left: readonly number[], right: readonly number[]): [number, number, number] {
    if (left.length === 0 || right.length === 0) {
      return [0, 0, 0];
    }

    if (this.lcsBuffer.length < right.length + 1) {
      this.lcsBuffer = new Int32Array(right.length + 1025);
    }

    this.lcsBuffer.fill(0, 0, right.length + 1);
    let maxLength = 0;
    let endLeft = 0;
    let endRight = 0;

    for (let leftIndex = 1; leftIndex <= left.length; leftIndex += 1) {
      let previous = 0;
      for (let rightIndex = 1; rightIndex <= right.length; rightIndex += 1) {
        const temp = this.lcsBuffer[rightIndex] ?? 0;
        if (left[leftIndex - 1] === right[rightIndex - 1]) {
          this.lcsBuffer[rightIndex] = previous + 1;
          if ((this.lcsBuffer[rightIndex] ?? 0) > maxLength) {
            maxLength = this.lcsBuffer[rightIndex] ?? 0;
            endLeft = leftIndex;
            endRight = rightIndex;
          }
        } else {
          this.lcsBuffer[rightIndex] = 0;
        }
        previous = temp;
      }
    }

    return [endLeft - maxLength, endRight - maxLength, maxLength];
  }

  private verifyFrameAlignment(left: readonly FrameAlignedToken[], right: readonly FrameAlignedToken[]): boolean {
    if (left.length !== right.length) {
      return false;
    }

    return left.every((token, index) => Math.abs(token.absTime - (right[index]?.absTime ?? 0)) <= this.timeTolerance);
  }

  private arbitrateByLogProb(pathA: readonly FrameAlignedToken[], pathB: readonly FrameAlignedToken[]): FrameAlignedToken[] {
    const hasLogProbs = pathA.some((token) => token.logProb !== 0) || pathB.some((token) => token.logProb !== 0);
    if (!hasLogProbs) {
      return [...pathA];
    }

    const scoreA = pathA.reduce((sum, token) => sum + (token.logProb * (token.vignetteWeight ?? 1)), 0);
    const scoreB = pathB.reduce((sum, token) => sum + (token.logProb * (token.vignetteWeight ?? 1)), 0);
    return [...(scoreA >= scoreB ? pathA : pathB)];
  }
}
