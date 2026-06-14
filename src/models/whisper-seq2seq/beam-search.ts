export interface WhisperBeamState<TToken = unknown> {
  readonly tokens: readonly number[];
  readonly score: number;
  readonly completed: boolean;
  readonly payload?: TToken;
}

export interface WhisperBeamCandidateOptions<TToken = unknown> {
  readonly beams: readonly WhisperBeamState<TToken>[];
  readonly logitsByBeam: readonly Float32Array[];
  readonly beamWidth: number;
  readonly eosTokenId: number;
  readonly lengthPenalty?: number;
  readonly expandPayload?: (beam: WhisperBeamState<TToken>, tokenId: number, logProb: number) => TToken | undefined;
}

export function createInitialWhisperBeam<TToken = unknown>(
  tokens: readonly number[],
  score = 0,
  payload?: TToken,
): WhisperBeamState<TToken> {
  return { tokens: [...tokens], score, completed: false, payload };
}

export function normalizedBeamScore(beam: WhisperBeamState, lengthPenalty: number): number {
  if (lengthPenalty === 0) return beam.score;
  const generatedLength = Math.max(1, beam.tokens.length);
  return beam.score / Math.pow(generatedLength, lengthPenalty);
}

/** Normalize a raw cumulative log-prob score by token count. */
export function normalizedSequenceScore(
  cumulativeLogProb: number,
  tokenCount: number,
  lengthPenalty: number,
): number {
  if (lengthPenalty === 0) return cumulativeLogProb;
  return cumulativeLogProb / Math.pow(Math.max(1, tokenCount), lengthPenalty);
}

export function rankWhisperBeamCandidates<TToken = unknown>({
  beams,
  logitsByBeam,
  beamWidth,
  eosTokenId,
  lengthPenalty = 0,
  expandPayload,
}: WhisperBeamCandidateOptions<TToken>): WhisperBeamState<TToken>[] {
  const candidateLimit = Math.max(1, beamWidth);
  const candidates: RankedWhisperCandidate<TToken>[] = [];

  for (let beamIndex = 0; beamIndex < beams.length; beamIndex++) {
    const beam = beams[beamIndex];
    if (!beam) continue;

    if (beam.completed) {
      insertRankedCandidate(candidates, {
        beam,
        normalizedScore: normalizedBeamScore(beam, lengthPenalty),
      }, candidateLimit);
      continue;
    }

    const logits = logitsByBeam[beamIndex];
    if (!logits) continue;
    const normalizers = getLogSoftmaxNormalizers(logits);
    if (!normalizers) continue;

    const candidateLength = beam.tokens.length + 1;
    for (let tokenId = 0; tokenId < logits.length; tokenId++) {
      const logProb = (logits[tokenId] ?? Number.NEGATIVE_INFINITY) - normalizers.logSumExp;
      const score = beam.score + logProb;
      insertRankedCandidate(candidates, {
        beam,
        tokenId,
        logProb,
        score,
        completed: tokenId === eosTokenId,
        normalizedScore: normalizeScore(score, candidateLength, lengthPenalty),
      }, candidateLimit);
    }
  }

  return candidates.map((candidate) => {
    if (candidate.tokenId === undefined) return candidate.beam;
    return {
      tokens: [...candidate.beam.tokens, candidate.tokenId],
      score: candidate.score ?? candidate.beam.score,
      completed: candidate.completed ?? false,
      payload: expandPayload?.(candidate.beam, candidate.tokenId, candidate.logProb ?? Number.NEGATIVE_INFINITY),
    };
  });
}

interface RankedWhisperCandidate<TToken = unknown> {
  readonly beam: WhisperBeamState<TToken>;
  readonly tokenId?: number;
  readonly logProb?: number;
  readonly score?: number;
  readonly completed?: boolean;
  readonly normalizedScore: number;
}

function getLogSoftmaxNormalizers(logits: Float32Array): { readonly logSumExp: number } | undefined {
  if (logits.length === 0) return undefined;

  let maxLogit = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < logits.length; i++) {
    const logit = logits[i] ?? Number.NEGATIVE_INFINITY;
    if (logit > maxLogit) maxLogit = logit;
  }

  let expSum = 0;
  for (let i = 0; i < logits.length; i++) {
    expSum += Math.exp((logits[i] ?? Number.NEGATIVE_INFINITY) - maxLogit);
  }

  return { logSumExp: maxLogit + Math.log(expSum) };
}

function normalizeScore(score: number, tokenCount: number, lengthPenalty: number): number {
  return lengthPenalty === 0 ? score : score / Math.pow(Math.max(1, tokenCount), lengthPenalty);
}

function insertRankedCandidate<TToken>(
  candidates: RankedWhisperCandidate<TToken>[],
  candidate: RankedWhisperCandidate<TToken>,
  limit: number,
): void {
  let insertAt = candidates.length;
  while (
    insertAt > 0 &&
    candidate.normalizedScore > (candidates[insertAt - 1]?.normalizedScore ?? Number.NEGATIVE_INFINITY)
  ) {
    insertAt--;
  }

  if (insertAt >= limit) return;
  candidates.splice(insertAt, 0, candidate);
  if (candidates.length > limit) {
    candidates.pop();
  }
}

export function selectBestWhisperBeam<TToken = unknown>(
  beams: readonly WhisperBeamState<TToken>[],
  lengthPenalty = 0,
): WhisperBeamState<TToken> | undefined {
  return [...beams].sort(
    (a, b) => normalizedBeamScore(b, lengthPenalty) - normalizedBeamScore(a, lengthPenalty),
  )[0];
}
