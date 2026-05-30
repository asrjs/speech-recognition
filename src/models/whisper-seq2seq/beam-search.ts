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

function logSoftmax(logits: Float32Array): Float32Array {
  let maxLogit = Number.NEGATIVE_INFINITY;
  for (const logit of logits) {
    if (logit > maxLogit) maxLogit = logit;
  }

  let expSum = 0;
  for (const logit of logits) {
    expSum += Math.exp(logit - maxLogit);
  }

  const logSumExp = maxLogit + Math.log(expSum);
  const logProbs = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    logProbs[i] = (logits[i] ?? Number.NEGATIVE_INFINITY) - logSumExp;
  }
  return logProbs;
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
  const candidates: WhisperBeamState<TToken>[] = [];

  for (let beamIndex = 0; beamIndex < beams.length; beamIndex++) {
    const beam = beams[beamIndex];
    if (!beam) continue;

    if (beam.completed) {
      candidates.push(beam);
      continue;
    }

    const logits = logitsByBeam[beamIndex];
    if (!logits) continue;
    const logProbs = logSoftmax(logits);

    for (let tokenId = 0; tokenId < logProbs.length; tokenId++) {
      const logProb = logProbs[tokenId] ?? Number.NEGATIVE_INFINITY;
      candidates.push({
        tokens: [...beam.tokens, tokenId],
        score: beam.score + logProb,
        completed: tokenId === eosTokenId,
        payload: expandPayload?.(beam, tokenId, logProb),
      });
    }
  }

  return candidates
    .sort((a, b) => normalizedBeamScore(b, lengthPenalty) - normalizedBeamScore(a, lengthPenalty))
    .slice(0, Math.max(1, beamWidth));
}

export function selectBestWhisperBeam<TToken = unknown>(
  beams: readonly WhisperBeamState<TToken>[],
  lengthPenalty = 0,
): WhisperBeamState<TToken> | undefined {
  return [...beams].sort(
    (a, b) => normalizedBeamScore(b, lengthPenalty) - normalizedBeamScore(a, lengthPenalty),
  )[0];
}
