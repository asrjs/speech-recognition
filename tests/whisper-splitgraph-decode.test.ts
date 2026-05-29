import { describe, expect, it } from 'vitest';
import { splitGraphDecodeLoop } from '../src/models/whisper-seq2seq/executor.js';
import type { WhisperModelConfig } from '../src/models/whisper-seq2seq/generation-config.js';

// This is a pure unit test — no ONNX runtime needed.
// Tests the autoregressive decode loop logic for the 4-graph path.

function mockRunInit(
  promptTokens: number[],
  _encoderHiddenStates: Float32Array,
): { logits: Float32Array; presentKv: Record<string, Float32Array>; vocabSize: number } {
  const promptLen = promptTokens.length;
  const vocabSize = 10;
  const logits = new Float32Array(promptLen * vocabSize);
  // Set last position to token ID 7 (non-EOS) so we continue
  const lastPos = (promptLen - 1) * vocabSize;
  logits[lastPos + 7] = 10.0; // argmax = 7

  const initKv: Record<string, Float32Array> = {
    'present.0.decoder.key': new Float32Array([1, 2, 3]),
    'present.0.decoder.value': new Float32Array([4, 5]),
    'present.0.encoder.key': new Float32Array([6, 7, 8, 9]),
    'present.0.encoder.value': new Float32Array([10, 11, 12, 13]),
  };

  return { logits, presentKv: initKv, vocabSize };
}

function mockRunStep(
  _tokenId: number,
  _pastKv: Record<string, Float32Array>,
): { logits: Float32Array; presentKv: Record<string, Float32Array> } {
  const vocabSize = 10;
  const logits = new Float32Array(vocabSize);
  // Use closure counter (mutating pastKv object doesn't persist across new presentKv returns)
  mockRunStep.callCount += 1;

  if (mockRunStep.callCount >= 3) {
    // 3rd step → emit EOS (token 5)
    logits[5] = 10.0;
  } else {
    logits[7] = 10.0;
  }

  const presentKv: Record<string, Float32Array> = {
    'present.0.decoder.key': new Float32Array([99]),
    'present.0.decoder.value': new Float32Array([100]),
  };

  return { logits, presentKv, vocabSize };
}
mockRunStep.callCount = 0;

const tinyConfig: WhisperModelConfig = {
  decoderLayers: 4,
  decoderAttentionHeads: 6,
  dModel: 384,
  headDim: 64,
  medianFilterWidth: 7,
};

describe('splitGraphDecodeLoop (4-graph init→step autoregressive)', () => {
  it('runs init→step loop and stops on EOS', async () => {
    const promptTokens = [50258, 50259, 50359, 50363]; // SOT, en, transcribe, notimestamps
    const encoderHiddenStates = new Float32Array(100);
    const eosTokenId = 5;

    const result = await splitGraphDecodeLoop({
      promptTokens,
      encoderHiddenStates,
      eosTokenId,
      maxNewTokens: 20,
      modelConfig: tinyConfig,
      runInit: mockRunInit,
      runStep: mockRunStep,
    });

    // Should produce tokens: init gave 7 (from last pos), step gave [7, 7, 5]
    // But init doesn't add token — we extract from logits
    // Actually: first token from init logits = 7
    // Step 1: token 7
    // Step 2: token 7  
    // Step 3: EOS 5 → stop
    expect(result.tokens.length).toBe(4); // [7, 7, 7, 5]
    expect(result.tokens).toEqual([7, 7, 7, 5]);
    // last token is EOS
    expect(result.tokens[result.tokens.length - 1]).toBe(eosTokenId);
  });

  it('stops after maxNewTokens even without EOS', async () => {
    const maxNewTokens = 2;
    const result = await splitGraphDecodeLoop({
      promptTokens: [50258, 50259, 50359],
      encoderHiddenStates: new Float32Array(100),
      eosTokenId: 5,
      maxNewTokens,
      modelConfig: tinyConfig,
      runInit: mockRunInit,
      runStep: () => ({
        // Never emit EOS
        logits: (() => {
          const l = new Float32Array(10);
          l[7] = 10.0;
          return l;
        })(),
        presentKv: {},
      }),
    });

    expect(result.tokens.length).toBeLessThanOrEqual(maxNewTokens);
  });
});
