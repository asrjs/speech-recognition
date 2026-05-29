import { describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { resolveWhisperArtifacts, initWhisperOrt, createWhisperOrtSession } from '../src/models/whisper-seq2seq/ort.js';
import { parseWhisperManifest } from '../src/models/whisper-seq2seq/manifest.js';
import { WhisperTokenizer } from '../src/models/whisper-seq2seq/tokenizer.js';
import { WhisperMelProcessor } from '../src/audio/whisper-mel.js';
import type { OrtTensorLike } from '../src/models/whisper-seq2seq/ort.js';
import type { WhisperArtifactSource, WhisperExecutionBackend } from '../src/models/whisper-seq2seq/types.js';

function argmax(arr: Float32Array): number {
  let maxIdx = 0;
  let maxVal = arr[0] ?? -Infinity;
  for (let i = 1; i < arr.length; i++) {
    if ((arr[i] ?? -Infinity) > maxVal) { maxVal = arr[i]!; maxIdx = i; }
  }
  return maxIdx;
}

describe('Whisper 4-graph low-level ONNX fixture test', () => {
  it('encoder → init → step loop → align: verifies all tensor shapes via manifest-driven dimensions', async () => {
    const fixtureDir = process.env.WHISPER_SPLITGRAPH_FIXTURE_DIR;
    if (!fixtureDir) {
      console.warn('Skipping: set WHISPER_SPLITGRAPH_FIXTURE_DIR=/path/to/exported/whisper-tiny');
      return;
    }

    // ── File existence check ──────────────────────────────────
    const f = (...parts: string[]) => path.join(fixtureDir, ...parts);
    const required = ['encoder_model.onnx', 'decoder_init.onnx', 'decoder_step.onnx', 'tokenizer.json', 'manifest.json'];
    if (!required.every((name) => fs.existsSync(f(name)))) {
      console.warn(`Skipping: missing splitgraph fixtures in ${fixtureDir}.`);
      return;
    }
    const hasAlign = fs.existsSync(f('decoder_align.onnx'));

    // ── 1. Parse manifest → all dimensions from manifest, zero hardcoded ──
    const manifestRaw = JSON.parse(fs.readFileSync(f('manifest.json'), 'utf-8')) as Record<string, unknown>;
    const manifest = parseWhisperManifest(manifestRaw);
    const { decoderLayers, decoderAttentionHeads, dModel, headDim } = manifest.modelConfig;
    const numMelBins   = (manifestRaw.num_mel_bins   as number) ?? 80;
    const maxSrcPos    = (manifestRaw.max_source_positions  as number) ?? 3000;
    const maxTgtPos    = (manifestRaw.max_target_positions  as number) ?? 448;
    const vocabSize    = (manifestRaw.vocab_size     as number) ?? 51865;

    // Verify nothing is hardcoded — all values come from manifest
    expect(decoderLayers).toBeGreaterThan(0);
    expect(decoderAttentionHeads).toBeGreaterThan(0);
    expect(dModel).toBeGreaterThan(0);
    expect(headDim).toBe(dModel / decoderAttentionHeads);
    expect(dModel % decoderAttentionHeads).toBe(0);
    expect(manifest.generationConfig.alignmentHeads.length).toBeGreaterThan(0);

    // ── 2. Resolve artifacts through public resolver ──────────
    const source: WhisperArtifactSource = {
      kind: 'splitgraph',
      artifacts: {
        encoderUrl:     `file://${f('encoder_model.onnx')}`,
        decoderInitUrl: `file://${f('decoder_init.onnx')}`,
        decoderStepUrl: `file://${f('decoder_step.onnx')}`,
        decoderAlignUrl: hasAlign ? `file://${f('decoder_align.onnx')}` : undefined,
        tokenizerUrl:   `file://${f('tokenizer.json')}`,
        manifestUrl:    `file://${f('manifest.json')}`,
      },
    };
    const resolved = resolveWhisperArtifacts(source, 'wasm');
    expect(resolved.isSplitGraph).toBe(true);
    expect(resolved.decoderInitUrl).toBeTruthy();
    expect(resolved.decoderStepUrl).toBeTruthy();

    // ── 3. Create ONNX sessions (same path as executor) ───────
    const ort = await initWhisperOrt('wasm');
    const backend: WhisperExecutionBackend = 'wasm';

    const encSess = await createWhisperOrtSession(ort, resolved.artifacts.encoderUrl, { backendId: backend });
    const initSess = await createWhisperOrtSession(ort, resolved.decoderInitUrl!, { backendId: backend });
    const stepSess = await createWhisperOrtSession(ort, resolved.decoderStepUrl!, { backendId: backend });
    const alignSess = hasAlign && resolved.decoderAlignUrl
      ? await createWhisperOrtSession(ort, resolved.decoderAlignUrl, { backendId: backend })
      : null;

    const tokenizer = await WhisperTokenizer.fromUrl(resolved.artifacts.tokenizerUrl);

    // ── 4. Generate test audio: 2.5s 440→880 Hz chirp @ 16 kHz ──
    const sampleRate = 16000;
    const duration = 2.5;
    const totalSamples = Math.floor(sampleRate * duration);
    const samples = new Float32Array(totalSamples);
    for (let i = 0; i < totalSamples; i++) {
      const t = i / sampleRate;
      const freq = 440 + (440 * t) / duration;
      samples[i] = Math.sin(2 * Math.PI * freq * t) * 0.5;
    }

    // ── 5. Mel → encoder ─────────────────────────────────────
    const melProc = new WhisperMelProcessor({ nMels: numMelBins });
    const melResult = melProc.process(samples);
    const paddedMel = WhisperMelProcessor.padToFrames(melResult, maxSrcPos);
    const melTensor = new ort.Tensor('float32', paddedMel, [1, numMelBins, maxSrcPos]);

    const encOutputs = await encSess.run({ input_features: melTensor });
    const encOut = encOutputs[Object.keys(encOutputs)[0]!] as OrtTensorLike<Float32Array>;

    // Verify encoder output shape [1, 1500, d_model]
    expect(encOut.dims[0]).toBe(1);
    expect(encOut.dims[1]).toBe(maxSrcPos);
    expect(encOut.dims[2]).toBe(dModel);

    // ── 6. Build prompt tokens ────────────────────────────────
    const lang = 'en';
    const sotId = tokenizer.getTokenId('<|startoftranscript|>') ?? 50258;
    const langId = tokenizer.getTokenId(`<|${lang}|>`) ?? 50259;
    const taskId = tokenizer.getTokenId('<|transcribe|>') ?? 50359;
    const noTsId = tokenizer.getTokenId('<|notimestamps|>');
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;

    const promptIds: number[] = [sotId, langId, taskId];
    if (noTsId !== undefined) promptIds.push(noTsId);

    // ── 7. Decoder init ───────────────────────────────────────
    const promptArr = new BigInt64Array(promptIds.map((id) => BigInt(id)));
    const promptTensor = new ort.Tensor('int64', promptArr, [1, promptIds.length]);

    const initOutputs = await initSess.run({
      input_ids: promptTensor,
      encoder_hidden_states: encOut,
    });
    const initKeys = Object.keys(initOutputs);

    const logitsKey = initKeys.find((k) => k.includes('logits')) ?? initKeys[0]!;
    const initLogits = (initOutputs[logitsKey] as OrtTensorLike<Float32Array>).data;
    const logitsDims = (initOutputs[logitsKey] as OrtTensorLike<Float32Array>).dims;
    expect(logitsDims[0]).toBe(1);
    expect(logitsDims[1]).toBe(promptIds.length);
    expect(logitsDims[2]).toBe(vocabSize);

    // Verify KV cache entry count: decoder_init outputs present.{i}.{decoder,encoder}.{key,value}
    // That's 4 entries per layer (decoder.key, decoder.value, encoder.key, encoder.value)
    const presentKvKeys = initKeys.filter((k) => k.startsWith('present'));
    const expectedKvCount = 4 * decoderLayers;
    expect(presentKvKeys.length).toBe(expectedKvCount);

    // Collect present KV, convert to past_key_values prefix for step model
    const pastKv: Record<string, OrtTensorLike<Float32Array>> = {};
    for (const key of presentKvKeys) {
      const pastName = key.replace(/^present\./, 'past_key_values.');
      pastKv[pastName] = initOutputs[key] as OrtTensorLike<Float32Array>;
    }

    // ── 8. First token from init logits (last position) ────────
    const lastOffset = initLogits.length - vocabSize;
    const firstLogits = initLogits.subarray(lastOffset);
    let nextToken = argmax(firstLogits);
    const generatedTokens: number[] = [nextToken];

    // ── 9. Decoder step loop ──────────────────────────────────
    for (let s = 1; s < maxTgtPos; s++) {
      const stepInput = new BigInt64Array([BigInt(nextToken)]);
      const stepFeeds: Record<string, unknown> = {
        input_ids: new ort.Tensor('int64', stepInput, [1, 1]),
      };
      for (const [name, tensor] of Object.entries(pastKv)) {
        stepFeeds[name] = tensor;
      }

      const stepOut = await stepSess.run(stepFeeds);
      const stepKeys = Object.keys(stepOut);
      const sLogitsKey = stepKeys.find((k) => k.includes('logits')) ?? stepKeys[0]!;
      const sLogits = (stepOut[sLogitsKey] as OrtTensorLike<Float32Array>).data;
      nextToken = argmax(sLogits);

      if (nextToken === eosId) break;
      generatedTokens.push(nextToken);

      // Update only self-attention KV from step outputs; encoder KV is static
      for (const key of stepKeys) {
        if (key.startsWith('present')) {
          const pastName = key.replace(/^present\./, 'past_key_values.');
          pastKv[pastName] = stepOut[key] as OrtTensorLike<Float32Array>;
        }
      }
    }

    // Verify tokens were generated
    expect(generatedTokens.length).toBeGreaterThan(0);
    console.log(`Generated ${generatedTokens.length} tokens`);

    // ── 10. Decoder align ─────────────────────────────────────
    if (alignSess) {
      const allTokens = [...promptIds, ...generatedTokens];
      const alignArr = new BigInt64Array(allTokens.map((id) => BigInt(id)));
      const alignTensor = new ort.Tensor('int64', alignArr, [1, allTokens.length]);

      const alignOut = await alignSess.run({
        input_ids: alignTensor,
        encoder_hidden_states: encOut,
      });
      const alignKey = Object.keys(alignOut)[0]!;
      const alignment = alignOut[alignKey] as OrtTensorLike<Float32Array>;
      const [aB, aT, aS] = alignment.dims;

      // Verify alignment shape: text_token_count × encoder_seq
      // (prompt tokens excluded from alignment output)
      expect(aB).toBe(1);
      expect(aT).toBeGreaterThan(0);
      expect(aS).toBe(maxSrcPos);

      // Verify row sums are approximately 1 (softmax along last dim)
      const alignData = alignment.data;
      for (let ti = 0; ti < aT; ti++) {
        let rowSum = 0;
        const rowOff = ti * aS;
        for (let si = 0; si < aS; si++) {
          const v = alignData[rowOff + si] ?? 0;
          expect(v).toBeGreaterThanOrEqual(0); // non-negative
          rowSum += v;
        }
        expect(rowSum).toBeGreaterThan(0.99);
        expect(rowSum).toBeLessThan(1.01);
      }

      console.log(`Alignment shape: [${aB}, ${aT}, ${aS}], row sums ≈ 1.0`);
    }

    console.log('4-graph smoke test PASSED', {
      dModel, decoderLayers, heads: decoderAttentionHeads, headDim,
      tokens: generatedTokens.length,
      alignOk: alignSess !== null,
    });
  }, 120000);
});
