#!/usr/bin/env node
/**
 * Quality Gates Smoke Test
 *
 * Tests all 4 quality gates + temperature fallback + composite evaluator.
 * Uses synthetic fixtures for controlled gate evaluation.
 *
 * Usage:
 *   node tests/smoke/quality-gates-smoke.mjs          # unit tests only (fast)
 *   RUN_ASR=1 node tests/smoke/quality-gates-smoke.mjs # + ASR integration
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../..');

// ---------------------------------------------------------------------------
let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (condition) { passed++; }
  else { console.error(`  FAIL: ${msg}`); failed++; }
}

function assertApprox(actual, expected, tolerance, msg) {
  const ok = Math.abs(actual - expected) <= tolerance;
  if (ok) { passed++; }
  else {
    console.error(`  FAIL: ${msg} (expected ${expected}, got ${actual})`);
    failed++;
  }
}

// ---------------------------------------------------------------------------
// Synthetic logit builders
// ---------------------------------------------------------------------------

/** Build per-step logit vectors: one dominant token at dominantVal, rest at fill. */
function buildLogits(vocabSize, numSteps, dominant, dominantVal, fill = -100) {
  const out = [];
  for (let s = 0; s < numSteps; s++) {
    const v = new Float32Array(vocabSize);
    v.fill(fill);
    v[dominant] = dominantVal;
    out.push(v);
  }
  return out;
}

/**
 * Build per-step logits with exactly 2 competitive tokens.
 * Used to test noSpeechGate: token 50362 is one competitor, `chosen` is the other.
 * Chosen token has lower logit → low probability → low avgLogProb.
 */
function buildDualCompetitorLogits(vocabSize, numSteps, tokenA, logitA, tokenB, logitB, fill = -100) {
  const out = [];
  for (let s = 0; s < numSteps; s++) {
    const v = new Float32Array(vocabSize);
    v.fill(fill);
    v[tokenA] = logitA;
    v[tokenB] = logitB;
    out.push(v);
  }
  return out;
}

// ---------------------------------------------------------------------------
async function run() {
  // ── Inline fixtures (tests/fixtures/ dir is gitignored) ──
  const fixtures = {
    compressionRatio: {
      normalText: 'Merhaba dünya nasılsın bugün hava çok güzel',
      repetitiveText: 'AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA AAAA',
    },
    logitConfig: {
      highConfidence: { vocabSize: 10, logitMin: -100, logitMax: 20, dominantToken: 3, numSteps: 3, expectedAvgLogProb: 0, expectedVerdict: 'accept' },
      lowConfidence: { vocabSize: 100, logitMin: 0, logitMax: 0, dominantToken: 3, numSteps: 5, expectedAvgLogProb: -4.605, expectedVerdict: 'reject' },
      lowEntropy: { vocabSize: 10, logitMin: -1000, logitMax: 100, dominantToken: 0, numSteps: 3, expectedEntropy: 0, expectedVerdict: 'accept' },
      highEntropy: { vocabSize: 12, logitMin: 0, logitMax: 0, dominantToken: 0, numSteps: 3, expectedEntropy: 2.485, expectedVerdict: 'reject' },
    },
    whisperBase: { modelPath: '/tmp/whisper-base-4graph/fp32', fixture12Dans: 'tests/fixtures/12_dans.tr.m4a', lang: 'tr' },
  };
  console.log(`\n=== Quality Gates Smoke Test ===\n`);

  // ── 1. compressionRatioGate ──
  console.log('── compressionRatioGate ──');
  const { compressionRatioGate } = await loadQuality('compression-ratio.js');
  const crGateDefault = compressionRatioGate();

  // Normal text → accept
  const r1 = crGateDefault(fixtures.compressionRatio.normalText);
  assert(r1.verdict === 'accept', `normal text: ${r1.verdict} (ratio=${r1.compressionRatio?.toFixed(2)})`);
  assert(typeof r1.compressionRatio === 'number', `ratio is number: ${typeof r1.compressionRatio}`);

  // Repetitive text → reject (high compression)
  const r2 = crGateDefault(fixtures.compressionRatio.repetitiveText);
  assert(r2.verdict === 'reject', `repetitive text: ${r2.verdict} (ratio=${r2.compressionRatio?.toFixed(1)})`);
  const repRatio = r2.compressionRatio ?? 0;
  assert(repRatio > 3, `compression ratio > 3: ${repRatio.toFixed(1)}`);
  console.log(`    Repetitive text ratio: ${repRatio.toFixed(2)}`);

  // Repetitive text with much higher threshold (ratio+50%) → accept
  const crGateVeryHigh = compressionRatioGate(repRatio + repRatio * 0.5);
  const r3 = crGateVeryHigh(fixtures.compressionRatio.repetitiveText);
  assert(r3.verdict === 'accept', `repetitive text + adaptive threshold: ${r3.verdict}`);

  // Empty text → ratio = 1 (length 0 / max(compressed.len, 1))
  const rEmpty = crGateDefault('');
  assert(rEmpty.verdict === 'accept', `empty text: ${rEmpty.verdict}`);
  assert(rEmpty.compressionRatio === 0, `empty ratio: ${rEmpty.compressionRatio}`);

  console.log('  OK\n');

  // ── 2. logProbGate ──
  console.log('── logProbGate ──');
  const { logProbGate } = await loadQuality('log-probability.js');
  const lpGate = logProbGate();

  // High confidence: dominant token has very high logit, all others extremely low
  const hc = fixtures.logitConfig.highConfidence;
  const hcLogits = buildLogits(hc.vocabSize, hc.numSteps, hc.dominantToken, hc.logitMax, hc.logitMin);
  const hcTokens = new Array(hc.numSteps).fill(hc.dominantToken);
  const r4 = lpGate('high conf', hcTokens, hcLogits, hc.vocabSize);
  assert(r4.verdict === hc.expectedVerdict, `high confidence: ${r4.verdict} (avgLogProb=${r4.avgLogProb?.toFixed(3)})`);

  // Low confidence: uniform distribution (all logits = 0) → low probability for chosen token
  const lc = fixtures.logitConfig.lowConfidence;
  const lcLogits = buildLogits(lc.vocabSize, lc.numSteps, lc.dominantToken, lc.logitMax, lc.logitMin);
  const lcTokens = new Array(lc.numSteps).fill(lc.dominantToken);
  const r5 = lpGate('low conf', lcTokens, lcLogits, lc.vocabSize);
  assert(r5.verdict === lc.expectedVerdict, `low confidence: ${r5.verdict} (avgLogProb=${r5.avgLogProb?.toFixed(3)})`);
  assertApprox(r5.avgLogProb ?? 0, lc.expectedAvgLogProb, 0.01, `avgLogProb ≈ ${lc.expectedAvgLogProb}`);

  // Custom very low threshold → even uniform gets accepted
  const lpGateLenient = logProbGate(-10.0);
  const r6 = lpGateLenient('low conf lenient', lcTokens, lcLogits, lc.vocabSize);
  assert(r6.verdict === 'accept', `uniform with threshold=-10: ${r6.verdict}`);

  console.log('  OK\n');

  // ── 3. entropyGate ──
  console.log('── entropyGate ──');
  const { entropyGate } = await loadQuality('entropy.js');
  const eGate = entropyGate();

  // Low entropy: one token dominates
  const le = fixtures.logitConfig.lowEntropy;
  const leLogits = buildLogits(le.vocabSize, le.numSteps, le.dominantToken, le.logitMax, le.logitMin);
  const r7 = eGate('low entropy', [], leLogits, le.vocabSize);
  assert(r7.verdict === le.expectedVerdict, `low entropy: ${r7.verdict} (entropy=${r7.entropy?.toFixed(3)})`);

  // High entropy: uniform distribution over 12 tokens → H = ln(12) ≈ 2.485
  const he = fixtures.logitConfig.highEntropy;
  const heLogits = buildLogits(he.vocabSize, he.numSteps, he.dominantToken, he.logitMax, he.logitMin);
  const r8 = eGate('high entropy', [], heLogits, he.vocabSize);
  assert(r8.verdict === he.expectedVerdict, `high entropy: ${r8.verdict} (entropy=${r8.entropy?.toFixed(3)})`);
  assertApprox(r8.entropy ?? 0, he.expectedEntropy, 0.01, `entropy ≈ ${he.expectedEntropy}`);

  console.log('  OK\n');

  // ── 4. noSpeechGate ──
  console.log('── noSpeechGate ──');
  const { noSpeechGate } = await loadQuality('no-speech.js');
  const nsGate = noSpeechGate();

  // High no-speech: token 50362 dominates but model chooses a different token (low confidence)
  // Token 50362 = 3.0 → softmax prob ≈ 0.73 > 0.6
  // Chosen token = 1 at logit 2.0 → prob ≈ 0.27 → log(0.27) ≈ -1.31 < -1.0
  const nsHighLogits = buildDualCompetitorLogits(50363, 1, 50362, 3.0, 1, 2.0, -100);
  const r9 = nsGate('ns high', [1], nsHighLogits, 50363);
  assert(r9.verdict === 'no_speech', `high no_speech: ${r9.verdict} (noSpeechProb=${r9.noSpeechProb?.toFixed(3)}, avgLogProb=${r9.avgLogProb?.toFixed(3)})`);
  assert((r9.noSpeechProb ?? 0) > 0.6, `noSpeechProb > 0.6: ${r9.noSpeechProb?.toFixed(3)}`);
  assert((r9.avgLogProb ?? 0) < -1.0, `avgLogProb < -1.0: ${r9.avgLogProb?.toFixed(3)}`);

  // Low no-speech: token 1 dominates, token 50362 very low → no_speech_prob ≈ 0
  // Token 1 = 10.0, token 50362 = -100
  const nsLowLogits = buildDualCompetitorLogits(50363, 1, 1, 10.0, 50362, -100, -200);
  const r10 = nsGate('ns low', [1], nsLowLogits, 50363);
  assert(r10.verdict === 'accept', `low no_speech: ${r10.verdict} (noSpeechProb=${r10.noSpeechProb?.toFixed(3)})`);
  assert((r10.noSpeechProb ?? 1) < 0.6, `noSpeechProb < 0.6: ${r10.noSpeechProb?.toFixed(3)}`);

  console.log('  OK\n');

  // ── 5. evaluateGates (composite) ──
  console.log('── evaluateGates (composite) ──');
  const { evaluateGates } = await loadQuality('index.js');

  // All gates accept
  const allAccept = evaluateGates(
    fixtures.compressionRatio.normalText, hcTokens, hcLogits, hc.vocabSize,
    [compressionRatioGate(), logProbGate(), entropyGate(), noSpeechGate()],
  );
  assert(allAccept.verdict === 'accept', `all gates accept: ${allAccept.verdict}`);

  // First gate rejects (compression ratio too high)
  const firstRejects = evaluateGates(
    fixtures.compressionRatio.repetitiveText, [], [], 100,
    [compressionRatioGate()],
  );
  assert(firstRejects.verdict === 'reject', `compression rejects first: ${firstRejects.verdict}`);

  // No speech short-circuits (precedes other gates)
  const noSpeech = evaluateGates(
    '', [1], nsHighLogits, 50363,
    [noSpeechGate(), compressionRatioGate()],
  );
  assert(noSpeech.verdict === 'no_speech', `no_speech short-circuits: ${noSpeech.verdict}`);

  console.log('  OK\n');

  // ── 6. withTemperatureFallback ──
  console.log('── withTemperatureFallback ──');
  const { withTemperatureFallback } = await loadQuality('temperature-fallback.js');

  // Mock: all chosen tokens have high logit → all gates accept
  const mockHighConf = async () => {
    const logits = [
      buildDominantLogit(10, 1, 20),
      buildDominantLogit(10, 2, 20),
      buildDominantLogit(10, 3, 20),
    ];
    return {
      result: { text: 'mock ok' },
      text: 'mock ok',
      tokens: [1, 2, 3],
      logits,
      vocabSize: 10,
    };
  };

  // Accepts on first try (temperature=0)
  const fr1 = await withTemperatureFallback(mockHighConf, [compressionRatioGate(), logProbGate()], [0.0, 0.2]);
  assert(fr1.temperature === 0, `fallback accepts at temp=0: temp=${fr1.temperature}`);

  // No speech on first try → immediate return (no fallback)
  const mockNoSpeech = async () => ({
    result: { text: '' },
    text: '',
    tokens: [1],
    logits: buildDualCompetitorLogits(50363, 1, 50362, 3.0, 1, 2.0, -100),
    vocabSize: 50363,
  });
  const fr2 = await withTemperatureFallback(mockNoSpeech, [noSpeechGate()], [0.0, 0.2, 0.4]);
  assert(fr2.temperature === 0, `no-speech immediate: temp=${fr2.temperature}`);

  // All temperatures exhausted → returns last result
  let callC = 0;
  const alwaysReject = () => (_t, _tok, _l, _v) => ({ verdict: 'reject', reason: 'always' });
  const fr3 = await withTemperatureFallback(
    async (t) => {
      callC++;
      const logits = buildLogits(10, 1, 1, -100, -200);
      return { result: 'final_result', text: 'exhausted', tokens: [1], logits, vocabSize: 10 };
    },
    [alwaysReject()],
    [0.0, 0.2, 0.4],
  );
  assert(fr3.result === 'final_result', `all exhausted returns last result: ${fr3.result}`);
  assert(callC === 3, `all exhausted called 3 times: ${callC}`);

  console.log('  OK\n');

  // ── Summary ──
  const total = passed + failed;
  console.log(`=== Results: ${passed}/${total} passed, ${failed} failed ===\n`);

  // ── 7. ASR integration test (optional) ──
  if (process.env.RUN_ASR) {
    console.log('── ASR Integration: whisperx-runner with quality gates ──\n');
    await testAsrIntegration(fixtures);
  }

  process.exit(failed > 0 ? 1 : 0);
}

// ---------------------------------------------------------------------------
// ASR integration — runs the refactored whisperx-runner against Turkish fixture
async function testAsrIntegration(fixtures) {
  const modelPath = process.env.WHISPER_MODEL_DIR || fixtures.whisperBase.modelPath;
  const modelExists = fs.existsSync(path.join(modelPath, 'encoder_model.onnx'));

  if (!modelExists) {
    console.log('  SKIP: model not found at', modelPath);
    return;
  }

  const audioPath = path.join(REPO_ROOT, fixtures.whisperBase.fixture12Dans);
  if (!fs.existsSync(audioPath)) {
    console.log('  SKIP: audio fixture not found at', audioPath);
    return;
  }

  // Dynamically import the runner's pipeline function
  let runnerModule;
  try {
    runnerModule = await import(path.join(REPO_ROOT, 'tests/smoke/whisperx-runner.mjs'));
  } catch {
    console.log('  SKIP: whisperx-runner not yet refactored for module export');
    return;
  }

  if (!runnerModule.runAsrPipeline) {
    console.log('  SKIP: runAsrPipeline not exported (runner not yet refactored)');
    return;
  }

  const result = await runnerModule.runAsrPipeline({
    model: modelPath,
    language: 'tr',
    vadOnset: 0.5,
    verbose: true,
    compressionRatioThreshold: 2.4,
    logprobThreshold: -1.0,
    noSpeechThreshold: 0.6,
    entropyThreshold: 2.4,
    temperature: 0.0,
    temperatureIncrement: 0.2,
    audioPath,
  });

  console.log(`\n  Segments: ${result.segments?.length || 0}`);
  console.log(`  Words: ${result.wordCount ?? 0}`);
  console.log(`  ASR time: ${result.asrTime?.toFixed(1) ?? '?'}s`);
  console.log(`  Temperatures used: ${result.temperaturesUsed?.join(', ') || 'none'}`);
  console.log(`  Fallbacks triggered: ${result.fallbackCount ?? 0}`);

  // Compare with reference VTT
  const refVtt = path.join(REPO_ROOT, 'tests/fixtures', '12_dans.tr.vtt');
  if (fs.existsSync(refVtt) && result.fullText) {
    const refContent = fs.readFileSync(refVtt, 'utf-8');
    const refWords = refContent.split('\n')
      .filter(l => l && !l.startsWith('WEBVTT') && !l.startsWith('Kind:') && !l.startsWith('Language:') && !l.includes('-->'))
      .map(l => l.trim()).filter(Boolean).join(' ').split(/\s+/);
    const hypWords = result.fullText.split(/\s+/).filter(Boolean);

    const refSet = new Set(refWords.map(w => w.toLowerCase()));
    const hypSet = new Set(hypWords.map(w => w.toLowerCase()));
    const correct = [...hypSet].filter(w => refSet.has(w)).length;
    const wer = hypSet.size > 0
      ? (hypSet.size - correct + [...refSet].filter(w => !hypSet.has(w)).length) / refSet.size
      : 1;

    console.log(`  Reference words: ${refWords.length}`);
    console.log(`  Hypothesis words: ${hypWords.length}`);
    console.log(`  Correct unique: ${correct}`);
    console.log(`  Estimated WER: ${(wer * 100).toFixed(1)}%`);
    console.log(`  Quality gates active: ${result.gateResults ? 'yes' : 'no'}`);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a single logit vector with one dominant token. */
function buildDominantLogit(vocabSize, dominant, dominantVal, fill = -100) {
  const v = new Float32Array(vocabSize);
  v.fill(fill);
  v[dominant] = dominantVal;
  return v;
}

/** Dynamically import quality modules from dist/ */
async function loadQuality(module) {
  return import(path.join(REPO_ROOT, 'dist', 'quality', module));
}

// ---------------------------------------------------------------------------
run().catch(e => { console.error(e.stack); process.exit(1); });
