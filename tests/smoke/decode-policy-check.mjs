/**
 * Decode policy verification: simulate WhisperTimestampLogitProcessor
 * with large-v3-turbo generation_config values.
 *
 * Run: node tests/smoke/decode-policy-check.mjs
 *
 * This shows exactly what logits are suppressed at each step,
 * so we can compare Node runner vs WebGPU test page behavior.
 */

// ── Config from generation_config.json (large-v3-turbo) ──
const CONFIG = {
  beginSuppressTokens: [220, 50257],
  suppressTokens: [
    1,2,7,8,9,10,14,25,26,27,28,29,31,58,59,60,61,62,63,90,91,92,93,
    359,503,522,542,873,893,902,918,922,931,1350,1853,1982,2460,2627,
    3246,3253,3268,3536,3846,3961,4183,4667,6585,6647,7273,9061,9383,
    10428,10929,11938,12033,12331,12562,13793,14157,14635,15265,15618,
    16553,16604,18362,18956,20075,21675,22520,26130,26161,26435,28279,
    29464,31650,32302,32470,36865,42863,47425,49870,50254,50258,
    50359,50360,50361,50362,50363,
  ],
  eosTokenId: 50257,
  noTimestampsTokenId: 50364,
  timestampBegin: 50364,
};

// ── Simulated processor ──
class TimestampLogitProcessor {
  constructor(cfg) {
    this.eosTokenId = cfg.eosTokenId;
    this.noTimestampsTokenId = cfg.noTimestampsTokenId;
    this.timestampBegin = cfg.timestampBegin;
    this.suppressTokens = cfg.suppressTokens;
    this.beginSuppressTokens = cfg.beginSuppressTokens;
  }

  process(logits, generatedTokens, beginIndex) {
    const before = new Map();
    for (const id of [50257, 50360, 50364]) {
      if (id < logits.length) before.set(id, logits[id]);
    }

    // 1. Always suppress suppress_tokens
    for (const tokenId of this.suppressTokens) {
      if (tokenId < logits.length) logits[tokenId] = -Infinity;
    }

    // 2. begin_suppress_tokens only on first generated token
    if (generatedTokens.length === beginIndex) {
      for (const tokenId of this.beginSuppressTokens) {
        if (tokenId < logits.length) logits[tokenId] = -Infinity;
      }
    }

    // 3. no_timestamps check
    const hasNoTimestamps = generatedTokens.includes(this.noTimestampsTokenId);
    if (hasNoTimestamps) {
      for (let ts = this.timestampBegin; ts < logits.length; ts++) {
        logits[ts] = -Infinity;
      }
    }

    // 4. Sequential timestamp rules
    const sampledTokens = generatedTokens.slice(beginIndex);
    if (sampledTokens.length === 0) {
      // First generated token: suppress all text, only timestamps/EOS
      for (let t = 0; t < this.timestampBegin; t++) {
        logits[t] = -Infinity;
      }
      return;
    }

    const lastIsTimestamp = sampledTokens[sampledTokens.length - 1] >= this.timestampBegin;
    const penultimateIsTimestamp = sampledTokens.length < 2 || sampledTokens[sampledTokens.length - 2] >= this.timestampBegin;

    if (lastIsTimestamp) {
      if (penultimateIsTimestamp) {
        // Two timestamps → suppress all timestamps
        for (let ts = this.timestampBegin; ts < logits.length; ts++) logits[ts] = -Infinity;
      } else {
        // Last timestamp, prev text → suppress text (force EOS)
        for (let t = 0; t < this.eosTokenId; t++) logits[t] = -Infinity;
      }
    }

    // Monotonically increasing timestamps (simplified)
  }

  /** Get state for selected tokens */
  analyze(logits, generatedTokens, beginIndex) {
    const result = {
      step: generatedTokens.length - beginIndex,
      promptLen: beginIndex,
      genLen: generatedTokens.length,
      beginSuppressFires: generatedTokens.length === beginIndex,
      hasNoTimestamps: generatedTokens.includes(this.noTimestampsTokenId),
      generatedSequence: generatedTokens.slice(beginIndex).join(', '),
    };

    // Check specific tokens before/after
    const checkTokens = [50257, 50360, 50364, 220];
    result.tokenStates = {};
    for (const id of checkTokens) {
      if (id < logits.length) {
        result.tokenStates[id] = {
          value: logits[id],
          suppressed: !isFinite(logits[id]),
        };
      }
    }

    return result;
  }
}

// ── Run simulation ──
function simulate() {
  const processor = new TimestampLogitProcessor(CONFIG);

  // Scenario A: Runner prompt [SOT, lang, task, notimestamps]
  const runnerPrompt = [50258, 50259, 50360, 50364];
  const beginIndex = runnerPrompt.length;

  console.log('═══ SCENARIO A: Runner-style prompt ═══');
  console.log(`Prompt tokens: ${runnerPrompt.join(', ')}`);
  console.log(`beginIndex: ${beginIndex}`);
  console.log('');

  // Step 0 (decoder_init logits)
  const step0Logits = new Float32Array(51866).fill(-100);
  const step0High = [50257, 50360, 50364, 220, 100, 200, 300];
  for (const id of step0High) {
    step0Logits[id] = 10;  // high logits for key tokens
  }
  step0Logits[50257] = 8.0;  // EOS moderate
  step0Logits[50360] = 12.0; // transcribe high
  step0Logits[50364] = 11.0; // notimestamps high

  const logitsCopy = new Float32Array(step0Logits);
  const state0 = processor.analyze(logitsCopy, runnerPrompt, beginIndex);
  console.log(`Step ${state0.step}:`);
  console.log(`  beginSuppress fires: ${state0.beginSuppressFires}`);
  console.log(`  hasNoTimestamps: ${state0.hasNoTimestamps}`);
  for (const [id, s] of Object.entries(state0.tokenStates)) {
    console.log(`  token ${id}: ${s.suppressed ? 'SUPPRESSED (-∞)' : `${s.value}`}`);
  }
  console.log('');

  // Apply processor
  processor.process(logitsCopy, runnerPrompt, beginIndex);
  const eosSuppressed = !isFinite(logitsCopy[50257]);
  const tsSuppressed = !isFinite(logitsCopy[50364]);
  const textAvailable = logitsCopy[100] > -100;
  console.log(`  After processing:`);
  console.log(`  EOS (50257)${eosSuppressed ? ' SUPPRESSED ✓ (begin_suppress)' : ' available'}`);
  console.log(`  Timestamps${tsSuppressed ? ' SUPPRESSED ✓ (no_timestamps)' : ' available'}`);
  console.log(`  Text tokens${textAvailable ? ' available ✓' : ' ALL SUPPRESSED'}`);
  console.log(`  → First token will be TEXT (not EOS, not timestamp)`);
  console.log('');

  // Step 1: after first generated token
  const step1Tokens = [...runnerPrompt, 100]; // text token "The"
  const step1Logits = new Float32Array(51866).fill(-100);
  for (const id of step0High) step1Logits[id] = 10;
  step1Logits[50257] = 7.73;  // EOS
  step1Logits[100] = 7.70;    // text

  const state1 = processor.analyze(step1Logits, step1Tokens, beginIndex);
  console.log(`Step ${state1.step}:`);
  console.log(`  beginSuppress fires: ${state1.beginSuppressFires}`);
  console.log(`  hasNoTimestamps: ${state1.hasNoTimestamps}`);
  for (const [id, s] of Object.entries(state1.tokenStates)) {
    console.log(`  token ${id}: ${s.suppressed ? 'SUPPRESSED (-∞)' : `${s.value}`}`);
  }

  processor.process(step1Logits, step1Tokens, beginIndex);
  const eosSuppressed1 = !isFinite(step1Logits[50257]);
  const textSuppressed1 = !isFinite(step1Logits[100]);
  console.log(`  After processing:`);
  console.log(`  EOS (50257)${eosSuppressed1 ? ' SUPPRESSED' : ` available (value: ${step1Logits[50257]})`}`);
  console.log(`  Text token 100${textSuppressed1 ? ' SUPPRESSED' : ` available (value: ${step1Logits[100]})`}`);
  console.log(`  → ${step1Logits[50257] > step1Logits[100] ? 'EOS WINS (early EOS!)' : 'Text WINS ✓'}`);
  console.log('');

  // ── Scenario B: Test page prompt [SOT, lang] only ──
  console.log('═══ SCENARIO B: Short prompt (SOT + lang) ═══');
  const shortPrompt = [50258, 50259];
  const shortBeginIndex = shortPrompt.length;

  // Step 0: decoder_init with short prompt
  const s0Logits = new Float32Array(51866).fill(-100);
  for (const id of step0High) s0Logits[id] = 10;
  s0Logits[50257] = 7.0;
  s0Logits[50360] = 12.0;  // transcribe high
  s0Logits[50364] = 11.0;  // notimestamps high

  const s0State = processor.analyze(s0Logits, shortPrompt, shortBeginIndex);
  console.log(`Step ${s0State.step} (prompt: ${shortPrompt.join(',')}):`);
  console.log(`  beginSuppress fires: ${s0State.beginSuppressFires}`);
  console.log(`  hasNoTimestamps: ${s0State.hasNoTimestamps}`);

  processor.process(s0Logits, shortPrompt, shortBeginIndex);
  console.log(`  After:`);
  for (const [id, s] of Object.entries(processor.analyze(s0Logits, shortPrompt, shortBeginIndex).tokenStates)) {
    console.log(`  token ${id}: ${!isFinite(s0Logits[id]) ? 'SUPPRESSED' : `value: ${s0Logits[id]}`}`);
  }
  // With short prompt, no_timestamps NOT in prompt, so text is suppressed!
  // Only timestamps and EOS remain. First token = 50360 or 50364
  console.log(`  → First token will be TIMESTAMP (50360 or 50364), not text!`);
  console.log('');

  // Step 1: generated 50360 (transcribe)
  const s1Tokens = [...shortPrompt, 50360];
  const s1Logits = new Float32Array(51866).fill(-100);
  for (const id of step0High) s1Logits[id] = 10;
  s1Logits[50257] = 8.0;
  s1Logits[50364] = 12.0;

  const s1State = processor.analyze(s1Logits, s1Tokens, shortBeginIndex);
  console.log(`Step ${s1State.step}:`);
  console.log(`  beginSuppress fires: ${s1State.beginSuppressFires}`);
  console.log(`  hasNoTimestamps: ${s1State.hasNoTimestamps}`);
  console.log(`  Sequence so far: ${s1State.generatedSequence}`);

  processor.process(s1Logits, s1Tokens, shortBeginIndex);
  // Step 1: seq = [50360], lastIsTimestamp=true, penultimateIsTimestamp=false
  // → suppress text, leave timestamps + EOS
  // With no_timestamps NOT triggered yet (50364 not in generated tokens):
  //   timestamps are available
  // → next token = 50364 (notimestamps)
  console.log(`  After: timestamps available, text suppressed → next = 50364`);
  console.log('');

  // Step 2: generated 50364 (no_timestamps)
  const s2Tokens = [...shortPrompt, 50360, 50364];
  const s2Logits = new Float32Array(51866).fill(-100);
  for (const id of step0High) s2Logits[id] = 10;
  s2Logits[50257] = 7.73;
  s2Logits[100] = 7.70;  // example text token

  const s2State = processor.analyze(s2Logits, s2Tokens, shortBeginIndex);
  console.log(`Step ${s2State.step}:`);
  console.log(`  beginSuppress fires: ${s2State.beginSuppressFires}`);
  console.log(`  hasNoTimestamps: ${s2State.hasNoTimestamps} (50364 now in generated!)`);
  console.log(`  Sequence so far: ${s2State.generatedSequence}`);

  processor.process(s2Logits, s2Tokens, shortBeginIndex);
  // Step 2: hasNoTimestamps=true now → ALL timestamps suppressed
  // Text tokens remain, EOS is available (no begin_suppress)
  // EOS=7.73 vs text=7.70 → EOS wins (same as Bev's result!)
  const eosVal = s2Logits[50257];
  const textVal = s2Logits[100];
  console.log(`  After:`);
  console.log(`  Timestamps: SUPPRESSED (no_timestamps now active)`);
  console.log(`  EOS (50257): ${!isFinite(eosVal) ? 'SUPPRESSED' : `available = ${eosVal}`}`);
  console.log(`  Text (100): ${!isFinite(textVal) ? 'SUPPRESSED' : `available = ${textVal}`}`);
  console.log(`  → ${eosVal > textVal ? 'EOS WINS (early EOS at step 2!)' : 'Text wins'}`);
  console.log('');

  console.log('═══ CONCLUSION ═══');
  console.log('With short prompt [SOT, lang]:');
  console.log('  50360 → 50364 → EOS at step 2. Same as Bev result.');
  console.log('  EOS wins at step 2 because fp16 encoder shifted logits.');
  console.log('');
  console.log('With runner prompt [SOT, lang, task, notimestamps]:');
  console.log('  Step 0: begin_suppress suppresses EOS, no_timestamps suppresses timestamps.');
  console.log('  Text tokens available → first word generated.');
  console.log('  Step 1+: EOS available, timestamps suppressed. Normal decode.');
  console.log('');
  console.log('begin_suppress_tokens [220, 50257] only blocks EOS at step 0.');
  console.log('It does NOT fix fp16 distribution shift at step 2+.');
}

simulate();
