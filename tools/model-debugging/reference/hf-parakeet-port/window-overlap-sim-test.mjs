import assert from 'node:assert/strict';
import { simulateOverlappedWindowInference } from '../src/benchmark-utils.js';

function assertClose(actual, expected, tolerance, label) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, got ${actual}`
  );
}

const referenceAudioSec = 90 * 60;

{
  const result = simulateOverlappedWindowInference({
    totalAudioSec: referenceAudioSec,
    windowSec: 80,
    overlapSec: 6,
    rawRtfx: 86.5,
  });
  assert.equal(result.steps, 73);
  assert.equal(result.processedSec, 5840);
  assert.equal(result.overheadSec, 440);
  assertClose(result.overheadRatio, 440 / 5400, 1e-6, '80s overhead ratio');
}

{
  const result = simulateOverlappedWindowInference({
    totalAudioSec: referenceAudioSec,
    windowSec: 60,
    overlapSec: 6,
    rawRtfx: 86.5,
  });
  assert.equal(result.steps, 100);
  assert.equal(result.processedSec, 6000);
  assert.equal(result.overheadSec, 600);
  assertClose(result.overheadRatio, 600 / 5400, 1e-6, '60s overhead ratio');
}

{
  const result = simulateOverlappedWindowInference({
    totalAudioSec: referenceAudioSec,
    windowSec: 90,
    overlapSec: 6,
    rawRtfx: 86.5,
  });
  assert.equal(result.steps, 65);
  assert.equal(result.processedSec, 5850);
  assert.equal(result.overheadSec, 450);
  assertClose(result.overheadRatio, 450 / 5400, 1e-6, '90s overhead ratio');
}

console.log('window overlap simulation checks passed');
