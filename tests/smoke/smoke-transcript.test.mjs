import assert from 'node:assert/strict';
import test from 'node:test';
import { matchesSmokeTranscript } from './smoke-transcript.mjs';

test('requires the whole normalized transcript, not a substring', () => {
  assert.equal(matchesSmokeTranscript('The boy was there when the sun rose.', 'a'), false);
  assert.equal(matchesSmokeTranscript('The boy was there when the sun rose.', 'the boy'), false);
  assert.equal(matchesSmokeTranscript('The boy was there when the sun rose.', 'The boy was there when the sun rose'), true);
  assert.equal(matchesSmokeTranscript('Extra The boy was there when the sun rose.', 'The boy was there when the sun rose'), false);
  assert.equal(matchesSmokeTranscript('anything', '...'), false);
});
