import { describe, expect, it } from 'vitest';
import { createCaptureTranscript } from '../src/runtime/realtime.js';

describe('capture context transcript revisions', () => {
  it('reuses the established demo policy: replace revisions, retain independent turns', () => {
    const capture = createCaptureTranscript('Earlier recording.');
    expect(capture.update('ask', { id: 'a', revision: 1 })).toBe('Earlier recording. ask');
    expect(capture.update('ask what you can do', { id: 'a', revision: 2 })).toBe(
      'Earlier recording. ask what you can do',
    );
    expect(capture.update('next turn', { id: 'b', revision: 1 })).toBe(
      'Earlier recording. ask what you can do next turn',
    );
  });

  it('permits corrections and empty replacements while ignoring stale revisions', () => {
    const capture = createCaptureTranscript();
    capture.update('wrong', { id: 'a', revision: 1 });
    expect(capture.update('', { id: 'a', revision: 2 })).toBe('');
    expect(capture.update('stale', { id: 'a', revision: 1 })).toBe('');
    expect(capture.update('correct', { id: 'a', revision: 3 })).toBe('correct');
  });

  it('keeps intentional repetition across contexts and independent recordings', () => {
    const capture = createCaptureTranscript();
    capture.update('yes yes', { id: 'a', revision: 1 });
    expect(capture.update('yes', { id: 'b', revision: 1 })).toBe('yes yes yes');
    expect(capture.update('yes')).toBe('yes yes yes yes');
    expect(createCaptureTranscript().text()).toBe('');
  });
});
