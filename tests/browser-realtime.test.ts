import { createBrowserRealtimeStarter } from '@asrjs/speech-recognition/browser';
import { describe, expect, it } from 'vitest';

describe('browser realtime starter', () => {
  it('requires a transcribe callback when controllerOptions are provided', () => {
    expect(() =>
      createBrowserRealtimeStarter({
        controllerOptions: {
          finalizeSilenceSeconds: 0.5,
        },
      }),
    ).toThrow('requires transcribe when controllerOptions are provided');
  });
});
