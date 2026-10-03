import { describe, expect, it, vi } from 'vitest';
import {
  parseArgs,
  runDisposalDrainAgainstSession,
  runInspectorDisposalDrain,
  validateInspectorOptions,
} from '../tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs';

const reference = {
  tokens: [1],
  token_frames: [0],
  token_duration: [1],
  transcript: 'probe',
};
const progress = { stage: 'decode', completedUnits: 1, totalUnits: 10 };

interface InspectorOutput {
  canonical: { text: string };
  native: { debug: { tokenIds: number[]; frameIndices: number[]; tdtSteps: number[] } };
}

interface DrainProgressOptions {
  onProgress?: (event: { stage: string; completedUnits?: number }) => void;
}

interface DrainTarget {
  transcribe: (input: Float32Array, options: DrainProgressOptions) => Promise<InspectorOutput>;
  order: string[];
  dispose(): Promise<void>;
}

const output: InspectorOutput = {
  canonical: { text: 'probe' },
  native: { debug: { tokenIds: [1], frameIndices: [0], tdtSteps: [1] } },
};

function divergentOutput(): InspectorOutput {
  return { ...output, native: { debug: { ...output.native.debug, tokenIds: [2] } } };
}

/**
 * A target whose disposal only settles after the accepted call finishes.
 * Disposal completing earlier is the defect this control exists to catch.
 */
function createDrainTarget({
  result = output,
  failOnLaterCall = true,
  dispose = 'wait',
}: {
  result?: InspectorOutput;
  failOnLaterCall?: boolean;
  dispose?: 'wait' | 'early' | 'fail';
} = {}): DrainTarget {
  let acceptedSettled = false;
  const order: string[] = [];
  const transcribe = vi.fn(
    async (_input: Float32Array, received: DrainProgressOptions): Promise<InspectorOutput> => {
      if (acceptedSettled && failOnLaterCall) {
        throw new Error('Speech model handle is disposed.');
      }
      received.onProgress?.(progress);
      await Promise.resolve();
      acceptedSettled = true;
      order.push('accepted');
      return result;
    },
  );
  return {
    transcribe,
    order,
    async dispose(): Promise<void> {
      order.push('dispose');
      if (dispose === 'early') return;
      while (!acceptedSettled) {
        await Promise.resolve();
      }
      if (dispose === 'fail') throw new Error('executor release failed');
      order.push('drained');
    },
  };
}

/** A loaded handle whose disposal closes the model that owns created sessions. */
function createLoaded() {
  const direct = createDrainTarget();
  const handle = createDrainTarget();
  let disposed = false;
  return {
    loaded: {
      model: {
        createSession: vi.fn(async () => {
          if (disposed) throw new Error('Cannot create a session from a disposed model.');
          return direct;
        }),
      },
      transcribe: handle.transcribe,
      async dispose(): Promise<void> {
        disposed = true;
        await handle.dispose();
      },
    },
    handle,
    direct,
  };
}

function drain(target: DrainTarget, scope: string) {
  return runDisposalDrainAgainstSession(
    target,
    (received: DrainProgressOptions) => target.transcribe(Float32Array.of(0.1), received),
    {},
    reference,
    'fixture',
    scope,
  );
}

describe('NeMo inspector disposal-drain controls (simulated)', () => {
  it('requires an oracle and an exclusive loaded session', () => {
    const base = [
      '--family',
      'nemo-tdt',
      '--config',
      'runtime-config.json',
      '--model-dir',
      'models/oruk',
    ];
    expect(parseArgs([]).disposeDrain).toBe(false);
    expect(parseArgs([...base, '--dispose-drain']).disposeDrain).toBe(true);
    expect(() => validateInspectorOptions(parseArgs([...base, '--dispose-drain']))).toThrow(
      'requires an official --oracle',
    );
    expect(() =>
      validateInspectorOptions(
        parseArgs([
          ...base,
          '--dispose-drain',
          '--cancel-reuse',
          '--oracle',
          'o.json.gz',
          '--sample-id',
          'fixture',
          '--original-weights',
          'weights.nemo',
        ]),
      ),
    ).toThrow('request only one');
  });

  it('drains the accepted call before disposal on a direct session and the handle', async () => {
    const { loaded, handle, direct } = createLoaded();
    const result = await runInspectorDisposalDrain(
      loaded,
      Float32Array.of(0.1),
      {},
      reference,
      'fixture',
    );
    expect(result.result).toBe(output);
    expect(direct.order).toEqual(['dispose', 'accepted', 'drained']);
    expect(handle.order).toEqual(['dispose', 'accepted', 'drained']);
    expect(result.lifecycle.scope).toBe('DIRECT_SESSION_AND_LOADED_HANDLE_DISPOSAL_DRAIN');
    expect(result.lifecycle.createSessionAfterDisposal).toBe(
      'Cannot create a session from a disposed model.',
    );
    for (const record of [result.lifecycle.directSession, result.lifecycle.loadedHandle]) {
      expect(record.acceptedAt).toEqual({ stage: 'decode', completedUnits: 1 });
      expect(record.settleOrder).toEqual(['acceptedCall', 'dispose']);
      expect(record.acceptedCallSettledBeforeDisposal).toBe(true);
      expect(record.postDisposeRejection).toBe('Speech model handle is disposed.');
      expect(record.comparison.pass).toBe(true);
    }
  });

  it('fails when a disposed model still creates sessions', async () => {
    const { loaded } = createLoaded();
    loaded.model.createSession = vi.fn(async () => createDrainTarget());
    await expect(
      runInspectorDisposalDrain(loaded, Float32Array.of(0.1), {}, reference, 'fixture'),
    ).rejects.toThrow('was not rejected');
  });

  it('fails when disposal settles before the accepted call finishes', async () => {
    const target = createDrainTarget({ dispose: 'early' });
    await expect(drain(target, 'EARLY_DISPOSAL')).rejects.toThrow(
      'Disposal settled before the accepted call finished draining.',
    );
    expect(target.order).toEqual(['dispose', 'accepted']);
  });

  it('surfaces a disposal failure instead of certifying the drain', async () => {
    await expect(drain(createDrainTarget({ dispose: 'fail' }), 'FAILED_DISPOSAL')).rejects.toThrow(
      'executor release failed',
    );
  });

  it('fails when the accepted call changed its output before disposal', async () => {
    await expect(
      drain(createDrainTarget({ result: divergentOutput() }), 'DIVERGED_OUTPUT'),
    ).rejects.toThrow('changed its output: tokens');
  });

  it('fails when a call after disposal is silently accepted', async () => {
    await expect(
      drain(createDrainTarget({ failOnLaterCall: false }), 'NO_REJECTION'),
    ).rejects.toThrow('was not rejected');
  });
});
