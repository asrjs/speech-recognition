import { describe, expect, it } from 'vitest';

// Import the real extractCrossAttentions — it's a local function in executor.ts,
// so we duplicate it here for focused unit testing.
// The actual integration is tested via the timestamped decoder smoke test.
function extractCrossAttentions(outputs: Record<string, unknown>): unknown[] {
  const entries: { layer: number; tensor: unknown }[] = [];
  for (const [key, value] of Object.entries(outputs)) {
    const match = key.match(/^cross_attentions\.(\d+)$/);
    if (match) {
      entries.push({
        layer: parseInt(match[1]!, 10),
        tensor: value,
      });
    }
  }
  entries.sort((a, b) => a.layer - b.layer);
  return entries.map((e) => e.tensor);
}

describe('Whisper executor cross-attention collection', () => {
  it('extracts cross_attentions from decoder outputs by layer', () => {
    const t0 = { dims: [1, 6, 5, 1500], data: new Float32Array(0) };
    const t1 = { dims: [1, 6, 5, 1500], data: new Float32Array(0) };
    const t2 = { dims: [1, 6, 5, 1500], data: new Float32Array(0) };
    const t3 = { dims: [1, 6, 5, 1500], data: new Float32Array(0) };
    const mockOutputs: Record<string, unknown> = {
      logits: {},
      'present.0.decoder.key': {},
      'present.0.encoder.key': {},
      'cross_attentions.0': t0,
      'cross_attentions.1': t1,
      'cross_attentions.2': t2,
      'cross_attentions.3': t3,
    };

    const crossAttentions = extractCrossAttentions(mockOutputs);
    expect(crossAttentions.length).toBe(4);
    expect(crossAttentions[0]).toBe(t0);
    expect(crossAttentions[1]).toBe(t1);
    expect(crossAttentions[2]).toBe(t2);
    expect(crossAttentions[3]).toBe(t3);
  });

  it('returns empty array when decoder has no cross-attention outputs', () => {
    const mockOutputs: Record<string, unknown> = {
      logits: {},
      'present.0.decoder.key': {},
      'present.0.encoder.key': {},
    };
    const crossAttentions = extractCrossAttentions(mockOutputs);
    expect(crossAttentions).toEqual([]);
  });

  it('sorts cross-attention layers numerically', () => {
    const t0 = { data: new Float32Array(0) };
    const t1 = { data: new Float32Array(0) };
    const t2 = { data: new Float32Array(0) };
    const t3 = { data: new Float32Array(0) };
    const mockOutputs: Record<string, unknown> = {
      'cross_attentions.3': t3,
      'cross_attentions.1': t1,
      'cross_attentions.0': t0,
      'cross_attentions.2': t2,
      logits: {},
    };
    const crossAttentions = extractCrossAttentions(mockOutputs);
    expect(crossAttentions.length).toBe(4);
    expect(crossAttentions[0]).toBe(t0);
    expect(crossAttentions[1]).toBe(t1);
    expect(crossAttentions[2]).toBe(t2);
    expect(crossAttentions[3]).toBe(t3);
  });
});
