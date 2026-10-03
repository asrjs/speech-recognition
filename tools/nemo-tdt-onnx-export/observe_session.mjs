// Test-only bounded observation. No search/decoding, tensor mutation or GPU readback.
const STORAGE = {
  float32: Float32Array,
  int32: Int32Array,
  int64: BigInt64Array,
};

function snapshot(value) {
  if (!value || !(value.type in STORAGE))
    throw new Error('Observer requires CPU float32/int32/int64 tensors');
  const expected = STORAGE[value.type];
  if (!(value.data instanceof expected)) throw new Error('Observer tensor storage differs');
  const shape = [...value.dims];
  if (!shape.length || shape.some((n) => !Number.isSafeInteger(n) || n < 1))
    throw new Error('Observer malformed tensor shape');
  const count = shape.reduce((a, b) => a * b, 1);
  if (count !== value.data.length) throw new Error('Observer tensor count differs');
  if (value.type !== 'int64') {
    const data = Array.from(value.data);
    if (data.some((n) => !Number.isFinite(n)))
      throw new Error('Observer malformed/nonfinite tensor');
    return { dtype: value.type, shape, data };
  }
  // int64 is carried as BigInt. Convert only when every value round-trips
  // exactly through Number, so a silently rounded integer can never enter a
  // receipt. Anything outside the safe range is rejected, not truncated.
  const data = Array.from(value.data, (n) => {
    if (n > BigInt(Number.MAX_SAFE_INTEGER) || n < BigInt(Number.MIN_SAFE_INTEGER))
      throw new Error('Observer int64 value outside exact Number range');
    return Number(n);
  });
  return { dtype: value.type, shape, data };
}

export function observeSessionRuns(InferenceSession, inputNames, outputNames, maxCalls) {
  if (!Number.isSafeInteger(maxCalls) || maxCalls < 1 || maxCalls > 2000)
    throw new Error('Observer requires a bounded positive call count');
  const prototype = InferenceSession?.prototype;
  const descriptor = Object.getOwnPropertyDescriptor(prototype ?? {}, 'run');
  if (typeof descriptor?.value !== 'function' || !descriptor.writable)
    throw new Error('Actual mutable session.run prototype required');
  const original = descriptor.value,
    calls = [];
  let active = 0,
    restored = false;
  const copy = (values, names) =>
    Object.fromEntries(names.map((name) => [name, snapshot(values[name])]));
  const wrapper = async function (...args) {
    const names = this.inputNames;
    if (
      !Array.isArray(names) ||
      names.length !== inputNames.length ||
      !inputNames.every((name) => names.includes(name))
    )
      return Reflect.apply(original, this, args);
    if (active || calls.length >= maxCalls)
      throw new Error('Concurrent or excessive observed calls');
    active++;
    try {
      const feeds = copy(args[0], inputNames);
      const outputs = await Reflect.apply(original, this, args);
      calls.push({ index: calls.length, feeds, outputs: copy(outputs, outputNames) });
      return outputs;
    } finally {
      active--;
    }
  };
  Object.defineProperty(prototype, 'run', { ...descriptor, value: wrapper });
  return {
    calls,
    restore() {
      if (active) throw new Error('Cannot restore an in-flight observer');
      if (restored) return;
      // Chained observers each save the wrapper that was installed before them.
      // Restoring out of order would reinstall an inner wrapper permanently, so
      // only restore when this wrapper is still the installed one. Callers must
      // still restore in reverse installation order.
      if (prototype.run !== wrapper)
        throw new Error('Observer restore order must be reverse installation order');
      Object.defineProperty(prototype, 'run', descriptor);
      restored = true;
    },
  };
}
