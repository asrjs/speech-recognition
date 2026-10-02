// Test-only bounded observation. No search/decoding, tensor mutation or GPU readback.
function snapshot(value) {
  if (!value || !['float32', 'int32'].includes(value.type))
    throw new Error('Observer requires CPU float32/int32 tensors');
  const expected = value.type === 'float32' ? Float32Array : Int32Array;
  if (!(value.data instanceof expected)) throw new Error('Observer tensor storage differs');
  const shape = [...value.dims],
    data = Array.from(value.data);
  if (
    !shape.length ||
    shape.some((n) => !Number.isSafeInteger(n) || n < 1) ||
    shape.reduce((a, b) => a * b, 1) !== data.length ||
    data.some((n) => !Number.isFinite(n))
  )
    throw new Error('Observer malformed/nonfinite tensor');
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
  Object.defineProperty(prototype, 'run', {
    ...descriptor,
    value: async function (...args) {
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
    },
  });
  return {
    calls,
    restore() {
      if (active) throw new Error('Cannot restore an in-flight observer');
      if (!restored) {
        Object.defineProperty(prototype, 'run', descriptor);
        restored = true;
      }
    },
  };
}
