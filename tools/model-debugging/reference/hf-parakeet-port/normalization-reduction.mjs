// Diagnostic only: contiguous f32 cascade arithmetic, not a production processor.
// Reference algorithm: PyTorch v2.6.0 SumKernel.cpp (BSD-3-Clause).
// https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cpu/SumKernel.cpp
function validate(values, width) {
  if (!(values instanceof Float32Array) || !values.length || values.some(value => !Number.isFinite(value)))
    throw new Error('Expected nonempty finite Float32Array');
  if (!Number.isInteger(width) || ![8, 16].includes(width)) throw new Error('Expected explicit width 8 or 16');
}

function rowSum(values, start, stride, size) {
  const count = Math.floor(size / 4);
  const power = Math.max(4, Math.floor(Math.ceil(Math.log2(Math.max(1, count))) / 4));
  const step = 2 ** power;
  const acc = Array.from({ length: 4 }, () => new Float32Array(4));
  let index = 0;
  const accumulate = () => {
    for (let column = 0; column < 4; column++)
      acc[0][column] += values[start + (index * 4 + column) * stride];
    index++;
  };
  while (index + step <= count) {
    for (let item = 0; item < step; item++) accumulate();
    for (let level = 1; level < 4; level++) {
      for (let column = 0; column < 4; column++) {
        acc[level][column] += acc[level - 1][column];
        acc[level - 1][column] = 0;
      }
      // Arithmetic mask avoids JavaScript's signed 32-bit shift coercion.
      if (Math.floor(index / 2 ** (level * power)) % step !== 0) break;
    }
  }
  while (index < count) accumulate();
  for (let level = 1; level < 4; level++)
    for (let column = 0; column < 4; column++) acc[0][column] += acc[level][column];
  for (let item = count * 4; item < size; item++) acc[0][0] += values[start + item * stride];
  for (let column = 1; column < 4; column++) acc[0][0] += acc[0][column];
  return acc[0][0];
}

export function cascadeSum(values, width) {
  validate(values, width);
  if (values.length < width) return rowSum(values, 0, 1, values.length);
  const count = Math.floor(values.length / width);
  let sum = 0;
  for (let item = count * width; item < values.length; item++) sum = Math.fround(sum + values[item]);
  for (let lane = 0; lane < width; lane++) sum = Math.fround(sum + rowSum(values, lane, width, count));
  return sum;
}

export function normalizeCandidate(raw, frames, validLength, width) {
  validate(raw, width);
  if (!Number.isSafeInteger(frames) || frames <= 0 || raw.length % frames !== 0 ||
      !Number.isSafeInteger(validLength) || validLength < 1 || validLength > frames)
    throw new Error('Expected complete rows and nonempty integer valid prefix');
  const rows = raw.length / frames;
  const features = new Float32Array(raw.length), mean = new Float32Array(rows), std = new Float32Array(rows);
  const masked = new Float32Array(frames), squared = new Float32Array(frames);
  for (let row = 0; row < rows; row++) {
    const start = row * frames;
    masked.fill(0);
    masked.set(raw.subarray(start, start + validLength));
    mean[row] = Math.fround(cascadeSum(masked, width) / Math.fround(validLength));
    squared.fill(0);
    for (let frame = 0; frame < validLength; frame++) {
      const delta = Math.fround(raw[start + frame] - mean[row]);
      squared[frame] = Math.fround(delta * delta);
    }
    const variance = validLength === 1 ? 0 : Math.fround(cascadeSum(squared, width) / Math.fround(validLength - 1));
    std[row] = Math.fround(Math.fround(Math.sqrt(variance)) + Math.fround(1e-5));
    for (let frame = 0; frame < validLength; frame++)
      features[start + frame] = Math.fround(Math.fround(raw[start + frame] - mean[row]) / std[row]);
  }
  return { features, mean, std };
}
