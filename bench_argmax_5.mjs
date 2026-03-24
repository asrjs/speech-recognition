import { performance } from 'node:perf_hooks';

// Dummy data
const frameCount = 1000;
const vocabSize = 5000;
const logits = new Float32Array(frameCount * vocabSize);
for (let i = 0; i < logits.length; i++) {
  logits[i] = Math.random();
}

// Original implementation
function argmaxAndSelectedLogProbsOriginal(logits, frameCount, vocabSize) {
  const frameIds = new Array(frameCount).fill(0);
  const selectedLogProbs = new Float32Array(frameCount);

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const rowOffset = frameIndex * vocabSize;
    let bestId = 0;
    let bestValue = Number.NEGATIVE_INFINITY;
    let rowMax = Number.NEGATIVE_INFINITY;

    for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
      const value = logits[rowOffset + vocabIndex] ?? Number.NEGATIVE_INFINITY;
      if (value > bestValue) {
        bestValue = value;
        bestId = vocabIndex;
      }
      if (value > rowMax) {
        rowMax = value;
      }
    }

    let expSum = 0;
    for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
      expSum += Math.exp((logits[rowOffset + vocabIndex] ?? Number.NEGATIVE_INFINITY) - rowMax);
    }

    frameIds[frameIndex] = bestId;
    selectedLogProbs[frameIndex] = bestValue - (rowMax + Math.log(expSum || 1));
  }

  return { frameIds, selectedLogProbs };
}

function argmaxAndSelectedLogProbsFastest(logits, frameCount, vocabSize) {
  const frameIds = new Array(frameCount).fill(0);
  const selectedLogProbs = new Float32Array(frameCount);

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const rowOffset = frameIndex * vocabSize;
    let bestId = 0;
    let bestValue = -Infinity;

    for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
      const value = logits[rowOffset + vocabIndex];
      if (value > bestValue) {
        bestValue = value;
        bestId = vocabIndex;
      }
    }

    let expSum = 0;
    for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
      expSum += Math.exp(logits[rowOffset + vocabIndex] - bestValue);
    }

    frameIds[frameIndex] = bestId;
    selectedLogProbs[frameIndex] = -Math.log(expSum || 1);
  }

  return { frameIds, selectedLogProbs };
}

function runBench() {
  const iters = 200;

  // Warmup
  for (let i = 0; i < 20; i++) {
    argmaxAndSelectedLogProbsOriginal(logits, frameCount, vocabSize);
    argmaxAndSelectedLogProbsFastest(logits, frameCount, vocabSize);
  }

  const startOriginal = performance.now();
  for (let i = 0; i < iters; i++) {
    argmaxAndSelectedLogProbsOriginal(logits, frameCount, vocabSize);
  }
  const endOriginal = performance.now();

  const startFastest = performance.now();
  for (let i = 0; i < iters; i++) {
    argmaxAndSelectedLogProbsFastest(logits, frameCount, vocabSize);
  }
  const endFastest = performance.now();

  console.log(`Original: ${(endOriginal - startOriginal).toFixed(2)}ms`);
  console.log(`Fastest: ${(endFastest - startFastest).toFixed(2)}ms`);
}

runBench();
